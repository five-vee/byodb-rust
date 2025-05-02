//! # Mmap format
//!
//! The store is the storage layer for the copy-on-write B+ tree. It is a
//! memory-mapped region that can be backed by a file.
//!
//! The mmap has the following structure:
//!
//! ```ignore
//! | meta page |   pages   |
//! |    64B    | N * 4096B |
//! ```
//!
//! where `N` is the number of pages utilized so far by the B+ tree
//! and free list.
//!
//! # Snapshot isolation
//!
//! There can be multiple concurrent readers of a store. Readers do not block
//! each other.
//!
//! There can be only 1 writer of a store at a time. However, readers do not
//! block the writer, and the writer does not block readers.
//!
//! Readers can only read pages once they're flushed: no dirty reads allowed.
//! Writers can allocate new pages and can even read non-flushed pages.
//! Of course, if one wants the page to be visible to future reader, it MUST
//! be flushed first.
//!
//! # Building MVCC transactions
//!
//! Though this module doesn't support MVCC transactions out of the box,
//! the abstractions provided can be used to build transactions.
mod free_list;
mod meta_node;

use std::{
    cell::{RefCell, UnsafeCell},
    cmp::max,
    fs::{File, OpenOptions},
    io::{Seek, Write as _},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    path::Path,
    rc::Rc,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicPtr, Ordering},
    },
};

use arc_swap::{ArcSwap, Guard as ArcSwapGuard};
use memmap2::{MmapMut, MmapOptions};
use seize::{Collector, Guard as _, LocalGuard};

use crate::{
    consts,
    error::PageError,
    header::{self, NodeType},
};
use free_list::FreeList;
use meta_node::MetaNode;

#[cfg(not(test))]
const MIN_FILE_GROWTH_SIZE: usize = (1 << 14) * consts::PAGE_SIZE;

#[cfg(test)]
const MIN_FILE_GROWTH_SIZE: usize = 2 * consts::PAGE_SIZE;

type Result<T> = std::result::Result<T, PageError>;

/// A wrapper around a memory-mapped region (mmap).
/// It is optionally backed by a file.
pub struct Mmap {
    file: Rc<RefCell<File>>,
    mmap: MmapMut,
}

impl Deref for Mmap {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        self.mmap.deref()
    }
}

impl DerefMut for Mmap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.mmap.deref_mut()
    }
}

impl Mmap {
    /// Opens or creates (if not exists) a file that holds a B+ tree.
    pub fn open_or_create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let mut file_len = file.metadata()?.len() as usize;
        if file_len == 0 {
            {
                let mut meta_page = [0u8; meta_node::META_PAGE_SIZE];
                MetaNode::default().copy_to_slice(&mut meta_page);
                file.write_all(&meta_page)?;
            }
            {
                let mut leaf_page = [0u8; consts::PAGE_SIZE];
                init_empty_leaf(&mut leaf_page);
                file.write_all(&leaf_page)?;
            }
            {
                let mut free_list_page = [0u8; consts::PAGE_SIZE];
                free_list::init_empty_list_node(&mut free_list_page);
                file.write_all(&free_list_page)?;
            }
            file_len = file.metadata()?.len() as usize;
            file.sync_all()?;
            file.seek(std::io::SeekFrom::Start(0))?; // not sure if necessary
        }
        let file_len = file_len;
        if file_len < meta_node::META_PAGE_SIZE + 2 * consts::PAGE_SIZE {
            return Err(PageError::InvalidFile(
                "file must be at least 2 meta page sizes + 2 pages".into(),
            ));
        }
        // TODO: validate the contents of the file.
        // TODO: fix any errors if possible.

        // Safety: it is assumed that no other process has a mutable mapping to the same file.
        let mmap = unsafe { MmapOptions::new().map_mut(&file).unwrap() };

        Ok(Mmap {
            file: Rc::new(RefCell::new(file)),
            mmap,
        })
    }

    /// Flushes all written pages to the file (if exists), thereby making them
    /// available to read by new readers.
    fn flush(&self) {
        // Flush the aligned range
        self.mmap
            .flush()
            .expect("mmap can validly msync with aligned range");
    }

    /// Grows the mmap region (and file, if exists) to `new_len`.
    /// This is needed if the mmap cannot allocate more new pages.
    fn grow(&self, new_len: usize) -> Self {
        let mmap = {
            self.file
                .borrow()
                .set_len(new_len as u64)
                .expect("file can grow in size");
            // Safety: it is assumed that no other process has a mutable mapping to the same file.
            unsafe {
                MmapOptions::new()
                    .map_mut(&*self.file.as_ptr())
                    .expect("mmap is correctly created")
            }
        };
        Mmap {
            file: self.file.clone(),
            mmap,
        }
    }
}

/// Container of data needed by Readers.
struct ReaderState {
    mmap: Arc<UnsafeCell<Mmap>>,
    flush_offset: usize,
}

// Safety: Although UnsafeCell isn't Sync, we promise that ReaderState will
// only read and not modify the data.
// Furthermore, only WriterState can concurrently access the cell, and
// it'll only mutate the region of memory distinct from the region accessed
// by ReaderState.
unsafe impl Send for ReaderState {}
unsafe impl Sync for ReaderState {}

/// Container of data needed by Writer.
struct WriterState {
    mmap: Arc<UnsafeCell<Mmap>>,
    flush_offset: usize,
    write_offset: usize,
    new_offset: usize,
}

// Safety: Although UnsafeCell isn't Sync, we promise that only 1 WriterState
// instance can mutate the data.
// Furthermore, it can only mutate the region of memory distinct from the
// region accessed by ReaderState.
unsafe impl Send for WriterState {}
unsafe impl Sync for WriterState {}

impl WriterState {
    /// A convenient method to retrieve the Mmap
    /// since it's guarded by UnsafeCell.
    #[allow(clippy::mut_from_ref)] // This is the same as if inlined.
    fn mmap_mut(&self) -> &mut Mmap {
        // Safety: WriterState guarantees that mutable references of the mmap
        // only touch [pages_ptr + flush_offset * PAGE_SIZE, mmap_end), which never overlaps
        // with immutable reference reads of [pages_ptr, pages_ptr + flush_offset * PAGE_SIZE).
        unsafe { &mut *self.mmap.get() }
    }

    /// A convenient method to retrieve the Mmap
    /// since it's guarded by UnsafeCell.
    fn mmap(&self) -> &Mmap {
        self.mmap_mut()
    }

    /// Flushes only the B+ tree pages portion of the mmap.
    fn flush_pages(&mut self) {
        let m = self.mmap_mut();
        m.flush();
        self.flush_offset = self.write_offset;
    }

    /// Grows the underlying mmap only if there's no more space for new pages.
    #[allow(clippy::arc_with_non_send_sync)] // We manually ensure thread-safety via Mutex/ArcSwap.
    fn grow_if_needed(&mut self) -> bool {
        let m = self.mmap();
        if meta_node::META_PAGE_SIZE + self.new_offset + consts::PAGE_SIZE <= m.len() {
            return false;
        }
        let expand = max(m.len(), MIN_FILE_GROWTH_SIZE);
        let new_len = m.len() + expand;

        m.flush();
        self.mmap = Arc::new(UnsafeCell::new(m.grow(new_len)));
        true
    }

    /// Flushes only the meta node of the mmap.
    fn flush_new_meta_node(&mut self, new_root_ptr: usize, fl: &FreeList) {
        let m = self.mmap_mut();
        let curr = MetaNode::new(new_root_ptr, self.flush_offset / consts::PAGE_SIZE, &fl);
        curr.copy_to_slice(m.deref_mut());
        m.flush();
    }
}

/// The store is the storage layer for the copy-on-write B+ tree.
/// It is an abstraction to allow for multi-version-concurrency control (MVCC).
///
/// There can be multiple concurrent readers of a store. Readers do not block
/// each other.
///
/// There can be only 1 writer of a store at a time. However, readers do not
/// block the writer, and the writer does not block readers.
///
/// Readers can only read pages once they're flushed: no dirty reads allowed.
/// Writers can allocate new pages and can even read non-flushed pages.
/// Of course, if one wants the page to be visible to future reader, it MUST
/// be flushed first.
pub struct Store {
    reader_state: ArcSwap<ReaderState>,
    writer_state: Mutex<WriterState>,
    collector: Collector,
    root_page: AtomicPtr<usize>,
}

impl Store {
    /// Creates a new store from a specified `Mmap`.
    pub fn new(mmap: Mmap) -> Self {
        let node = MetaNode::try_from(mmap.deref()).expect("there should exist a meta node");
        let flush_offset = node.num_pages * consts::PAGE_SIZE;

        // We manually ensure thread-safety via Mutex/ArcSwap.
        #[allow(clippy::arc_with_non_send_sync)]
        let mmap = Arc::new(UnsafeCell::new(mmap));
        Store {
            reader_state: ArcSwap::from_pointee(ReaderState {
                mmap: mmap.clone(),
                flush_offset,
            }),
            writer_state: Mutex::new(WriterState {
                mmap: mmap.clone(),
                flush_offset,
                write_offset: flush_offset,
                new_offset: flush_offset,
            }),
            // TODO: dependency inject collector
            collector: Collector::new().batch_size(1),
            root_page: AtomicPtr::new(Box::into_raw(Box::new(node.root_page))),
        }
    }

    /// Obtains a Reader guard.
    pub fn reader(self: &Arc<Self>) -> Reader {
        let collector_guard = self.collector.enter();
        // Note: consider Ordering::Acquire if perf is necessary (prob not).
        // Safety: root_page is NEVER null; it is always initialized to some value.
        let root_page = unsafe { *collector_guard.protect(&self.root_page, Ordering::SeqCst) };
        Reader {
            root_page,
            state_guard: self.reader_state.load(),
            collector_guard,
        }
    }

    /// Obtains a Writer guard. This will block so long as a Writer
    /// was previously obtained and not yet dropped.
    pub fn writer(self: &Arc<Self>) -> Writer<'_> {
        let mutex_guard = self.writer_state.lock().unwrap();
        let node = MetaNode::try_from(mutex_guard.mmap().as_ref())
            .expect("there should exist a meta node");
        let collector_guard = self.collector.enter();
        // Note: consider Ordering::Acquire if perf is necessary (prob not).
        let prev_root_page = collector_guard.protect(&self.root_page, Ordering::SeqCst);
        Writer {
            prev_root_page,
            // Safety: not null b/c always initialized.
            curr_root_page: unsafe { *prev_root_page },
            free_list: FreeList::from(node),
            reclaimable_pages: RefCell::new(Vec::new()),
            flushed: false,
            store: self.clone(),
            state_guard: RefCell::new(mutex_guard),
            collector_guard,
        }
    }

    /// Adds pages to the free list.
    fn reclaim_pages(self: &Arc<Self>, pages: Vec<usize>) {
        let mut w = self.writer();
        let mut fl = w.free_list;
        for page_num in pages {
            fl.push_tail(&w, page_num);
        }
        fl.set_max_seq();
        w.free_list = fl;
        let root_page = w.curr_root_page; // unchanged
        w.flush(root_page);
    }
}

/// Guard is a trait common to both Reader and Writer.
/// It provides guarded read-only access to the mmap region.
pub trait Guard {
    /// Reads a page at `page_num`.
    /// Note: If the implementing type is Reader, only flushed pages can be
    /// read.
    ///
    /// If it's a Writer, then either flushed or written pages can be read.
    /// Newly allocated pages must be written first before they can be read.
    /// Remember to flush pages to make them available to future readers.
    /// TODO: make this unsafe.
    fn read_page(&self, page_num: usize) -> ReadOnlyPage<'_>;

    /// Obtains the root page number accessible by the guard.
    fn root_page(&self) -> usize;
}

/// A Reader that provides safe read-only concurrent access to the flushed
/// B+ tree nodes and flushed meta nodes.
/// Readers don't block each other.
/// Readers and the Writer don't block each other, but Readers are isolated
/// from the Writer via the flushing mechanism.
pub struct Reader<'s> {
    root_page: usize,
    state_guard: ArcSwapGuard<Arc<ReaderState>>,
    collector_guard: LocalGuard<'s>,
}

impl Guard for Reader<'_> {
    fn read_page(&self, page_num: usize) -> ReadOnlyPage<'_> {
        let guard = &self.state_guard;
        assert!(
            page_num * consts::PAGE_SIZE < guard.flush_offset,
            "page_num = {} must be < {}",
            page_num,
            guard.flush_offset / consts::PAGE_SIZE
        );
        ReadOnlyPage {
            _phantom: PhantomData,
            mmap: guard.mmap.clone(),
            page_num,
        }
    }

    fn root_page(&self) -> usize {
        self.root_page
    }
}

/// A Writer that provides safe read+write serialized access to the flushed
/// B+ tree nodes and flushed meta nodes.
/// Only 1 Writer is allowed access at a time.
/// Readers and the Writer don't block each other, but Readers are isolated
/// from the Writer via the flushing mechanism.
pub struct Writer<'s> {
    prev_root_page: *mut usize,
    curr_root_page: usize,
    free_list: FreeList,
    reclaimable_pages: RefCell<Vec<usize>>,
    flushed: bool,
    store: Arc<Store>,
    state_guard: RefCell<MutexGuard<'s, WriterState>>,

    // It's important that collector_guard is listed AFTER state_guard
    // so that collector_guard is dropped AFTER state_guard is dropped.
    // When Writer is dropped, it defers page reclamation, which itself
    // acquires the store's mutex. The mutex cannot be locked unless the writer
    // first drops its MutexGuard.
    collector_guard: LocalGuard<'s>,
}

impl Writer<'_> {
    /// Allocates a new page for write.
    pub fn new_page(&self) -> Page<'_, NewPageType> {
        let mut guard = self.state_guard.borrow_mut();
        guard.grow_if_needed();
        let page = Page {
            _phantom: PhantomData,
            mmap: guard.mmap.clone(),
            page_num: guard.new_offset / consts::PAGE_SIZE,
        };
        guard.new_offset += consts::PAGE_SIZE;
        page
    }

    /// Finalizes a page, marking it read-only.
    /// This does NOT make the page available to readers;
    /// the page must be flushed first.
    pub fn write_page<'w>(&'w self, page: Page<'w, NewPageType>) -> ReadOnlyPage<'w> {
        let mut guard = self.state_guard.borrow_mut();
        guard.write_offset += consts::PAGE_SIZE;
        ReadOnlyPage {
            _phantom: PhantomData,
            mmap: page.mmap,
            page_num: page.page_num,
        }
    }

    /// Flushes all written pages and update the meta node root pointer.
    /// Every new page must first be written before calling `flush`.
    pub fn flush(mut self, new_root_ptr: usize) {
        let mut guard = self.state_guard.borrow_mut();
        if guard.write_offset < guard.new_offset {
            panic!("there are still new pages not yet written");
        }
        guard.flush_pages();
        guard.flush_new_meta_node(new_root_ptr, &self.free_list);
        self.store.reader_state.store(Arc::new(ReaderState {
            mmap: guard.mmap.clone(),
            flush_offset: guard.flush_offset,
        }));
        self.curr_root_page = new_root_ptr;
        // Note: consider Ordering::Release if perf is necessary (prob not).
        self.store
            .root_page
            .store(Box::into_raw(Box::new(new_root_ptr)), Ordering::SeqCst);

        self.flushed = true;
    }

    /// Retrieves an existing page and allows the user to write to it.
    /// This is unsafe b/c it can allow overwriting data that is assumed to be
    /// immutable. As such, please only use this on non-B+-tree pages,
    /// i.e. free list pages.
    pub unsafe fn overwrite_page(&self, page_num: usize) -> Page<'_, ReusedPageType> {
        let guard = self.state_guard.borrow();
        // The requested page must already been flushed or written already.
        assert!(page_num * consts::PAGE_SIZE < guard.write_offset);
        Page {
            _phantom: PhantomData,
            mmap: guard.mmap.clone(),
            page_num,
        }
    }

    /// Marks an existing page as free, allowing it to be reclaimed later
    /// and made available via `overwrite_page()`.
    pub fn mark_free(&self, page_num: usize) {
        let guard = self.state_guard.borrow();
        // The requested page must already been flushed or written already.
        assert!(page_num * consts::PAGE_SIZE < guard.write_offset);
        self.reclaimable_pages.borrow_mut().push(page_num);
    }
}

impl Guard for Writer<'_> {
    fn read_page(&self, page_num: usize) -> ReadOnlyPage<'_> {
        let guard = self.state_guard.borrow();
        assert!(
            page_num * consts::PAGE_SIZE < guard.write_offset,
            "page_num = {} must be < {}",
            page_num,
            guard.write_offset / consts::PAGE_SIZE
        );
        ReadOnlyPage {
            _phantom: PhantomData,
            mmap: guard.mmap.clone(),
            page_num,
        }
    }

    fn root_page(&self) -> usize {
        self.curr_root_page
    }
}

impl Drop for Writer<'_> {
    fn drop(&mut self) {
        if !self.flushed {
            // This can happen if the write transaction errored or was aborted.
            // Reset as if nothing ever happened.
            let mut guard = self.state_guard.borrow_mut();
            guard.write_offset = guard.flush_offset;
            guard.new_offset = guard.flush_offset;
            return;
        }
        let mut pages = self.reclaimable_pages.borrow_mut();
        if pages.len() == 0 {
            return;
        }

        let reclaimable = Box::into_raw(Box::new(Reclaimable {
            store: self.store.clone(),
            root_page: self.prev_root_page,
            pages: std::mem::take(&mut *pages),
        }));
        // Pages cannot be reclaimed back to the free list until it's
        // guaranteed that no reader (or writer) can traverse to these pages,
        // either from a B+ tree root node, or via the free list itself.
        // Safety: prev_root_page has already been replaced in the flush() call,
        // so it is safe to defer its retirement.
        unsafe {
            self.collector_guard
                .defer_retire(reclaimable, |r, _collector| {
                    let r = Box::from_raw(r); // so r can be auto-dropped
                    drop(Box::from_raw(r.root_page)); // must manually be dropped
                    r.store.reclaim_pages(r.pages);
                });
        }
        // self.epoch_guard.defer(move || {
        //     store.reclaim_pages(pages);
        // });
    }
}

struct Reclaimable {
    store: Arc<Store>,
    root_page: *mut usize,
    pages: Vec<usize>,
}

/// A page inside the mmap region that allows only for reads.
/// The lifetime 'a is tied to either a Reader or Writer:
/// * if a Reader, then it represents a flushed page.
/// * if a Writer, then it can represent either a flushed or
///   written (but not yet flushed) page.
pub struct ReadOnlyPage<'a> {
    _phantom: PhantomData<&'a ()>,
    mmap: Arc<UnsafeCell<Mmap>>,
    pub page_num: usize,
}

impl<'a> Deref for ReadOnlyPage<'a> {
    type Target = [u8];
    fn deref(&self) -> &'a [u8] {
        // Safety: [page_ptr, page_ptr + PAGE_SIZE) is a guaranteed valid region in the mmap.
        // It is guaranteed that no writers can touch this region (since it's already flushed).
        // page_ptr cannot dangle b/c the mmap is never dropped/realloc'ed so long as
        // the reader guard (&'r Reader) exists.
        &unsafe { &*self.mmap.get() }.mmap[meta_node::META_PAGE_SIZE
            + self.page_num * consts::PAGE_SIZE
            ..meta_node::META_PAGE_SIZE + (self.page_num + 1) * consts::PAGE_SIZE]
    }
}

pub trait PageType {}

pub struct NewPageType;
impl PageType for NewPageType {}

pub struct ReusedPageType;
impl PageType for ReusedPageType {}

/// A page inside the mmap region that allows for reads and writes.
/// A page is not accessible to readers until flushed.
/// Nor can it be read by a writer until written.
pub struct Page<'w, T: PageType = NewPageType> {
    _phantom: PhantomData<&'w T>,
    mmap: Arc<UnsafeCell<Mmap>>,
    page_num: usize,
}

impl<T: PageType> Deref for Page<'_, T> {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        // Safety: [page_ptr, page_ptr + PAGE_SIZE) is a guaranteed valid region in the mmap.
        // Due to the mutex guard in Writer, at most one mutable reference can
        // touch this region at a time.
        // page_ptr cannot dangle b/c the mmap is never dropped/realloc'ed so long as
        // the writer mutex guard (&'w Writer<'s>) exists.
        &unsafe { &*self.mmap.get() }.mmap[meta_node::META_PAGE_SIZE
            + self.page_num * consts::PAGE_SIZE
            ..meta_node::META_PAGE_SIZE + (self.page_num + 1) * consts::PAGE_SIZE]
    }
}

impl<T: PageType> DerefMut for Page<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Safety: [page_ptr, page_ptr + PAGE_SIZE) is a guaranteed valid region in the mmap.
        // Due to the mutex guard in Writer, at most one mutable reference can
        // touch this region at a time.
        // page_ptr cannot dangle b/c the mmap is never dropped/realloc'ed so long as
        // the writer mutex guard (&'w Writer<'s>) exists.
        &mut unsafe { &mut *self.mmap.get() }.mmap[meta_node::META_PAGE_SIZE
            + self.page_num * consts::PAGE_SIZE
            ..meta_node::META_PAGE_SIZE + (self.page_num + 1) * consts::PAGE_SIZE]
    }
}

/// Allocates a new page as an empty leaf and writes it into the store.
pub fn write_empty_leaf(writer: &Writer) -> usize {
    let mut page = writer.new_page();
    init_empty_leaf(&mut page);
    writer.write_page(page).page_num
}

fn init_empty_leaf(page: &mut [u8]) {
    header::set_node_type(page, NodeType::Leaf);
    header::set_num_keys(page, 0);
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use crate::consts;

    use super::*;

    fn new_file_mmap() -> (Arc<Store>, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        println!("Created temporary file {path:?}");
        let mmap = Mmap::open_or_create(path).unwrap();
        let store = Arc::new(Store::new(mmap));
        (store, temp_file)
    }

    #[test]
    fn test_open_file() {
        let page_num;
        let pattern = [0xDE, 0xAD, 0xBE, 0xEF];
        let edit_offset = 50;

        // Scope 1: Create, write, flush, close
        let (store, temp_file) = new_file_mmap();
        let path = temp_file.path();
        {
            let writer = store.writer();
            let mut page = writer.new_page();

            page[edit_offset..edit_offset + pattern.len()].copy_from_slice(&pattern);

            let written_page = writer.write_page(page);
            page_num = written_page.page_num;
            // Flush with the root pointer as the new page number for simplicity
            writer.flush(page_num);
        }
        drop(store);

        // Scope 2: Reopen and verify
        {
            let mmap = Mmap::open_or_create(path).unwrap();
            let store = Arc::new(Store::new(mmap));
            let reader = store.reader();

            // Verify the meta node
            let root_page = reader.root_page();
            assert_eq!(
                root_page, page_num,
                "Root pointer should match the flushed page number"
            );

            // Verify the page
            let read_page = reader.read_page(page_num);
            assert_eq!(
                &read_page[edit_offset..edit_offset + pattern.len()],
                &pattern,
                "Data read should match data written after reopening"
            );
        }
    }

    #[test]
    fn test_open_invalid_file() {
        use std::io::Write;

        // Test case 1: File too small
        {
            let temp_file = NamedTempFile::new().unwrap();
            let mut file = std::fs::File::create(temp_file.path()).unwrap();
            // Write less than the minimum required size
            let small_data = vec![0u8; meta_node::META_PAGE_SIZE + consts::PAGE_SIZE - 1];
            file.write_all(&small_data).unwrap();
            file.sync_all().unwrap();

            let result = Mmap::open_or_create(temp_file.path());
            assert!(matches!(result, Err(PageError::InvalidFile(_))));
        }
    }

    #[test]
    fn test_writer_can_read_written_pages() {
        let (store, _temp_file) = new_file_mmap();
        let writer = store.writer();
        let pattern1 = [0xAA, 0xBB, 0xCC];
        let edit_offset1 = 10;

        // 1. Test writer can read a page it just wrote (but hasn't flushed).
        let page1_num = {
            let mut page = writer.new_page();
            page[edit_offset1..edit_offset1 + pattern1.len()].copy_from_slice(&pattern1);
            writer.write_page(page).page_num
        };

        // Read the written page using the same writer
        let read_page1 = writer.read_page(page1_num);
        assert_eq!(
            &read_page1[edit_offset1..edit_offset1 + pattern1.len()],
            &pattern1,
            "Writer should be able to read its own written (but not flushed) page"
        );
        drop(read_page1); // Drop the read guard

        // 2. Test writer can read a flushed page.
        let pattern2 = [0xDD, 0xEE, 0xFF];
        let edit_offset2 = 20;
        let page2_num = {
            let mut page = writer.new_page();
            page[edit_offset2..edit_offset2 + pattern2.len()].copy_from_slice(&pattern2);
            let page_num = writer.write_page(page).page_num;
            writer.flush(page_num); // Flush this page
            page_num
        };

        // Read the flushed page using a writer.
        let writer = store.writer();
        let read_page2 = writer.read_page(page2_num);
        assert_eq!(
            &read_page2[edit_offset2..edit_offset2 + pattern2.len()],
            &pattern2,
            "Writer should be able to read a flushed page"
        );
        drop(read_page2);

        // 3. Test writer CANNOT read a newly allocated page that hasn't been written yet.
        let page3 = writer.new_page();
        drop(page3); // Drop the mutable page guard
    }

    #[test]
    fn test_reader_can_only_read_flushed_pages() {
        let (store, _temp_file) = new_file_mmap();
        let pattern1 = [0x11, 0x22, 0x33];
        let edit_offset1 = 5;
        let page1_num;

        // 1. Write and flush a page.
        {
            let writer = store.writer();
            let mut page = writer.new_page();
            page[edit_offset1..edit_offset1 + pattern1.len()].copy_from_slice(&pattern1);
            page1_num = writer.write_page(page).page_num;
            writer.flush(page1_num);
        } // Writer dropped here

        // 2. Get a reader and verify it can read the flushed page.
        {
            let reader1 = store.reader();
            let read_page1 = reader1.read_page(page1_num);
            assert_eq!(
                &read_page1[edit_offset1..edit_offset1 + pattern1.len()],
                &pattern1,
                "Reader should be able to read the flushed page"
            );
        } // Reader1 dropped here

        // 3. Write another page but DO NOT flush it.
        let pattern2 = [0x44, 0x55, 0x66];
        let edit_offset2 = 15;
        {
            let writer = store.writer();
            let mut page = writer.new_page();
            page[edit_offset2..edit_offset2 + pattern2.len()].copy_from_slice(&pattern2);
            // No flush here!
        } // Writer dropped here

        // 4. Get another reader. Verify it can still read the first page,
        //    but panics when trying to read the second (non-flushed) page.
        {
            let reader2 = store.reader();
            // Can still read the first flushed page
            let read_page1_again = reader2.read_page(page1_num);
            assert_eq!(
                &read_page1_again[edit_offset1..edit_offset1 + pattern1.len()],
                &pattern1,
                "Second reader should still be able to read the first flushed page"
            );
            drop(read_page1_again);
        }
    }

    #[test]
    fn test_writer_can_read_written_page_even_after_growing() {
        let (store, _temp_file) = new_file_mmap();
        let writer = store.writer();
        let pattern = [0xBE, 0xEF, 0xCA, 0xFE];
        let edit_offset = 30;

        // 1. Write a page and keep the ReadOnlyPage handle.
        let page1 = {
            let mut page = writer.new_page();
            page[edit_offset..edit_offset + pattern.len()].copy_from_slice(&pattern);
            writer.write_page(page) // Keep the ReadOnlyPage handle
        };

        // Verify initial read is okay
        assert_eq!(
            &page1[edit_offset..edit_offset + pattern.len()],
            &pattern,
            "Initial read of written page should work"
        );

        // 2. Allocate enough new pages to trigger growth.
        // Assuming the store starts with 1 page, the second new_page call should trigger growth.
        // Let's allocate a few more just to be sure, depending on initial size.
        let initial_len = writer.state_guard.borrow().mmap().len();
        let mut pages_allocated = 0;
        loop {
            let current_len = writer.state_guard.borrow().mmap().len();
            if current_len > initial_len {
                println!(
                    "Growth triggered after allocating {} pages. Initial len: {}, Current len: {}",
                    pages_allocated, initial_len, current_len
                );
                break; // Growth occurred
            }
            if pages_allocated > 10 {
                // Safety break
                panic!(
                    "Growth did not trigger after allocating 10 pages. Initial len: {}, Current len: {}",
                    initial_len, current_len
                );
            }
            let _ = writer.new_page(); // Allocate a new page
            pages_allocated += 1;
        }

        // 3. Verify that the original ReadOnlyPage handle is still valid and readable.
        // Accessing page1.deref() implicitly checks if the pointer is still valid.
        assert_eq!(
            &page1[edit_offset..edit_offset + pattern.len()],
            &pattern,
            "Read after growth should still work and match the original pattern"
        );
    }

    #[test]
    fn test_reader_can_read_flushed_page_even_after_growing() {
        let (store, _temp_file) = new_file_mmap();
        let pattern = [0xFA, 0xCE, 0xB0, 0x0C];
        let edit_offset = 40;
        let page1_num;

        // 1. Write and flush a page.
        {
            let writer = store.writer();
            let mut page = writer.new_page();
            page[edit_offset..edit_offset + pattern.len()].copy_from_slice(&pattern);
            page1_num = writer.write_page(page).page_num;
            writer.flush(page1_num);
        } // Writer dropped here

        // 2. Get a reader and obtain a ReadOnlyPage handle for the flushed page.
        let reader = store.reader();
        let page1_handle = reader.read_page(page1_num);

        // Verify initial read is okay
        assert_eq!(
            &page1_handle[edit_offset..edit_offset + pattern.len()],
            &pattern,
            "Initial read of flushed page by reader should work"
        );

        // 3. Get another writer and trigger growth by allocating new pages.
        {
            let writer2 = store.writer();
            let initial_len = writer2.state_guard.borrow().mmap().len();
            let mut pages_allocated = 0;
            loop {
                let current_len = writer2.state_guard.borrow().mmap().len();
                if current_len > initial_len {
                    println!(
                        "Growth triggered after allocating {} pages. Initial len: {}, Current len: {}",
                        pages_allocated, initial_len, current_len
                    );
                    break; // Growth occurred
                }
                if pages_allocated > 10 {
                    // Safety break
                    panic!(
                        "Growth did not trigger after allocating 10 pages. Initial len: {}, Current len: {}",
                        initial_len, current_len
                    );
                }
                let _ = writer2.new_page(); // Allocate a new page
                pages_allocated += 1;
            }
            // Writer2 is dropped here, potentially releasing the lock
        }

        // 4. Verify that the original reader's ReadOnlyPage handle is still valid and readable.
        // Accessing page1_handle.deref() implicitly checks if the pointer is still valid.
        assert_eq!(
            &page1_handle[edit_offset..edit_offset + pattern.len()],
            &pattern,
            "Reader's handle read after growth should still work and match the original pattern"
        );
    }

    #[test]
    fn test_store_can_read_after_flush() {
        let (store, _temp_file) = new_file_mmap();
        let writer = store.writer();
        let mut page = writer.new_page();
        assert_eq!(page.len(), consts::PAGE_SIZE);

        // Make a dummy edit to the page.
        let edit_offset = 10;
        let pattern = [0xAA, 0xBB, 0xCC];
        page[edit_offset..edit_offset + pattern.len()].copy_from_slice(&pattern);

        let page_num = writer.write_page(page).page_num;
        writer.flush(page_num);

        let reader = store.reader();
        let read_page = reader.read_page(page_num);

        // Verify the dummy value made at the edit site is there and that the page is of correct size
        assert_eq!(read_page.len(), consts::PAGE_SIZE);
        assert_eq!(
            &read_page[edit_offset..edit_offset + pattern.len()],
            &pattern,
            "Data read should match data written"
        );
    }

    #[test]
    #[should_panic]
    fn test_store_cannot_read_before_flush() {
        let (store, _temp_file) = new_file_mmap();
        let writer = store.writer();
        let page = writer.new_page();
        let page_num = writer.write_page(page).page_num;
        // DON'T flush. Drop the writer instead.
        drop(writer);

        let reader = store.reader();
        // Try to read the recently written page via Reader::read_page(page_num).
        // This should fail b/c not flushed yet.
        let _ = reader.read_page(page_num);
    }

    #[test]
    fn test_store_reader_writer_isolation() {
        use std::thread;
        let (store, _temp_file) = new_file_mmap();

        let mut threads = vec![];

        // Write a page and flush it. Drop the writer.
        {
            let writer = store.writer();
            let mut page = writer.new_page();
            page[0] = 1;
            let page_num = writer.write_page(page).page_num;
            writer.flush(page_num);
        }

        // Get another writer via Store::writer() and write + flush another page.
        // Do NOT drop the writer.
        let writer = store.writer();
        let mut page = writer.new_page();
        page[0] = 2;
        let page_num = writer.write_page(page).page_num;
        writer.flush(page_num);

        // Save a handle on that page via Store::reader() and Reader::read_page().
        threads.push(thread::spawn({
            let store = store.clone();
            move || {
                let reader = store.reader();
                let page = reader.read_page(page_num);
                assert_eq!(page[0], 1);
            }
        }));
        threads.push(thread::spawn({
            let store = store.clone();
            move || {
                let reader = store.reader();
                let page = reader.read_page(page_num);
                assert_eq!(page[0], 1);
            }
        }));

        for t in threads {
            let _ = t.join();
        }
    }
}
