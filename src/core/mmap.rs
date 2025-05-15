//! [`Mmap`] represents a file-backed memory-mapped region that serves as the
//! underlying data layer of the B+ [`crate::core::tree::Tree`].
//!
//! # Mmap format
//!
//! The store is the storage layer for the copy-on-write B+
//! [`crate::core::tree::Tree`]. It is a memory-mapped region that is backed by
//! a file.
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
//! [Multiversion concurrency control](https://en.wikipedia.org/wiki/Multiversion_concurrency_control)
//! (MVCC) is a non-locking concurrency control method to allow concurrent
//! access in the database.
//!
//! A [`Store`] provides guarded concurrent access to the underlying [`Mmap`].
//! [`Store::reader`] returns a [`Reader`], which is essentially a read-only
//! transaction that has global access to a snapshot of the underlying data.
//! There can be multiple concurrent readers of a store. [`Reader`]s do not
//! block each other.
//!
//! [`Store::writer`] returns a [`Writer`], which is a read-write transaction.
//! There can be only 1 [`Writer`] of a store at a time (achieved via
//! [`Mutex`]). However, readers do not block the writer, and the writer does
//! not block readers. This is achieved via the [`arc_swap`] crate (more on
//! this later).
//!
//! [`Reader`]s can only read pages once they're flushed: no dirty reads
//! allowed. [`Writer`]s can allocate new pages and can even read non-flushed
//! pages. Of course, if one wants the page to be visible to future reader, it
//! **MUST** be flushed first, i.e. [`Writer::flush`].
//!
//! A transaction can be aborted instead by [`Writer::abort`]. Though this
//! doesn't revert any dirty pages, those pages are inaccessible b/c the
//! meta node won't be updated unless flushed.
//!
//! We use the crate [`arc_swap`] for multi-buffering of the memory-mapped
//! file. Through a [`arc_swap::Guard`], a [`Reader`] will still hold reference
//! to the old mmap even if a [`Writer`] has completely replaced the mmap with
//! a newer one that's larger in size due to file growth.
//!
//! ## Meta page
//!
//! The meta page is crucial for write transaction atomicity and durability
//! guarantees. It acts similarly to a memory barrier for accessing the tree
//! (through the root pointer), or the free list.
//!
//! ## Garbage collection
//!
//! The [`Writer`] is responsible for collecting garbage pages and making them
//! available for re-use.
//!
//! Pages that are marked free are put in a pending "bag". When it's guaranteed
//! that no readers can traverse from their B+ tree root page to a page, that
//! page can be safely reclaimed into the [`FreeList`].
//!
//! This is performed through a reclamation mechanism that, relies on the
//! [`seize`] crate. The benefit of this over a traditional epoch-based
//! reclamation (EBR) process (such as
//! [`crossbeam_epoch`](https://docs.rs/crossbeam-epoch)) is that EBR lacks
//! predictability or even guarantees that garbage will be reclaimed in a
//! certain time period. Periodic checks are often required to determine when
//! it is safe to free memory. Unlike EBR, [`seize`] can be more easily
//! configured when it can reclaim.
mod free_list;
mod meta_node;

use std::{
    cell::{RefCell, UnsafeCell},
    cmp::max,
    fs::{File, OpenOptions},
    io::{Seek, Write as _},
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

use crate::core::{
    consts,
    error::MmapError,
    header::{self, NodeType},
};
use free_list::FreeList;
use meta_node::MetaNode;

// 64MB
#[cfg(not(test))]
pub(crate) const DEFAULT_MIN_FILE_GROWTH_SIZE: usize = (1 << 14) * consts::PAGE_SIZE;

// 8KB
#[cfg(test)]
pub(crate) const DEFAULT_MIN_FILE_GROWTH_SIZE: usize = 2 * consts::PAGE_SIZE;

type Result<T> = std::result::Result<T, MmapError>;

/// A wrapper around a memory-mapped region (mmap).
/// It is backed by a file.
pub struct Mmap {
    min_file_growth_size: usize,
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
    pub fn open_or_create<P: AsRef<Path>>(path: P, min_file_growth_size: usize) -> Result<Self> {
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
            return Err(MmapError::InvalidFile(
                "file must be at least 2 meta page sizes + 2 pages".into(),
            ));
        }
        // TODO: validate the contents of the file.
        // TODO: fix any errors if possible.

        // Safety: it is assumed that no other process has a mutable mapping to the same file.
        let mmap = unsafe { MmapOptions::new().map_mut(&file).unwrap() };

        Ok(Mmap {
            min_file_growth_size,
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
            min_file_growth_size: self.min_file_growth_size,
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
    new_offset: usize,
    reclaimable_pages: Vec<usize>,
    free_list: FreeList,
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
        self.flush_offset = self.new_offset;
    }

    /// Grows the underlying mmap only if there's no more space for new pages.
    #[allow(clippy::arc_with_non_send_sync)] // We manually ensure thread-safety via Mutex/ArcSwap.
    fn grow_if_needed(&mut self) -> bool {
        let m = self.mmap();
        if meta_node::META_PAGE_SIZE + self.new_offset + consts::PAGE_SIZE <= m.len() {
            return false;
        }
        let expand = max(m.len(), DEFAULT_MIN_FILE_GROWTH_SIZE);
        let new_len = m.len() + expand;

        m.flush();
        self.mmap = Arc::new(UnsafeCell::new(m.grow(new_len)));
        true
    }

    /// Flushes only the meta node of the mmap.
    fn flush_new_meta_node(&mut self, new_root_ptr: usize) {
        let m = self.mmap_mut();
        let curr = MetaNode::new(
            new_root_ptr,
            self.flush_offset / consts::PAGE_SIZE,
            &self.free_list,
        );
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
    pub fn new(mmap: Mmap, collector: Collector) -> Self {
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
                new_offset: flush_offset,
                reclaimable_pages: Vec::new(),
                free_list: FreeList::from(node),
            }),
            collector,
            // Remember to manually drop this leaked box in Store::drop.
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
            _collector_guard: collector_guard,
        }
    }

    /// Obtains a Writer guard. This will block so long as a Writer
    /// was previously obtained and not yet dropped.
    pub fn writer(self: &Arc<Self>) -> Writer<'_> {
        let mutex_guard = self.writer_state.lock().unwrap();
        let collector_guard = self.collector.enter();
        // Note: consider Ordering::Acquire if perf is necessary (prob not).
        let prev_root_page = collector_guard.protect(&self.root_page, Ordering::SeqCst);
        Writer {
            prev_root_page,
            // Safety: not null b/c always initialized.
            curr_root_page: unsafe { *prev_root_page },
            abort: true,
            store: self.clone(),
            state_guard: RefCell::new(mutex_guard),
            collector_guard,
        }
    }

    /// Adds pages to the free list.
    fn reclaim_pages(self: &Arc<Self>) {
        let w = self.writer();
        let (mut fl, pages) = {
            let mut borrow = w.state_guard.borrow_mut();
            if borrow.reclaimable_pages.is_empty() {
                drop(borrow);
                w.abort();
                return;
            }
            let fl = borrow.free_list;
            let pages = std::mem::take(&mut borrow.reclaimable_pages);
            (fl, pages)
        };
        for page_num in pages {
            fl.push_tail(&w, page_num);
        }
        w.state_guard.borrow_mut().free_list = fl;
        let root_page = w.curr_root_page; // unchanged
        w.flush(root_page);
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        // Safety: root_page is never null.
        drop(unsafe { Box::from_raw(self.root_page.load(Ordering::SeqCst)) });
    }
}

/// Guard is a trait common to both Reader and Writer.
/// It provides guarded read-only access to the mmap region.
pub trait Guard<'g, P: ImmutablePage<'g>> {
    /// Reads a page at `page_num`.
    ///
    /// This function is unsafe b/c if used incorrectly, it can access pages
    /// that is actively shared with a writer. This can happen if
    /// `writer.overwrite_page(page_num)` was called and the `Page` was not yet
    /// dropped.
    ///
    /// To safely use this function, please ensure to only read pages that
    /// are guaranteed to never have any concurrent readers accessing it.
    ///
    /// Note: If the implementing type is Reader, only flushed pages can be
    /// read.
    ///
    /// If it's a Writer, then either flushed or written pages can be read.
    /// Newly allocated pages must be written first before they can be read.
    /// Remember to flush pages to make them available to future readers.
    unsafe fn read_page(&'g self, page_num: usize) -> P;

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
    _collector_guard: LocalGuard<'s>,
}

impl<'r> Guard<'r, ReaderPage<'r>> for Reader<'_> {
    unsafe fn read_page(&self, page_num: usize) -> ReaderPage<'_> {
        let guard = &self.state_guard;
        assert!(
            page_num * consts::PAGE_SIZE < guard.flush_offset,
            "page_num = {} must be < {}",
            page_num,
            guard.flush_offset / consts::PAGE_SIZE
        );
        ReaderPage {
            // Safety: The Arc cannot be dropped as long as Reader lives
            // (due to the ArcSwapGuard).
            mmap: unsafe {
                let ptr = &guard.mmap as *const Arc<UnsafeCell<Mmap>>;
                &*ptr
            },
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
    abort: bool,
    store: Arc<Store>,
    state_guard: RefCell<MutexGuard<'s, WriterState>>,

    // It's important that collector_guard is listed AFTER state_guard
    // so that collector_guard is dropped AFTER state_guard is dropped.
    // When Writer is dropped, it defers page reclamation, which itself
    // acquires the store's mutex. The mutex cannot be locked unless the writer
    // first drops its MutexGuard.
    collector_guard: LocalGuard<'s>,
}

impl<'s> Writer<'s> {
    /// Allocates a new page for write.
    pub fn new_page(&self) -> WriterPage<'_, 's> {
        let guard = &self.state_guard;
        // Try from the free list first.
        let mut fl = guard.borrow_mut().free_list;
        let page_num = fl.pop_head(self).unwrap_or_else(|| {
            // Otherwise allocate from the page bank.
            let mut borrow = guard.borrow_mut();
            borrow.grow_if_needed();
            borrow.new_offset += consts::PAGE_SIZE;
            (borrow.new_offset - 1) / consts::PAGE_SIZE
        });
        guard.borrow_mut().free_list = fl;
        WriterPage {
            writer: self,
            page_num,
        }
    }

    /// Flushes all written pages and update the meta node root pointer.
    /// Every new page must first be written before calling `flush`.
    pub fn flush(mut self, new_root_ptr: usize) {
        self.abort = false;
        let mut guard = self.state_guard.borrow_mut();
        guard.flush_pages();
        guard.flush_new_meta_node(new_root_ptr);
        self.store.reader_state.store(Arc::new(ReaderState {
            mmap: guard.mmap.clone(),
            flush_offset: guard.flush_offset,
        }));
        self.curr_root_page = new_root_ptr;
        // Safety: prev_root_page is never uninitialized.
        if new_root_ptr == unsafe { *self.prev_root_page } {
            return;
        }
        // Note: consider Ordering::Release if perf is necessary (prob not).
        let _ = self.collector_guard.swap(
            &self.store.root_page,
            Box::into_raw(Box::new(new_root_ptr)),
            Ordering::SeqCst,
        );
    }

    /// Retrieves an existing page and allows the user to write to it.
    ///
    /// This is unsafe b/c it can allow overwriting data that is assumed to be
    /// immutable. As such, please only use this on non-B+-tree pages,
    /// i.e. free list pages. Also, make sure there exists only one mutable
    /// reference at a time to the underlying page.
    pub unsafe fn overwrite_page(&self, page_num: usize) -> WriterPage<'_, 's> {
        let guard = self.state_guard.borrow();
        assert!(page_num * consts::PAGE_SIZE < guard.new_offset);
        WriterPage {
            writer: self,
            page_num,
        }
    }

    /// Marks an existing page as free, allowing it to be reclaimed later
    /// and made available via `overwrite_page()`.
    pub fn mark_free(&self, page_num: usize) {
        let offset = page_num * consts::PAGE_SIZE;
        let mut borrow = self.state_guard.borrow_mut();
        assert!(offset < borrow.new_offset);
        // We can put directly into the free list if it's not flushed,
        // i.e. not accessible by concurrent readers.
        if offset >= borrow.flush_offset {
            let mut fl = borrow.free_list;

            // FreeList::push_tail uses self, which may itself try to call
            // borrow()/borrow_mut(), which can cause a panic.
            drop(borrow);

            fl.push_tail(self, page_num);
            self.state_guard.borrow_mut().free_list = fl;
            return;
        }
        borrow.reclaimable_pages.push(page_num);
    }

    /// Aborts the write transaction.
    pub fn abort(mut self) {
        self.abort = true;
    }
}

impl<'w, 's> Guard<'w, WriterPage<'w, 's>> for Writer<'s> {
    unsafe fn read_page(&'w self, page_num: usize) -> WriterPage<'w, 's> {
        let borrow = self.state_guard.borrow();
        assert!(
            page_num * consts::PAGE_SIZE < borrow.new_offset,
            "page_num = {} must be < {}",
            page_num,
            borrow.new_offset / consts::PAGE_SIZE
        );
        WriterPage {
            writer: self,
            page_num,
        }
    }

    fn root_page(&self) -> usize {
        self.curr_root_page
    }
}

impl Drop for Writer<'_> {
    fn drop(&mut self) {
        let mut borrow = self.state_guard.borrow_mut();
        if self.abort {
            // This can happen if the write transaction errored or was aborted.
            // Reset as if nothing ever happened.
            borrow.new_offset = borrow.flush_offset;
            borrow.reclaimable_pages.truncate(0);
            borrow.free_list = FreeList::from(
                MetaNode::try_from(borrow.mmap().as_ref()).expect("there should exist a meta node"),
            );
            return;
        }

        /* Pages cannot be reclaimed back to the free list until it's
        guaranteed that no reader (or writer) can traverse to these pages,
        either from a B+ tree root node, or via the free list itself.
         */

        // Safety: prev_root_page is never null.
        let root_page = if self.curr_root_page != unsafe { *self.prev_root_page } {
            Some(self.prev_root_page)
        } else {
            None
        };
        let reclaim_pages = !borrow.reclaimable_pages.is_empty();
        if root_page.is_none() && !reclaim_pages {
            // Don't bother with retirement mechanism.
            return;
        }

        // Box::into_raw so reclaimable doesn't get dropped until
        // defer_retire
        let reclaimable = Box::into_raw(Box::new(Reclaimable {
            store: self.store.clone(),
            root_page,
            reclaim_pages,
        }));
        // Safety: prev_root_page has already been replaced in the flush() call,
        // so it is safe to defer its retirement.
        unsafe {
            self.collector_guard.defer_retire(reclaimable, |r, _| {
                let r = Box::from_raw(r); // so r can be auto-dropped
                if let Some(root_page) = r.root_page {
                    drop(Box::from_raw(root_page)); // must manually be dropped
                }
                if r.reclaim_pages {
                    r.store.reclaim_pages();
                }
            });
        }
    }
}

// Safety: This is needed b/c `prev_root_page` is `*mut usize`. Fortunately,
// this pointer isn't self referential, and it is never mutated.
unsafe impl Send for Writer<'_> {}
unsafe impl Sync for Writer<'_> {}

/// A container to be passed to [`seize::LocalGuard::defer_retire`]
/// so that its underlying garbage can be collected.
struct Reclaimable {
    store: Arc<Store>,
    root_page: Option<*mut usize>,
    reclaim_pages: bool,
}

/// A page that provides read-only access to its data.
/// This trait is implemented only by [`ReaderPage`]
/// and [`WriterPage`].
pub trait ImmutablePage<'g>: Deref<Target = [u8]> {
    /// Returns the page number of this page.
    fn page_num(&self) -> usize;
}

/// A page inside the mmap region that allows for only reads.
/// Only flushed pages are accessible by readers.
/// This is also an implementation of [`ImmutablePage`] on the reader side.
pub struct ReaderPage<'r> {
    mmap: &'r Arc<UnsafeCell<Mmap>>,
    page_num: usize,
}

impl ReaderPage<'_> {
    pub fn page_num(&self) -> usize {
        self.page_num
    }
}

impl Deref for ReaderPage<'_> {
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

impl ImmutablePage<'_> for ReaderPage<'_> {
    fn page_num(&self) -> usize {
        self.page_num
    }
}

/// A page inside the mmap region that allows for reads and possibly writes.
/// A page is not accessible to readers until flushed.
/// This is also an implementation of [`ImmutablePage`] on the writer side.
pub struct WriterPage<'w, 's> {
    writer: &'w Writer<'s>,
    page_num: usize,
}

impl<'w, 's> WriterPage<'w, 's> {
    pub fn read_only(self) -> WriterPage<'w, 's> {
        WriterPage {
            writer: self.writer,
            page_num: self.page_num,
        }
    }
}

impl Deref for WriterPage<'_, '_> {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        let borrow = self.writer.state_guard.borrow();
        // Safety: Due to the mutex guard in Writer, at most one mutable
        // reference can touch this region at a time.
        &unsafe { &*(borrow.mmap() as *const Mmap) }.mmap[meta_node::META_PAGE_SIZE
            + self.page_num * consts::PAGE_SIZE
            ..meta_node::META_PAGE_SIZE + (self.page_num + 1) * consts::PAGE_SIZE]
    }
}

impl DerefMut for WriterPage<'_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let borrow_mut = self.writer.state_guard.borrow_mut();
        // Safety: Due to the mutex guard in Writer, at most one mutable
        // reference can touch this region at a time.
        &mut unsafe { &mut *(borrow_mut.mmap_mut() as *mut Mmap) }.mmap[meta_node::META_PAGE_SIZE
            + self.page_num * consts::PAGE_SIZE
            ..meta_node::META_PAGE_SIZE + (self.page_num + 1) * consts::PAGE_SIZE]
    }
}

impl ImmutablePage<'_> for WriterPage<'_, '_> {
    fn page_num(&self) -> usize {
        self.page_num
    }
}

/// Allocates a new page as an empty leaf and writes it into the store.
pub fn write_empty_leaf(writer: &Writer) -> usize {
    let mut page = writer.new_page();
    init_empty_leaf(&mut page);
    page.read_only().page_num()
}

fn init_empty_leaf(page: &mut [u8]) {
    header::set_node_type(page, NodeType::Leaf);
    header::set_num_keys(page, 0);
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use crate::core::consts;

    use super::*;

    fn new_file_mmap() -> (Arc<Store>, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        println!("Created temporary file {path:?}");
        let mmap = Mmap::open_or_create(path, DEFAULT_MIN_FILE_GROWTH_SIZE).unwrap();

        // Use batch size of 1 to trigger garbage collection ASAP.
        let collector = Collector::new().batch_size(1);

        let store = Arc::new(Store::new(mmap, collector));
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

            page_num = page.read_only().page_num();
            // Flush with the root pointer as the new page number for simplicity
            writer.flush(page_num);
        }
        drop(store);

        // Scope 2: Reopen and verify
        {
            let mmap = Mmap::open_or_create(path, DEFAULT_MIN_FILE_GROWTH_SIZE).unwrap();
            let collector = Collector::new();
            let store = Arc::new(Store::new(mmap, collector));
            let reader = store.reader();

            // Verify the meta node
            let root_page = reader.root_page();
            assert_eq!(
                root_page, page_num,
                "Root pointer should match the flushed page number"
            );

            // Verify the page
            let read_page = unsafe { reader.read_page(page_num) };
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

            let result = Mmap::open_or_create(temp_file.path(), DEFAULT_MIN_FILE_GROWTH_SIZE);
            assert!(matches!(result, Err(MmapError::InvalidFile(_))));
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
            page.read_only().page_num()
        };

        // Read the written page using the same writer
        let read_page1 = unsafe { writer.read_page(page1_num) };
        assert_eq!(
            &read_page1[edit_offset1..edit_offset1 + pattern1.len()],
            &pattern1,
            "Writer should be able to read its own written (but not flushed) page"
        );

        // 2. Test writer can read a flushed page.
        let pattern2 = [0xDD, 0xEE, 0xFF];
        let edit_offset2 = 20;
        let page2_num = {
            let mut page = writer.new_page();
            page[edit_offset2..edit_offset2 + pattern2.len()].copy_from_slice(&pattern2);
            let page_num = page.read_only().page_num;
            writer.flush(page_num); // Flush this page
            page_num
        };

        // Read the flushed page using a writer.
        let writer = store.writer();
        let read_page2 = unsafe { writer.read_page(page2_num) };
        assert_eq!(
            &read_page2[edit_offset2..edit_offset2 + pattern2.len()],
            &pattern2,
            "Writer should be able to read a flushed page"
        );

        // 3. Test writer CANNOT read a newly allocated page that hasn't been written yet.
        let _page3 = writer.new_page();
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
            page1_num = page.read_only().page_num();
            writer.flush(page1_num);
        } // Writer dropped here

        // 2. Get a reader and verify it can read the flushed page.
        {
            let reader1 = store.reader();
            let read_page1 = unsafe { reader1.read_page(page1_num) };
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
            let read_page1_again = unsafe { reader2.read_page(page1_num) };
            assert_eq!(
                &read_page1_again[edit_offset1..edit_offset1 + pattern1.len()],
                &pattern1,
                "Second reader should still be able to read the first flushed page"
            );
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
            page.read_only() // Keep the ReadOnlyPage handle
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
            page1_num = page.read_only().page_num();
            writer.flush(page1_num);
        } // Writer dropped here

        // 2. Get a reader and obtain a ReadOnlyPage handle for the flushed page.
        let reader = store.reader();
        let page1_handle = unsafe { reader.read_page(page1_num) };

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

        let page_num = page.read_only().page_num();
        writer.flush(page_num);

        let reader = store.reader();
        let read_page = unsafe { reader.read_page(page_num) };

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
        let page_num = page.read_only().page_num();
        // DON'T flush. Drop the writer instead.
        drop(writer);

        let reader = store.reader();
        // Try to read the recently written page via Reader::read_page(page_num).
        // This should fail b/c not flushed yet.
        let _ = unsafe { reader.read_page(page_num) };
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
            let page_num = page.read_only().page_num();
            writer.flush(page_num);
        }

        // Get another writer via Store::writer() and write + flush another page.
        // Do NOT drop the writer.
        let writer = store.writer();
        let mut page = writer.new_page();
        page[0] = 2;
        let page_num = page.read_only().page_num();
        writer.flush(page_num);

        // Save a handle on that page via Store::reader() and Reader::read_page().
        threads.push(thread::spawn({
            let store = store.clone();
            move || {
                let reader = store.reader();
                let page = unsafe { reader.read_page(page_num) };
                assert_eq!(page[0], 1);
            }
        }));
        threads.push(thread::spawn({
            let store = store.clone();
            move || {
                let reader = store.reader();
                let page = unsafe { reader.read_page(page_num) };
                assert_eq!(page[0], 1);
            }
        }));

        for t in threads {
            let _ = t.join();
        }
    }

    #[test]
    fn test_free_list_reclaimed_after_flush() {
        // Start with empty free list.
        let (store, temp_file) = new_file_mmap();
        // Alloc 3 pages.
        // Flush.
        {
            let writer = store.writer();
            assert_eq!(writer.new_page().read_only().page_num(), 2);
            assert_eq!(writer.new_page().read_only().page_num(), 3);
            assert_eq!(writer.new_page().read_only().page_num(), 4);
            writer.flush(0 /* dummy */);
        }
        // Read disk free list; it should be empty.
        {
            let mmap =
                Mmap::open_or_create(temp_file.path(), DEFAULT_MIN_FILE_GROWTH_SIZE).unwrap();
            let meta_node = MetaNode::try_from(mmap.as_ref()).unwrap();
            assert_eq!(meta_node.head_seq, meta_node.tail_seq);
        }
        // Free 2 pages.
        // Alloc a new page. Its page_num should NOT one of the previous 3.
        // Flush.
        {
            let writer = store.writer();
            writer.mark_free(4);
            writer.mark_free(2);
            writer.new_page();
            writer.flush(0 /* dummy */);
        }
        // Read disk free list; it should have 2 free page (tail_seq - head_seq == 2).
        // Verify that meta node's num_pages == 4 + 2,
        // where 2 is the starting page count.
        {
            let mmap =
                Mmap::open_or_create(temp_file.path(), DEFAULT_MIN_FILE_GROWTH_SIZE).unwrap();
            let meta_node = MetaNode::try_from(mmap.as_ref()).unwrap();
            assert!(
                meta_node.tail_seq - meta_node.head_seq == 2,
                "meta_node: {meta_node:?}"
            );
            assert_eq!(meta_node.num_pages, 6, "meta_node: {meta_node:?}");
        }
        // Alloc a new page.
        // Flush.
        // Flush again (idempotent for test determinism).
        {
            let writer = store.writer();
            writer.new_page().read_only().page_num();
            writer.flush(0 /* dummy */);
            store.writer().flush(0 /* dummy */);
        }
        // Read disk free list; it should have 1 free page (tail_seq - head_seq == 1).
        // Verify that meta node's num_pages == 4 + 2,
        // where 2 is the starting page count.
        {
            let mmap =
                Mmap::open_or_create(temp_file.path(), DEFAULT_MIN_FILE_GROWTH_SIZE).unwrap();
            let meta_node = MetaNode::try_from(mmap.as_ref()).unwrap();
            assert!(
                meta_node.tail_seq - meta_node.head_seq == 1,
                "meta_node: {meta_node:?}"
            );
            assert_eq!(meta_node.num_pages, 6, "meta_node: {meta_node:?}");
        }
    }

    #[test]
    fn test_free_list_reclaimed_before_flush() {
        // Start with empty free list.
        let (store, temp_file) = new_file_mmap();
        // Alloc 3 pages.
        // Free 2 pages.
        // Alloc a new page. Its page_num should be one of the previous 3.
        // Flush.
        // Flush again (idempotent for test determinism).
        {
            let writer = store.writer();
            assert_eq!(writer.new_page().read_only().page_num(), 2);
            assert_eq!(writer.new_page().read_only().page_num(), 3);
            assert_eq!(writer.new_page().read_only().page_num(), 4);
            writer.mark_free(4);
            writer.mark_free(2);
            let page_num = writer.new_page().read_only().page_num();
            assert_eq!(page_num, 4, "got page_num = {page_num}, want == 4");
            writer.flush(0 /* dummy */);
            store.writer().flush(0 /* dummy */);
        }
        // Read disk free list; it should have 1 free page (tail_seq - head_seq == 1).
        // Verify that meta node's num_pages == 3 + 2,
        // where 2 is the starting page count.
        {
            let mmap =
                Mmap::open_or_create(temp_file.path(), DEFAULT_MIN_FILE_GROWTH_SIZE).unwrap();
            let meta_node = MetaNode::try_from(mmap.as_ref()).unwrap();
            assert!(
                meta_node.tail_seq - meta_node.head_seq == 1,
                "meta_node: {meta_node:?}"
            );
            assert_eq!(meta_node.num_pages, 5, "meta_node: {meta_node:?}");
        }
    }

    #[test]
    fn test_free_list_grow_then_shrink() {
        let (store, temp_file) = new_file_mmap();
        let prev_tail_seq = {
            let writer = store.writer();
            let page_nums = (0..2 * free_list::FREE_LIST_CAP)
                .map(|_| writer.new_page().read_only().page_num())
                .collect::<Vec<_>>();
            for page_num in page_nums {
                writer.mark_free(page_num);
            }
            writer.flush(0 /* dummy */);
            let mmap =
                Mmap::open_or_create(temp_file.path(), DEFAULT_MIN_FILE_GROWTH_SIZE).unwrap();
            let node = MetaNode::try_from(mmap.as_ref()).unwrap();
            assert_ne!(node.head_page, node.tail_page);
            assert!(node.tail_seq > node.head_seq);
            node.tail_seq
        };
        {
            let writer = store.writer();
            for _ in 0..2 * free_list::FREE_LIST_CAP {
                let _ = writer.new_page();
            }
            writer.flush(0 /* dummy */);
            let mmap =
                Mmap::open_or_create(temp_file.path(), DEFAULT_MIN_FILE_GROWTH_SIZE).unwrap();
            let node = MetaNode::try_from(mmap.as_ref()).unwrap();
            assert_eq!(node.head_page, node.tail_page);
            assert!(node.tail_seq > prev_tail_seq);
        }
    }

    #[test]
    fn test_free_list_grow_and_shrink() {
        // Start with empty free list.
        let (store, temp_file) = new_file_mmap();
        // Repeat:
        // * let writer = store.writer();
        // * let page_num = writer.new_page().read_only().page_num();
        // * writer.mark_free(page_num);
        // * writer.flush(0 /* dummy */);
        // * Read file's free list. If tail_page != head_page, done.
        let mut i = 0;
        loop {
            let writer = store.writer();
            let page_num = writer.new_page().read_only().page_num();
            writer.mark_free(page_num);
            writer.flush(0 /* dummy */);
            i += 1;
            let mmap =
                Mmap::open_or_create(temp_file.path(), DEFAULT_MIN_FILE_GROWTH_SIZE).unwrap();
            let node = MetaNode::try_from(mmap.as_ref()).unwrap();
            if node.tail_page != node.head_page {
                break;
            }
            if i == 1000 {
                panic!("i = {i}");
            }
        }
    }
}
