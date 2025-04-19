mod meta_node;

use std::{
    cell::{RefCell, UnsafeCell},
    cmp::max,
    fs::{File, OpenOptions},
    io::Write as _,
    marker::PhantomData,
    ops::{Deref, DerefMut, Range},
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
};

use arc_swap::{ArcSwap, Guard as ArcSwapGuard};
use mmap_rs::{MmapMut, MmapOptions};

use crate::{consts, error::PageError};
use meta_node::MetaNode;

type Result<T> = std::result::Result<T, PageError>;

pub struct Mmap {
    file: Option<Arc<File>>,
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
    pub fn new_anonymous(num_pages: usize) -> Self {
        let mut mmap = MmapOptions::new(meta_node::META_OFFSET + num_pages * consts::PAGE_SIZE)
            .expect("len > 0")
            .map_mut()
            .expect("mmap is correctly created");
        MetaNode::new()
            .copy_to_slice(mmap.deref_mut(), meta_node::Position::A)
            .expect("mmap is already large enough to hold 2 meta nodes");
        Mmap { file: None, mmap }
    }

    pub fn open_or_create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let file_len = file.metadata()?.len() as usize;
        if file_len < meta_node::META_OFFSET {
            return Err(PageError::InvalidFile(
                "file must be at least 2 meta page sizes".into(),
            ));
        }
        if (file_len - meta_node::META_OFFSET) % consts::PAGE_SIZE != 0 {
            return Err(PageError::InvalidFile(
                "file size must be 2 meta pages, plus a multiple of page size".into(),
            ));
        }
        if file_len == 0 {
            let mut meta_prelude = [0u8; meta_node::META_OFFSET];
            MetaNode::new().copy_to_slice(&mut meta_prelude, meta_node::Position::A)?;
            file.write_all(&meta_prelude)?;
            // Don't need to sync until a write happens later.
        }
        // Safety: it is assumed that no other process has a mutable mapping to the same file.
        let mmap = unsafe { MmapOptions::new(0)?.with_file(&file, 0).map_mut()? };
        Ok(Mmap {
            file: Some(Arc::new(file)),
            mmap,
        })
    }

    fn flush(&self, range: Range<usize>) {
        if self.file.is_some() {
            self.mmap.flush(range).expect("mmap can validly msync");
        }
    }

    fn grow(&self, new_len: usize) -> Self {
        let mut mmap_opts = MmapOptions::new(new_len).expect("new_len > 0");
        let mmap = match &self.file {
            None => {
                let mut mmap = mmap_opts.map_mut().expect("mmap is correctly created");
                mmap[..self.mmap.len()].copy_from_slice(self.mmap.deref());
                mmap
            }
            Some(f) => {
                // Safety: it is assumed that no other process has a mutable mapping to the same file.
                mmap_opts = unsafe { mmap_opts.with_file(f, 0) };
                mmap_opts.map_mut().expect("mmap is correctly created")
            }
        };
        Mmap {
            file: self.file.clone(),
            mmap,
        }
    }

    fn pages_ptr(&self) -> *mut u8 {
        // Safety: ptr + META_OFFSET is a valid pointer into the mmap region.
        unsafe { self.mmap.as_ptr().add(meta_node::META_OFFSET) as *mut u8 }
    }
}

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

impl ReaderState {
    fn mmap(&self) -> &Mmap {
        // Safety: ReaderState guarantees that immutable references of the mmap
        // only read [pages_ptr, pages_ptr + flush_offset * PAGE_SIZE), which never overlaps
        // with mmutable reference touches of [pages_ptr + flush_offset * PAGE_SIZE, mmap_end).
        unsafe { &*self.mmap.get() }
    }
}

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
    #[allow(clippy::mut_from_ref)] // This is the same as if inlined.
    fn mmap_mut(&self) -> &mut Mmap {
        // Safety: WriterState guarantees that mutable references of the mmap
        // only touch [pages_ptr + flush_offset * PAGE_SIZE, mmap_end), which never overlaps
        // with immutable reference reads of [pages_ptr, pages_ptr + flush_offset * PAGE_SIZE).
        unsafe { &mut *self.mmap.get() }
    }

    fn mmap(&self) -> &Mmap {
        self.mmap_mut()
    }

    fn flush_pages(&mut self) {
        let m = self.mmap_mut();
        let range =
            meta_node::META_OFFSET + self.flush_offset..meta_node::META_OFFSET + self.write_offset;
        m.flush(range);
        self.flush_offset = self.write_offset;
    }

    #[allow(clippy::arc_with_non_send_sync)] // We manually ensure thread-safety via Mutex/ArcSwap.
    fn grow_if_needed(&mut self) -> bool {
        let m = self.mmap();
        if meta_node::META_OFFSET + self.new_offset + consts::PAGE_SIZE <= m.len() {
            return false;
        }
        let expand = max(
            meta_node::META_OFFSET + self.new_offset + consts::PAGE_SIZE - m.len(),
            m.len(),
        );
        let expand = max(expand, 1 << 26 /* 64MB */);
        let new_len = m.len() + expand;
        self.mmap = Arc::new(UnsafeCell::new(m.grow(new_len)));
        true
    }

    fn flush_new_meta_node(&mut self, new_root_ptr: usize) {
        let m = self.mmap_mut();
        let (
            MetaNode {
                sequence: prev_sequence,
                ..
            },
            prev_pos,
        ) = MetaNode::read_last_valid_meta_node(m.deref())
            .expect("there exists at least 1 valid meta node");
        let mut curr = MetaNode {
            signature: meta_node::DB_SIG,
            root_ptr: new_root_ptr,
            num_pages: self.flush_offset / consts::PAGE_SIZE,
            sequence: prev_sequence + 1,
            checksum: 0,
        };
        curr.checksum = curr.checksum();
        curr.copy_to_slice(m.deref_mut(), prev_pos.next())
            .expect("mmap is already large enough to hold 2 meta nodes");
        m.flush(0..meta_node::META_OFFSET);
    }
}

pub struct Store {
    reader: Arc<ArcSwap<ReaderState>>,
    writer: Arc<Mutex<WriterState>>,
}

impl Store {
    pub fn new(mmap: Mmap) -> Result<Self> {
        let (MetaNode { num_pages, .. }, _) = MetaNode::read_last_valid_meta_node(mmap.deref())?;
        let offset = num_pages * consts::PAGE_SIZE;

        // We manually ensure thread-safety via Mutex/ArcSwap.
        #[allow(clippy::arc_with_non_send_sync)]
        let mmap = Arc::new(UnsafeCell::new(mmap));
        Ok(Store {
            reader: Arc::new(ArcSwap::from_pointee(ReaderState {
                mmap: mmap.clone(),
                flush_offset: offset,
            })),
            writer: Arc::new(Mutex::new(WriterState {
                mmap: mmap.clone(),
                flush_offset: offset,
                write_offset: offset,
                new_offset: offset,
            })),
        })
    }

    pub fn reader(&self) -> Reader {
        Reader {
            guard: self.reader.load(),
        }
    }

    pub fn writer(&self) -> Writer<'_> {
        Writer {
            guard: RefCell::new(self.writer.lock().unwrap()),
            store: self,
        }
    }
}

pub struct Reader {
    guard: ArcSwapGuard<Arc<ReaderState>>,
}

impl Reader {
    pub fn read_page(&self, page_num: usize) -> Result<ReadOnlyPage<'_>> {
        if page_num * consts::PAGE_SIZE >= self.guard.flush_offset {
            return Err(PageError::Read(
                format!(
                    "page_num = {} must be < {}",
                    page_num,
                    self.guard.flush_offset / consts::PAGE_SIZE
                )
                .into(),
            ));
        }
        Ok(ReadOnlyPage {
            _phantom: PhantomData,
            // Safety: pages_ptr + page_num * PAGE_SIZE <= mmap_end
            page_ptr: unsafe {
                self.guard
                    .mmap()
                    .pages_ptr()
                    .add(page_num * consts::PAGE_SIZE)
            },
        })
    }

    pub fn sequence(&self) -> u64 {
        MetaNode::read_last_valid_meta_node(self.guard.mmap().deref())
            .expect("there exists at least 1 valid meta node")
            .0
            .sequence
    }
}

pub struct Writer<'s> {
    guard: RefCell<MutexGuard<'s, WriterState>>,
    store: &'s Store,
}

impl<'s> Writer<'s> {
    pub fn new_page<'w>(&'w self) -> Page<'s, 'w> {
        let mut guard = self.guard.borrow_mut();
        guard.grow_if_needed();
        let m = guard.mmap();
        let page = Page {
            _phantom: PhantomData,
            // Safety: pages_ptr + new_offset * PAGE_SIZE <= mmap_end
            page_ptr: unsafe { m.pages_ptr().add(guard.new_offset) },
        };
        guard.new_offset += consts::PAGE_SIZE;
        page
    }

    pub fn write_page<'w>(&'w self, page: Page<'s, 'w>) -> usize {
        let mut guard = self.guard.borrow_mut();
        guard.write_offset += consts::PAGE_SIZE;
        (page.page_ptr as usize - guard.mmap().pages_ptr() as usize) / consts::PAGE_SIZE
    }

    pub fn flush(&self, new_root_ptr: usize) {
        let mut guard = self.guard.borrow_mut();
        if guard.write_offset < guard.new_offset {
            panic!("there are still new pages not yet written");
        }
        guard.flush_pages();
        guard.flush_new_meta_node(new_root_ptr);
        self.store.reader.store(Arc::new(ReaderState {
            mmap: guard.mmap.clone(),
            flush_offset: guard.flush_offset,
        }));
    }

    pub fn sequence(&self) -> u64 {
        MetaNode::read_last_valid_meta_node(self.guard.borrow().mmap().deref())
            .expect("there exists at least 1 valid meta node")
            .0
            .sequence
    }
}

pub struct ReadOnlyPage<'r> {
    _phantom: PhantomData<&'r Reader>,
    page_ptr: *const u8,
}

impl Deref for ReadOnlyPage<'_> {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        // Safety: [page_ptr, page_ptr + PAGE_SIZE) is a guaranteed valid region in the mmap.
        // It is guaranteed that no writers can touch this region (since it's already flushed).
        // page_ptr cannot dangle b/c the mmap is never dropped/realloc'ed so long as
        // the reader guard (&'r Reader) exists.
        unsafe { std::slice::from_raw_parts(self.page_ptr, consts::PAGE_SIZE) }
    }
}

pub struct Page<'s, 'w> {
    _phantom: PhantomData<&'w Writer<'s>>,
    page_ptr: *mut u8,
}

impl Deref for Page<'_, '_> {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        // Safety: [page_ptr, page_ptr + PAGE_SIZE) is a guaranteed valid region in the mmap.
        // Due to the mutex guard in Writer, at most one mutable reference can
        // touch this region at a time.
        // page_ptr cannot dangle b/c the mmap is never dropped/realloc'ed so long as
        // the writer mutex guard (&'w Writer<'s>) exists.
        unsafe { std::slice::from_raw_parts(self.page_ptr, consts::PAGE_SIZE) }
    }
}

impl DerefMut for Page<'_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Safety: [page_ptr, page_ptr + PAGE_SIZE) is a guaranteed valid region in the mmap.
        // Due to the mutex guard in Writer, at most one mutable reference can
        // touch this region at a time.
        // page_ptr cannot dangle b/c the mmap is never dropped/realloc'ed so long as
        // the writer mutex guard (&'w Writer<'s>) exists.
        unsafe { std::slice::from_raw_parts_mut(self.page_ptr, consts::PAGE_SIZE) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts;

    #[test]
    fn test_store_can_read_after_flush() {
        let mmap = Mmap::new_anonymous(1);
        let store = Store::new(mmap).expect("Failed to create store");

        let writer = store.writer();
        let mut page = writer.new_page();
        assert_eq!(page.len(), consts::PAGE_SIZE);

        // Make a dummy edit to the page.
        let edit_offset = 10;
        let pattern = [0xAA, 0xBB, 0xCC];
        page[edit_offset..edit_offset + pattern.len()].copy_from_slice(&pattern);

        let page_num = writer.write_page(page);
        writer.flush(page_num);
        drop(writer);

        let reader = store.reader();
        let read_result = reader.read_page(page_num);
        assert!(
            read_result.is_ok(),
            "Should be able to read page after flush"
        );
        let read_page = read_result.unwrap();

        // Verify the dummy value made at the edit site is there and that the page is of correct size
        assert_eq!(read_page.len(), consts::PAGE_SIZE);
        assert_eq!(
            &read_page[edit_offset..edit_offset + pattern.len()],
            &pattern,
            "Data read should match data written"
        );
    }

    #[test]
    fn test_store_cannot_read_before_flush() {
        let mmap = Mmap::new_anonymous(1);
        let store = Store::new(mmap).expect("Failed to create store");

        let writer = store.writer();
        let page = writer.new_page();
        let page_num = writer.write_page(page);
        // DON'T flush. Drop the writer instead.
        drop(writer);

        let reader = store.reader();
        // Try to read the recently written page via Reader::read_page(page_num).
        // This should fail b/c not flushed yet.
        let read_result = reader.read_page(page_num);
        assert!(
            read_result.is_err(),
            "Should not be able to read page before flush"
        );
        match read_result {
            Err(PageError::Read(msg)) => {
                assert!(
                    msg.contains(&format!(
                        "page_num = {} must be < 0", // Initial flush_offset is 0 pages
                        page_num
                    )),
                    "Error message should indicate page number out of bounds"
                );
            }
            _ => panic!("Expected PageError::Read"),
        }
    }

    #[test]
    fn test_store_flush_increases_sequence() {
        let mmap = Mmap::new_anonymous(0);
        let store = Store::new(mmap).expect("Failed to create store");

        let dummy_root_ptr = 0;
        let seq0 = store.reader().sequence();
        let mut prev_reader_seq = seq0;
        let writer = store.writer();
        for i in 1..=2 {
            let writer_seq = writer.sequence();
            assert_eq!(
                writer_seq, prev_reader_seq,
                "Writer sequence should match previous reader sequence (Iteration {})",
                i
            );
            writer.flush(dummy_root_ptr);

            let reader = store.reader();
            let reader_seq = reader.sequence();
            assert_eq!(
                reader_seq,
                writer_seq + 1,
                "Reader sequence should be writer sequence + 1 (Iteration {})",
                i
            );
            assert_eq!(
                reader_seq, i,
                "Reader sequence should be {} after iteration {}",
                i, i
            );
            prev_reader_seq = reader_seq;
        }
    }

    #[test]
    fn test_store_new_page_address_always_increases() {
        let mmap = Mmap::new_anonymous(0);
        let store = Store::new(mmap).expect("Failed to create store");
        let writer = store.writer();
        let mut last_page_ptr = usize::MAX;
        let mut last_page_num = usize::MAX;

        for i in 0..2 {
            let pages_ptr = writer.guard.borrow().mmap().pages_ptr() as usize;

            let page = writer.new_page();
            let current_page_ptr = page.page_ptr as usize;
            if i > 0 {
                assert_eq!(
                    current_page_ptr - pages_ptr,
                    last_page_ptr - pages_ptr + consts::PAGE_SIZE,
                    "Page address should increase by exactly PAGE_SIZE (Iteration {})",
                    i
                );
            }

            let current_page_num = writer.write_page(page);
            if i > 0 {
                assert_eq!(
                    current_page_num,
                    last_page_num + 1,
                    "Page number should increase by 1 (Iteration {})",
                    i
                );
            }
            assert_eq!(
                current_page_num, i,
                "Page number should be {} for iteration {}",
                i, i
            );

            writer.flush(current_page_num); // Use page_num as dummy root pointer

            // Update state for next iteration
            last_page_ptr = current_page_ptr;
            last_page_num = current_page_num;
        }
    }
}
