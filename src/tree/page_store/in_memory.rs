#![cfg(test)]

use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

use crate::consts;
use crate::tree::page_store::{PageError, PageStore, ReadOnlyPage};

type Result<T> = std::result::Result<T, PageError>;

/// An in-memory store of pages. Backed by a hash map.
#[derive(Clone)]
pub struct InMemory {
    state: Arc<Mutex<InMemoryState>>,
}

struct InMemoryState {
    counter: usize,
    // Pages that haven't been written.
    new: HashMap<usize, Box<[u8]>>,
    // Pages that have been written but not yet flushed.
    written: HashMap<usize, Box<[u8]>>,
    // Pages that have been flushed.
    flushed: HashMap<usize, Box<[u8]>>,
}

impl InMemory {
    /// Creates a new in-memory page store.
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(InMemoryState {
                counter: 0,
                new: HashMap::new(),
                written: HashMap::new(),
                flushed: HashMap::new(),
            })),
        }
    }
}

impl Default for InMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl PageStore for InMemory {
    type Page = InMemoryPage;
    type ReadOnlyPage = ReadOnlyInMemoryPage;

    fn read_page(&self, page_num: usize) -> Result<Self::ReadOnlyPage> {
        let guard = self.state.lock().unwrap();
        if guard.new.contains_key(&page_num) {
            return Err(PageError::Read(
                format!("page {page_num} was not written yet").into(),
            ));
        }
        if guard.written.contains_key(&page_num) {
            return Err(PageError::Read(
                format!("page {page_num} was not flushed yet").into(),
            ));
        }
        if let Some(page) = guard.flushed.get(&page_num) {
            return Ok(ReadOnlyInMemoryPage {
                _ref: self.clone(),
                ptr: page.as_ptr(),
                page_num,
            });
        }
        Err(PageError::NotFound(page_num))
    }

    fn new_page(&self) -> Result<Self::Page> {
        let buf = Box::new([0u8; consts::PAGE_SIZE]);
        let ptr = buf.as_ptr();
        let mut guard = self.state.lock().unwrap();
        let page_num = guard.counter;
        guard.counter += 1;
        guard.new.insert(page_num, buf);
        Ok(InMemoryPage {
            _ref: self.clone(),
            ptr,
            page_num,
        })
    }

    fn write_page(&self, page: Self::Page) -> Self::ReadOnlyPage {
        let mut guard = self.state.lock().unwrap();
        let buf = guard.new.remove(&page.page_num).unwrap();
        guard.written.insert(page.page_num, buf);
        ReadOnlyInMemoryPage {
            _ref: page._ref,
            ptr: page.ptr,
            page_num: page.page_num,
        }
    }

    fn flush(&self) -> Result<()> {
        let mut guard = self.state.lock().unwrap();
        let written = std::mem::take(&mut guard.written);
        for (page_num, buf) in written {
            guard.flushed.insert(page_num, buf);
        }
        Ok(())
    }
}

pub struct InMemoryPage {
    _ref: InMemory,
    ptr: *const u8,
    page_num: usize,
}

impl Deref for InMemoryPage {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        // SAFETY: The Box<[u8]> is of size PAGE_SIZE.
        // SAFETY: ptr is to a Box<[u8]> that will never be dropped
        // while self._ref still references the backing store.
        // SAFETY: The Box<[u8]> is of size PAGE_SIZE.
        unsafe { &*std::ptr::slice_from_raw_parts(self.ptr, consts::PAGE_SIZE) }
    }
}

impl DerefMut for InMemoryPage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: The Box<[u8]> is of size PAGE_SIZE.
        // SAFETY: ptr is to a Box<[u8]> that will never be dropped
        // while self._ref still references the backing store.
        // SAFETY: ptr is unique, and there will only be one copy at any moment
        // in time. new_page creates a unique page, and write_page consumes it,
        // finalizing all edits to the buffer, thereby allowing it to be
        // dereferenced immutably by read_page.
        unsafe {
            let ptr = self.ptr as *mut u8;
            &mut *std::ptr::slice_from_raw_parts_mut(ptr, consts::PAGE_SIZE)
        }
    }
}

pub struct ReadOnlyInMemoryPage {
    _ref: InMemory,
    ptr: *const u8,
    page_num: usize,
}

impl Deref for ReadOnlyInMemoryPage {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        // SAFETY: The Box<[u8]> is of size PAGE_SIZE.
        // SAFETY: ptr is to a Box<[u8]> that will never be dropped
        // while self._ref still references the backing store.
        // SAFETY: The Box<[u8]> is of size PAGE_SIZE.
        unsafe { &*std::ptr::slice_from_raw_parts(self.ptr, consts::PAGE_SIZE) }
    }
}

impl ReadOnlyPage for ReadOnlyInMemoryPage {
    fn page_num(&self) -> usize {
        self.page_num
    }
}

mod tests {
    use super::*;

    #[test]
    fn functional_test() {
        let store = InMemory::new();
        assert!(
            matches!(store.read_page(0), Err(PageError::NotFound(_))),
            "page 0 should not yet exist"
        );
        let mut page = store.new_page().unwrap();
        page[0] = 42;
        assert!(
            matches!(store.read_page(0), Err(PageError::Read(_))),
            "page 0 should not be written yet"
        );
        assert_eq!(page[0], 42);
        let read_only_page = store.write_page(page);
        assert_eq!(read_only_page[0], 42);
        assert!(
            matches!(
                store.read_page(read_only_page.page_num()),
                Err(PageError::Read(_))
            ),
            "read_only_page hasn't flushed yet"
        );
        store.flush().unwrap();
        assert_eq!(
            read_only_page.deref(),
            store.read_page(read_only_page.page_num()).unwrap().deref()
        );
    }
}
