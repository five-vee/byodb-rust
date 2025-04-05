use std::rc::Rc;

use crate::tree::buffer_store::BufferStore;
use crate::tree::error::PageStoreError;

mod in_memory;
mod mmap_file;

pub use in_memory::InMemory;
pub use mmap_file::MmapFile;

type Result<T> = std::result::Result<T, PageStoreError>;

/// A store of pages that backs a COW B+ Tree.
pub trait PageStore: Clone {
    type B: BufferStore;

    /// Reads a page from disk into an in-memory B+ tree node.
    fn read_page(&self, page_num: usize) -> Result<Rc<[u8]>>;

    /// Writes an in-memory B+ tree node into a page on disk.
    fn write_page(&self, page: Rc<[u8]>) -> Result<usize>;

    fn buffer_store(&self) -> &Self::B;
}
