use std::ops::{Deref, DerefMut};

use crate::tree::error::PageStoreError;

mod in_memory;
mod mmap_file;

pub use mmap_file::{MmapFile, MmapPage, ReadOnlyMmapPage};

#[cfg(test)]
pub use in_memory::{InMemory, InMemoryPage, ReadOnlyInMemoryPage};

type Result<T> = std::result::Result<T, PageStoreError>;

/// A store of pages that backs a COW B+ Tree.
pub trait PageStore: Clone {
    type Page: DerefMut<Target = [u8]>;
    type OverflowPage: DerefMut<Target = [u8]>;
    type ReadOnlyPage: ReadOnlyPage;

    /// Reads a page from disk into an in-memory B+ tree node.
    fn read_page(&self, page_num: usize) -> Result<Self::ReadOnlyPage>;

    /// Creates a new page that can be later written into the store.
    fn new_page(&self) -> Result<Self::Page>;

    /// Writes a page to the store.
    fn write_page(&self, page: Self::Page) -> Self::ReadOnlyPage;

    /// Creates a new overflow page that can be later written into the store.
    fn new_overflow_page(&self) -> Result<Self::OverflowPage>;

    /// Writes the left split of an overflow page to the store.
    fn write_overflow_left_split(&self, page: Self::OverflowPage) -> Result<Self::ReadOnlyPage>;

    /// Flushes all written pages and makes them available for reading.
    fn flush(&self) -> Result<()>;
}

pub trait ReadOnlyPage: Deref<Target = [u8]> {
    fn page_num(&self) -> usize;
}
