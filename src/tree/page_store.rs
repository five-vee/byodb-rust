use std::ops::{Deref, DerefMut};

use crate::error::PageError;

mod in_memory;

#[cfg(test)]
pub use in_memory::{InMemory, InMemoryPage, ReadOnlyInMemoryPage};

type Result<T> = std::result::Result<T, PageError>;

/// A store of pages that backs a COW B+ Tree.
pub trait PageStore: Clone {
    type Page: DerefMut<Target = [u8]>;
    type ReadOnlyPage: ReadOnlyPage;

    /// Reads a page from disk into an in-memory B+ tree node.
    fn read_page(&self, page_num: usize) -> Result<Self::ReadOnlyPage>;

    /// Creates a new page that can be later written into the store.
    fn new_page(&self) -> Result<Self::Page>;

    /// Writes a page to the store.
    fn write_page(&self, page: Self::Page) -> Self::ReadOnlyPage;

    /// Flushes all written pages and makes them available for reading.
    fn flush(&self) -> Result<()>;
}

pub trait ReadOnlyPage: Deref<Target = [u8]> {
    fn page_num(&self) -> usize;
}
