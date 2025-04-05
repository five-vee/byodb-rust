use super::buffer_store::{BufferStore, Heap};
use super::error::PageStoreError;
use super::node::Node;
use std::sync::{Arc, Mutex};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
};

type Result<T> = std::result::Result<T, PageStoreError>;

/// A store of pages that backs a COW B+ Tree.
pub trait PageStore: Clone {
    type B: BufferStore;

    /// Reads a page from disk into an in-memory B+ tree node.
    fn read_page(&self, page_num: usize) -> Result<Node<Self::B>>;

    /// Writes an in-memory B+ tree node into a page on disk.
    fn write_page(&self, node: &Node<Self::B>) -> Result<usize>;

    fn buffer_store(&self) -> &Self::B;
}

/// An in-memory store of pages. Backed by a hash map.
#[derive(Clone)]
pub struct InMemory {
    state: Arc<Mutex<InMemoryState>>,
}

struct InMemoryState {
    pages: RefCell<HashMap<usize, Node<Heap>>>,
    counter: Cell<usize>,
}

impl InMemory {
    /// Creates a new in-memory page store.
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(InMemoryState {
                pages: HashMap::new().into(),
                counter: Cell::new(0),
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
    type B = Heap;

    fn read_page(&self, page_num: usize) -> Result<Node<Self::B>> {
        let state = self.state.lock().unwrap();
        let result = state.pages.borrow().get(&page_num).map_or(
            Err(PageStoreError::Read(
                format!("page_num {page_num} does not exist").into(),
            )),
            |n| Ok(n.clone()),
        );
        result
    }

    fn write_page(&self, node: &Node<Self::B>) -> Result<usize> {
        let state = self.state.lock().unwrap();
        let curr = state.counter.get();
        assert!(state
            .pages
            .borrow_mut()
            .insert(curr, node.clone())
            .is_none());
        state.counter.set(curr + 1);
        Ok(curr)
    }

    fn buffer_store(&self) -> &Self::B {
        &Heap
    }
}

struct FileState {
    mmap: mmap_rs::Mmap,
}

#[derive(Clone)]
struct File {
    state: Arc<FileState>,
}

impl PageStore for File {
    type B = Heap;

    fn read_page(&self, page_num: usize) -> Result<Node<Self::B>> {
        todo!()
    }

    fn write_page(&self, node: &Node<Self::B>) -> Result<usize> {
        todo!()
    }

    fn buffer_store(&self) -> &Self::B {
        &Heap
    }
}
