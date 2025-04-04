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
    fn read_page(&self, page_num: u64) -> Result<Node<Self::B>>;

    /// Writes an in-memory B+ tree node into a page on disk.
    fn write_page(&self, node: &Node<Self::B>) -> Result<u64>;
}

/// An in-memory store of pages. Backed by a hash map.
#[derive(Clone)]
pub struct InMemory {
    state: Arc<Mutex<InMemoryState>>,
}

struct InMemoryState {
    pages: RefCell<HashMap<u64, Node<Heap>>>,
    counter: Cell<u64>,
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

    fn read_page(&self, page_num: u64) -> Result<Node<Self::B>> {
        let state = self.state.lock().unwrap();
        let result = state.pages.borrow().get(&page_num).map_or(
            Err(PageStoreError::Read(
                format!("page_num {page_num} does not exist").into(),
            )),
            |n| Ok(n.clone()),
        );
        result
    }

    fn write_page(&self, node: &Node<Self::B>) -> Result<u64> {
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
}
