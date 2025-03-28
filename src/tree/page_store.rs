use super::buffer_store::HeapStore;
use super::{buffer_store::BufferStore, error::PageStoreError};
use super::node::Node;
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::Rc,
};

type Result<T> = std::result::Result<T, PageStoreError>;

/// A store of pages that backs a COW B+ Tree.
pub trait PageStore<B: BufferStore> : Clone {
    /// Reads a page from disk into an in-memory B+ tree node.
    fn read_page(&self, page_num: u64) -> Result<Node<B>>;

    /// Writes an in-memory B+ tree node into a page on disk.
    fn write_page(&self, node: &Node<B>) -> Result<u64>;
}

/// An in-memory store of pages. Backed by a hash map.
#[derive(Clone)]
pub struct InMemory {
    state: Rc<InMemoryState>,
}

struct InMemoryState {
    pages: RefCell<HashMap<u64, Node<HeapStore>>>,
    counter: Cell<u64>,
}

impl InMemory {
    /// Creates a new in-memory page store.
    fn new() -> Self {
        Self {
            state: InMemoryState {
                pages: HashMap::new().into(),
                counter: Cell::new(0),
            }
            .into(),
        }
    }
}

impl PageStore<HeapStore> for InMemory {
    fn read_page(&self, page_num: u64) -> Result<Node<HeapStore>> {
        self.state.pages.borrow().get(&page_num).map_or(
            Err(PageStoreError::Read(
                format!("page_num {page_num} does not exist").into(),
            )),
            |n| Ok(n.clone()),
        )
    }

    fn write_page(&self, node: &Node<HeapStore>) -> Result<u64> {
        let curr = self.state.counter.get();
        assert!(self.state.pages.borrow_mut().insert(curr, node.clone()).is_none());
        self.state.counter.set(curr + 1);
        Ok(curr)
    }
}
