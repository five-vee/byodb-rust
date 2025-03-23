use super::node::Node;
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::Rc,
};

pub type Result<T> = std::result::Result<T, ()>;

/// A store of pages that backs a COW B+ Tree.
pub trait Store {
    /// Reads a page from disk into an in-memory B+ tree node.
    fn read_page(&self, page_num: u64) -> Result<Node>;

    /// Writes an in-memory B+ tree node into a page on disk.
    fn write_page(&self, node: &Node) -> Result<u64>;
}

/// An in-memory store of pages. Backed by a hash map.
pub struct InMemory {
    pages: RefCell<HashMap<u64, Node>>,
    counter: Cell<u64>,
}

impl InMemory {
    /// Creates a new in-memory page store.
    fn new() -> Self {
        Self {
            pages: HashMap::new().into(),
            counter: Cell::new(0),
        }
    }
}

impl Store for InMemory {
    fn read_page(&self, page_num: u64) -> Result<Node> {
        self.pages
            .borrow()
            .get(&page_num)
            .map_or(Err(()), |n| Ok(n.clone()))
    }

    fn write_page(&self, node: &Node) -> Result<u64> {
        let curr = self.counter.get();
        if self.pages.borrow_mut().insert(curr, node.clone()).is_some() {
            return Err(())
        }
        self.counter.set(curr + 1);
        Ok(curr)
    }
}
