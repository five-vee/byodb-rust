use crate::tree::buffer_store::Heap;
use crate::tree::page_store::{PageStore, PageStoreError};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::Rc,
    sync::{Arc, Mutex},
};

type Result<T> = std::result::Result<T, PageStoreError>;

/// An in-memory store of pages. Backed by a hash map.
#[derive(Clone)]
pub struct InMemory {
    state: Arc<Mutex<InMemoryState>>,
}

struct InMemoryState {
    pages: RefCell<HashMap<usize, Rc<[u8]>>>,
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

    fn read_page(&self, page_num: usize) -> Result<Rc<[u8]>> {
        let state = self.state.lock().unwrap();
        let result = state.pages.borrow().get(&page_num).map_or(
            Err(PageStoreError::Read(
                format!("page_num {page_num} does not exist").into(),
            )),
            |n| Ok(n.clone()),
        );
        result
    }

    fn write_page(&self, page: Rc<[u8]>) -> Result<usize> {
        let state = self.state.lock().unwrap();
        let curr = state.counter.get();
        assert!(state
            .pages
            .borrow_mut()
            .insert(curr, page.clone())
            .is_none());
        state.counter.set(curr + 1);
        Ok(curr)
    }

    fn buffer_store(&self) -> &Self::B {
        &Heap
    }
}
