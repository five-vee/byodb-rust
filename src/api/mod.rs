pub mod error;

use std::{path::Path, sync::Arc};

use error::TxnError;

use crate::core::{
    mmap::{Guard, Mmap, Reader, Store, Writer},
    tree::Tree,
};

pub type Result<T> = std::result::Result<T, TxnError>;

pub struct DB {
    store: Arc<Store>,
}

impl DB {
    pub fn open_or_create<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(DB {
            store: Arc::new(Store::new(Mmap::open_or_create(path)?)),
        })
    }

    pub fn r_txn(&self) -> Txn<Reader> {
        let reader = self.store.reader();
        let root_page = reader.root_page();
        Txn {
            guard: reader,
            root_page,
        }
    }

    pub fn rw_txn(&self) -> Txn<Writer> {
        let writer = self.store.writer();
        let root_page = writer.root_page();
        Txn {
            guard: writer,
            root_page,
        }
    }
}

pub struct Txn<G: Guard> {
    guard: G,
    root_page: usize,
}

impl<G: Guard> Txn<G> {
    pub fn get(&self, key: &[u8]) -> Result<Option<&[u8]>> {
        let tree = Tree::new(&self.guard, self.root_page);
        let val = tree.get(key)?.map(|val| {
            // Safety: The underlying data is in the mmap, which
            // self.guard has access to. So long as self.guard exists,
            // so too does the mmap.
            unsafe { &*std::ptr::slice_from_raw_parts(val.as_ptr(), val.len()) }
        });
        Ok(val)
    }

    pub fn in_order_iter(&self) -> impl Iterator<Item = (&[u8], &[u8])> {
        Tree::new(&self.guard, self.root_page).in_order_iter()
    }
}

impl Txn<Writer<'_>> {
    pub fn insert(&mut self, key: &[u8], val: &[u8]) -> Result<()> {
        self.root_page = Tree::new(&self.guard, self.root_page)
            .insert(key, val)?
            .page_num();
        Ok(())
    }

    pub fn update(&mut self, key: &[u8], val: &[u8]) -> Result<()> {
        self.root_page = Tree::new(&self.guard, self.root_page)
            .update(key, val)?
            .page_num();
        Ok(())
    }

    pub fn delete(&mut self, key: &[u8]) -> Result<()> {
        self.root_page = Tree::new(&self.guard, self.root_page)
            .delete(key)?
            .page_num();
        Ok(())
    }

    pub fn commit(self) {
        self.guard.flush(self.root_page);
    }

    pub fn abort(self) {
        self.guard.abort();
    }
}
