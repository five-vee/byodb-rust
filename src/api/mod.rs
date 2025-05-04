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

    pub fn rw_txn(&self) -> RWTxn<'_> {
        let writer = self.store.writer();
        let curr_root_page = writer.root_page();
        RWTxn {
            writer,
            curr_root_page,
        }
    }

    pub fn r_txn(&self) -> RTxn<'_> {
        let reader = self.store.reader();
        let root_page = reader.root_page();
        RTxn { reader, root_page }
    }
}

pub struct RWTxn<'s> {
    writer: Writer<'s>,
    curr_root_page: usize,
}

impl RWTxn<'_> {
    pub fn get(&self, key: &[u8]) -> Result<Option<&[u8]>> {
        let tree = Tree::new_at(&self.writer, self.curr_root_page);
        let val = tree.get(key)?.map(|val| {
            // Safety: The underlying data is in the mmap, which
            // self.writer has access to. So long as self.writer exists,
            // so too does the mmap.
            unsafe { &*std::ptr::slice_from_raw_parts(val.as_ptr(), val.len()) }
        });
        Ok(val)
    }

    pub fn insert(&mut self, key: &[u8], val: &[u8]) -> Result<()> {
        self.curr_root_page = Tree::new_at(&self.writer, self.curr_root_page)
            .insert(key, val)?
            .page_num();
        Ok(())
    }

    pub fn update(&mut self, key: &[u8], val: &[u8]) -> Result<()> {
        self.curr_root_page = Tree::new_at(&self.writer, self.curr_root_page)
            .update(key, val)?
            .page_num();
        Ok(())
    }

    pub fn delete(&mut self, key: &[u8]) -> Result<()> {
        self.curr_root_page = Tree::new_at(&self.writer, self.curr_root_page)
            .delete(key)?
            .page_num();
        Ok(())
    }

    pub fn commit(self) {
        self.writer.flush(self.curr_root_page);
    }

    pub fn abort(self) {
        self.writer.abort();
    }
}

pub struct RTxn<'s> {
    reader: Reader<'s>,
    root_page: usize,
}

impl RTxn<'_> {
    pub fn get(&self, key: &[u8]) -> Result<Option<&[u8]>> {
        let tree = Tree::new_at(&self.reader, self.root_page);
        let val = tree.get(key)?.map(|val| {
            // Safety: The underlying data is in the mmap, which
            // self.writer has access to. So long as self.writer exists,
            // so too does the mmap.
            unsafe { &*std::ptr::slice_from_raw_parts(val.as_ptr(), val.len()) }
        });
        Ok(val)
    }
}
