//! Provides the public API for interacting with the database.
//!
//! This module defines the main `DB` struct for database instantiation and
//! transaction management, as well as the `Txn` struct for performing
//! read (and possibly write) operations within a transaction.

pub mod error;

use std::{ops::RangeBounds, path::Path, sync::Arc};

use error::TxnError;

use crate::core::{
    mmap::{Guard, Mmap, Reader, Store, Writer},
    tree::Tree,
};

/// A specialized `Result` type for database operations within this module.
pub type Result<T> = std::result::Result<T, TxnError>;

/// Represents the main database instance.
///
/// `DB` is the entry point for all database interactions. It handles the
/// underlying storage and provides methods to create read-only or read-write
/// transactions.
pub struct DB {
    store: Arc<Store>,
}

impl DB {
    /// Opens an existing database file or creates a new one if it doesn't exist.
    ///
    /// # Parameters
    /// - `path`: A path to the database file.
    ///
    /// # Errors
    /// Returns [`TxnError`] if the database file cannot be opened or created,
    /// or if there's an issue initializing the memory map.
    pub fn open_or_create<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(DB {
            store: Arc::new(Store::new(Mmap::open_or_create(path)?)),
        })
    }

    /// Begins a new read-only transaction.
    ///
    /// Read-only transactions allow for concurrent reads without blocking
    /// other readers or writers.
    ///
    /// # Returns
    /// A `Txn<Reader>` instance for performing read operations.
    pub fn r_txn(&self) -> Txn<Reader> {
        let reader = self.store.reader();
        let root_page = reader.root_page();
        Txn {
            guard: reader,
            root_page,
        }
    }

    /// Begins a new read-write transaction.
    ///
    /// Read-write transactions provide exclusive write access. Changes made
    /// within this transaction are isolated until `commit` is called.
    ///
    /// # Returns
    /// A `Txn<Writer>` instance for performing read and write operations.
    pub fn rw_txn(&self) -> Txn<Writer> {
        let writer = self.store.writer();
        let root_page = writer.root_page();
        Txn {
            guard: writer,
            root_page,
        }
    }
}

/// Represents a database transaction.
///
/// A transaction provides a consistent view of the database. It can be
/// either read-only (`Txn<Reader>`) or read-write (`Txn<Writer>`).
/// All operations on the database are performed within a transaction.
pub struct Txn<G: Guard> {
    guard: G,
    root_page: usize,
}

impl<G: Guard> Txn<G> {
    /// Retrieves the value associated with the given key.
    ///
    /// # Parameters
    /// - `key`: The key to search for.
    ///
    /// # Returns
    /// - `Ok(Some(value))` if the key is found.
    /// - `Ok(None)` if the key is not found.
    /// - `Err(TxnError)` if an error occurs during the tree traversal.
    ///
    /// # Safety
    /// The returned slice `&[u8]` is valid as long as the transaction [`Txn`]
    /// is alive, as it points directly into the memory-mapped region.
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

    /// Returns an iterator over all key-value pairs in the database, in key order.
    ///
    /// The iterator yields tuples of `(&[u8], &[u8])` representing key-value pairs.
    pub fn in_order_iter(&self) -> impl Iterator<Item = (&[u8], &[u8])> {
        Tree::new(&self.guard, self.root_page).in_order_iter()
    }

    /// Returns an iterator over key-value pairs within the specified range, in key order.
    ///
    /// # Parameters
    /// - `range`: A range bound (e.g., `start_key..end_key`, `..end_key`, `start_key..`)
    ///   that defines the keys to include in the iteration.
    ///
    /// The iterator yields tuples of `(&[u8], &[u8])` representing key-value pairs.
    pub fn in_order_range_iter<R: RangeBounds<[u8]>>(
        &self,
        range: &R,
    ) -> impl Iterator<Item = (&[u8], &[u8])> {
        Tree::new(&self.guard, self.root_page).in_order_range_iter(range)
    }
}

impl Txn<Writer<'_>> {
    /// Inserts a new key-value pair into the database.
    ///
    /// # Parameters
    /// - `key`: The key to insert.
    /// - `val`: The value to associate with the key.
    ///
    /// # Errors
    /// Returns [`TxnError`] if the key already exists or if an error occurs
    /// during the insertion process.
    pub fn insert(&mut self, key: &[u8], val: &[u8]) -> Result<()> {
        self.root_page = Tree::new(&self.guard, self.root_page)
            .insert(key, val)?
            .page_num();
        Ok(())
    }

    /// Updates the value associated with an existing key.
    ///
    /// # Parameters
    /// - `key`: The key whose value is to be updated.
    /// - `val`: The new value to associate with the key.
    ///
    /// # Errors
    /// Returns [`TxnError`] if the key does not exist or if an error occurs
    /// during the update process.
    pub fn update(&mut self, key: &[u8], val: &[u8]) -> Result<()> {
        self.root_page = Tree::new(&self.guard, self.root_page)
            .update(key, val)?
            .page_num();
        Ok(())
    }

    /// Deletes a key-value pair from the database.
    ///
    /// # Parameters
    /// - `key`: The key to delete.
    ///
    /// # Errors
    /// Returns [`TxnError`] if the key does not exist or if an error occurs
    /// during the deletion process.
    pub fn delete(&mut self, key: &[u8]) -> Result<()> {
        self.root_page = Tree::new(&self.guard, self.root_page)
            .delete(key)?
            .page_num();
        Ok(())
    }

    /// Commits the transaction, making all changes permanent and visible
    /// to subsequent transactions.
    ///
    /// If `commit` is not called, the transaction will be automatically
    /// aborted when it goes out of scope.
    pub fn commit(self) {
        self.guard.flush(self.root_page);
    }

    /// Aborts the transaction, discarding all changes made within it.
    ///
    /// This is automatically called if the transaction goes out of scope
    /// without [`Txn::commit`] being called.
    pub fn abort(self) {
        self.guard.abort();
    }
}
