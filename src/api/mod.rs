//! Provides the public API for interacting with the database.
//!
//! This module defines the main `DB` struct for database instantiation and
//! transaction management, as well as the `Txn` struct for performing
//! read (and possibly write) operations within a transaction.

pub mod error;

use std::{marker::PhantomData, ops::RangeBounds, path::Path, sync::Arc};

use error::TxnError;
use seize::Collector;

pub use crate::core::consts;
use crate::core::{
    mmap::{self, Guard, ImmutablePage, Mmap, Reader, ReaderPage, Store, Writer, WriterPage},
    tree::Tree,
};

/// A specialized `Result` type for database operations within this module.
pub type Result<T> = std::result::Result<T, TxnError>;
/// A read-write transaction. It must live as long as the [`DB`] that created
/// it (via [`DB::rw_txn`]).
pub type RWTxn<'t, 'd> = Txn<'t, WriterPage<'t, 'd>, Writer<'d>>;
/// A read-only transaction. It must live as long as the [`DB`] that created
/// it (via [`DB::r_txn`]).
pub type RTxn<'t, 'd> = Txn<'t, ReaderPage<'t>, Reader<'d>>;

/// Represents the main database instance.
///
/// `DB` is the entry point for all database interactions. It handles the
/// underlying storage and provides methods to create read-only or read-write
/// transactions.
#[derive(Clone)]
pub struct DB {
    store: Arc<Store>,
}

impl DB {
    /// Opens an existing database file or creates a new one if it doesn't exist.
    /// Uses default settings. If you need customization, use
    /// [`DBBuilder::new`] instead.
    ///
    /// # Parameters
    /// - `path`: A path to the database file.
    ///
    /// # Errors
    /// Returns [`TxnError`] if the database file cannot be opened or created,
    /// or if there's an issue initializing the memory map.
    pub fn open_or_create<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(DB {
            store: Arc::new(Store::new(
                Mmap::open_or_create(path, mmap::DEFAULT_MIN_FILE_GROWTH_SIZE)?,
                Collector::new(),
            )),
        })
    }

    /// Begins a new read-only transaction.
    ///
    /// Read-only transactions allow for concurrent reads without blocking
    /// other readers or writers.
    ///
    /// # Returns
    /// A [`RTxn`] instance for performing read operations.
    pub fn r_txn(&self) -> RTxn<'_, '_> {
        let reader = self.store.reader();
        let root_page = reader.root_page();
        Txn {
            _phantom: PhantomData,
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
    /// A [`RWTxn`] instance for performing read and write operations.
    pub fn rw_txn(&self) -> RWTxn<'_, '_> {
        let writer = self.store.writer();
        let root_page = writer.root_page();
        Txn {
            _phantom: PhantomData,
            guard: writer,
            root_page,
        }
    }
}

/// Builder of a [`DB`].
pub struct DBBuilder<P: AsRef<Path>> {
    db_path: P,
    pub free_batch_size: Option<usize>,
    min_file_growth_size: usize,
}

impl<P: AsRef<Path>> DBBuilder<P> {
    /// Creates a new [`DB`] from `db_path`.
    pub fn new(db_path: P) -> Self {
        let free_batch_size = if cfg!(test) { Some(1) } else { None };
        DBBuilder {
            db_path,
            free_batch_size,
            min_file_growth_size: mmap::DEFAULT_MIN_FILE_GROWTH_SIZE,
        }
    }

    /// Sets the number of free pages before reclamation can be attempted.
    /// The default is 32, though this can change in the future.
    pub fn free_batch_size(self, val: usize) -> Self {
        DBBuilder {
            db_path: self.db_path,
            free_batch_size: Some(val),
            min_file_growth_size: self.min_file_growth_size,
        }
    }

    /// Sets the minimum file growth size.
    /// The default is 64MB in release, though this can change in the future.
    pub fn min_file_growth_size(self, val: usize) -> Self {
        DBBuilder {
            db_path: self.db_path,
            free_batch_size: self.free_batch_size,
            min_file_growth_size: val,
        }
    }

    /// Builds a DB.
    pub fn build(self) -> Result<DB> {
        let collector = match &self.free_batch_size {
            &Some(size) => Collector::new().batch_size(size),
            None => Collector::new(),
        };
        Ok(DB {
            store: Arc::new(Store::new(
                Mmap::open_or_create(self.db_path, self.min_file_growth_size)?,
                collector,
            )),
        })
    }
}

/// Represents a database transaction.
///
/// A transaction provides a consistent view of the database. It can be
/// either read-only ([`RTxn`]) or read-write ([`RWTxn`]).
/// All operations on the database are performed within a transaction.
pub struct Txn<'g, P: ImmutablePage<'g>, G: Guard<'g, P>> {
    _phantom: PhantomData<&'g P>,
    guard: G,
    root_page: usize,
}

impl<'g, P: ImmutablePage<'g>, G: Guard<'g, P>> Txn<'g, P, G> {
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
    pub fn get(&'g self, key: &[u8]) -> Result<Option<&'g [u8]>> {
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
    pub fn in_order_iter(&'g self) -> impl Iterator<Item = (&'g [u8], &'g [u8])> {
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
        &'g self,
        range: &R,
    ) -> impl Iterator<Item = (&'g [u8], &'g [u8])> {
        Tree::new(&self.guard, self.root_page).in_order_range_iter(range)
    }
}

impl<'t, 'd> Txn<'t, WriterPage<'t, 'd>, Writer<'d>> {
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
    #[inline]
    pub fn commit(self) {
        self.guard.flush(self.root_page);
    }

    /// Aborts the transaction, discarding all changes made within it.
    ///
    /// This is automatically called if the transaction goes out of scope
    /// without [`Txn::commit`] being called.
    #[inline]
    pub fn abort(self) {
        self.guard.abort();
    }
}

#[cfg(test)]
mod tests {
    use core::str;
    use std::collections::HashSet;
    use std::ops::{Bound, Range};

    use anyhow::{Context, Result};
    use rand::distr::{Alphabetic, SampleString as _};
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;
    use tempfile::NamedTempFile;

    use super::{
        error::{NodeError, TreeError},
        *,
    };

    const DEFAULT_SEED: u64 = 1;
    const DEFAULT_NUM_SEEDED_KEY_VALS: usize = 1000;

    fn new_test_db() -> (DB, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        let db = DBBuilder::new(path).free_batch_size(1).build().unwrap();
        (db, temp_file)
    }

    struct Seeder {
        n: usize,
        rng: ChaCha8Rng,
    }

    impl Seeder {
        fn new(n: usize, seed: u64) -> Self {
            Seeder {
                n,
                rng: ChaCha8Rng::seed_from_u64(seed),
            }
        }

        fn seed_db(self, db: &DB) -> Result<()> {
            let mut t = db.rw_txn();
            for (i, (k, v)) in self.enumerate() {
                let result = t.insert(k.as_bytes(), v.as_bytes());
                if matches!(
                    result,
                    Err(TxnError::Tree(TreeError::AlreadyExists))
                ) {
                    // Skip
                    continue;
                }
                result.with_context(|| format!("failed to insert {i}th ({k}, {v})"))?;
            }
            t.commit();
            Ok(())
        }
    }

    impl Iterator for Seeder {
        type Item = (String, String);
        fn next(&mut self) -> Option<Self::Item> {
            if self.n == 0 {
                return None;
            }
            self.n -= 1;
            let key_len = self.rng.random_range(1..=consts::MAX_KEY_SIZE);
            let val_len = self.rng.random_range(1..=consts::MAX_VALUE_SIZE);
            let key: String = Alphabetic.sample_string(&mut self.rng, key_len);
            let val: String = Alphabetic.sample_string(&mut self.rng, val_len);
            Some((key, val))
        }
    }

    fn u64_to_key(i: u64) -> [u8; consts::MAX_KEY_SIZE] {
        let mut key = [0u8; consts::MAX_KEY_SIZE];
        key[0..8].copy_from_slice(&i.to_be_bytes());
        key
    }

    #[test]
    fn test_insert() {
        let (db, _temp_file) = new_test_db();
        Seeder::new(DEFAULT_NUM_SEEDED_KEY_VALS, DEFAULT_SEED)
            .seed_db(&db)
            .unwrap();
        let kvs = Seeder::new(DEFAULT_NUM_SEEDED_KEY_VALS, DEFAULT_SEED)
            .collect::<HashSet<(String, String)>>();
        let t = db.r_txn();
        for (k, v) in kvs {
            match t.get(k.as_bytes()) {
                Err(err) => panic!("get({k}) unexpectedly got err {err}"),
                Ok(None) => panic!("get({k}) unexpectedly got None"),
                Ok(Some(got)) => {
                    let got = str::from_utf8(got).expect("get({k}) is a alphabetic string");
                    assert_eq!(got, v.as_str(), "get({k}) got = {got}, want = {v}");
                }
            }
        }
        // Verify equal height invariant.
        Tree::new(&t.guard, t.root_page).check_height().unwrap();
    }

    #[test]
    fn test_update() {
        let (db, _temp_file) = new_test_db();
        Seeder::new(DEFAULT_NUM_SEEDED_KEY_VALS, DEFAULT_SEED)
            .seed_db(&db)
            .unwrap();
        let ks = Seeder::new(DEFAULT_NUM_SEEDED_KEY_VALS, DEFAULT_SEED)
            .map(|(k, _)| k)
            .collect::<HashSet<_>>();
        let updated_val = [1u8; consts::MAX_VALUE_SIZE];
        {
            let mut t = db.rw_txn();
            for k in ks.iter() {
                t.update(k.as_bytes(), &updated_val)
                    .unwrap_or_else(|_| panic!("update({k}, &updated_val) should succeed"));
            }
            t.commit();
        }
        {
            let t = db.r_txn();
            for k in ks.iter() {
                match t.get(k.as_bytes()) {
                    Err(err) => panic!("get({k}) unexpectedly got err {err}"),
                    Ok(None) => panic!("get({k}) unexpectedly got None"),
                    Ok(Some(got)) => {
                        assert_eq!(
                            got, &updated_val,
                            "get({k}) got = {got:?}, want = {updated_val:?}"
                        );
                    }
                }
            }
            // Verify equal height invariant.
            Tree::new(&t.guard, t.root_page).check_height().unwrap();
        }
    }

    #[test]
    fn test_delete() {
        let (db, _temp_file) = new_test_db();
        Seeder::new(DEFAULT_NUM_SEEDED_KEY_VALS, DEFAULT_SEED)
            .seed_db(&db)
            .unwrap();
        let ks = Seeder::new(DEFAULT_NUM_SEEDED_KEY_VALS, DEFAULT_SEED)
            .map(|(k, _)| k)
            .collect::<HashSet<_>>();
        let mut t = db.rw_txn();
        for k in ks.iter() {
            if let Err(err) = t.delete(k.as_bytes()) {
                panic!("delete({k}) unexpectedly got err {err}");
            }
            match t.get(k.as_bytes()) {
                Err(err) => panic!("get({k}) after delete() unexpectedly got err {err}"),
                Ok(Some(v)) => {
                    panic!("get({k}) after delete() unexpectedly got = Some({v:?}), want = None")
                }
                _ => {}
            };
        }
        t.commit();
        let t = db.r_txn();
        for k in ks.iter() {
            match t.get(k.as_bytes()) {
                Err(err) => panic!("get({k}) after delete() unexpectedly got err {err}"),
                Ok(Some(v)) => {
                    panic!("get({k}) after delete() unexpectedly got = Some({v:?}), want = None")
                }
                _ => {}
            };
        }
        // Verify equal height invariant.
        Tree::new(&t.guard, t.root_page).check_height().unwrap();
    }

    #[test]
    fn test_in_order_range_iter() {
        let (db, _temp_file) = new_test_db();
        // Setup.
        {
            let mut t = db.rw_txn();
            let mut inds = (1..=100).collect::<Vec<_>>();
            inds.shuffle(&mut rand::rng());
            for i in inds {
                let x = u64_to_key(i);
                t.insert(&x, &x).unwrap();
            }
            t.commit();
        }

        let t = db.r_txn();

        // Golang style table-driven tests.
        struct TestCase {
            name: &'static str,
            range: (Bound<&'static [u8]>, Bound<&'static [u8]>),
            want: Range<u64>,
        }
        impl Drop for TestCase {
            fn drop(&mut self) {
                for b in [self.range.0, self.range.1] {
                    match b {
                        Bound::Included(b) => {
                            drop(unsafe { Box::from_raw(b.as_ptr() as *mut u8) });
                        }
                        Bound::Excluded(b) => {
                            drop(unsafe { Box::from_raw(b.as_ptr() as *mut u8) });
                        }
                        _ => {}
                    }
                }
            }
        }
        let tests = [
            TestCase {
                name: "unbounded unbounded",
                range: (Bound::Unbounded, Bound::Unbounded),
                want: 1..101,
            },
            TestCase {
                name: "included included",
                range: (
                    Bound::Included(Box::leak(Box::new(u64_to_key(5)))),
                    Bound::Included(Box::leak(Box::new(u64_to_key(98)))),
                ),
                want: 5..99,
            },
            TestCase {
                name: "excluded included",
                range: (
                    Bound::Excluded(Box::leak(Box::new(u64_to_key(5)))),
                    Bound::Included(Box::leak(Box::new(u64_to_key(98)))),
                ),
                want: 6..99,
            },
            TestCase {
                name: "excluded excluded",
                range: (
                    Bound::Excluded(Box::leak(Box::new(u64_to_key(5)))),
                    Bound::Excluded(Box::leak(Box::new(u64_to_key(98)))),
                ),
                want: 6..98,
            },
            TestCase {
                name: "unbounded included",
                range: (
                    Bound::Unbounded,
                    Bound::Included(Box::leak(Box::new(u64_to_key(98)))),
                ),
                want: 1..99,
            },
            TestCase {
                name: "unbounded excluded",
                range: (
                    Bound::Unbounded,
                    Bound::Excluded(Box::leak(Box::new(u64_to_key(98)))),
                ),
                want: 1..98,
            },
            TestCase {
                name: "included unbounded",
                range: (
                    Bound::Included(Box::leak(Box::new(u64_to_key(5)))),
                    Bound::Unbounded,
                ),
                want: 5..101,
            },
            TestCase {
                name: "excluded unbounded",
                range: (
                    Bound::Excluded(Box::leak(Box::new(u64_to_key(5)))),
                    Bound::Unbounded,
                ),
                want: 6..101,
            },
            TestCase {
                name: "no overlap",
                range: (
                    Bound::Excluded(Box::leak(Box::new(u64_to_key(200)))),
                    Bound::Unbounded,
                ),
                want: 0..0,
            },
        ];
        for test in tests {
            let got = t
                .in_order_range_iter(&test.range)
                .map(|(k, _)| u64::from_be_bytes([k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]]))
                .collect::<Vec<_>>();
            let want = test.want.clone().collect::<Vec<_>>();
            assert_eq!(got, want, "Test case \"{}\" failed", test.name);
        }
    }
}
