use std::{rc::Rc, sync::Arc};

use anyhow::{Context, Result};
use divan::{Bencher, black_box};
use rand::{
    SeedableRng,
    distr::{Alphabetic, SampleString},
    prelude::*,
};
use rand_chacha::ChaCha8Rng;
use tempfile::NamedTempFile;

use byodb_rust::{
    DB, DBBuilder, consts,
    error::{NodeError, TreeError, TxnError},
};

const DEFAULT_SEED: u64 = 1;

fn main() {
    divan::main()
}

fn new_test_db() -> (DB, NamedTempFile) {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    let db = DBBuilder::new(path).build().unwrap();
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
                Err(TxnError::Tree(TreeError::Node(NodeError::AlreadyExists)))
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

#[divan::bench(threads = [1, 2, 4], args = [1000, 4000, 10000, 40000])]
fn bench_readers(b: Bencher, n: usize) {
    let (db, _temp_file) = new_test_db();
    Seeder::new(n, DEFAULT_SEED).seed_db(&db).unwrap();
    b.counter(n).bench({
        move || {
            let t = db.r_txn();
            for (k, v) in t.in_order_iter() {
                let (_k, _v) = (black_box(k), black_box(v));
            }
        }
    });
}

#[divan::bench(threads = [1, 2, 4], args = [1000, 4000, 10000, 40000])]
fn bench_writer_and_readers(b: Bencher, n: usize) {
    // Setup.
    let (db, _temp_file) = new_test_db();
    let db = Arc::new(db);
    Seeder::new(n, DEFAULT_SEED).seed_db(&db).unwrap();

    // Have a background rw txn aimlessly spinning.
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::thread;
    let (sender, receiver): (Sender<()>, Receiver<()>) = mpsc::channel();
    let background_thread = thread::spawn({
        let db = db.clone();
        move || {
            let mut t = db.rw_txn();
            // Get one key.
            let (k, _) = t.in_order_iter().next().unwrap();
            let k: Rc<[u8]> = k.into();
            let dummy_val = [1u8; 100];
            // Mindlessly do some busy work until termination.
            while receiver.try_recv().is_err() {
                t.update(&k, &dummy_val).unwrap();
            }
            t.abort();
        }
    });

    // Run the readers.
    b.counter(n).bench({
        let db = db.clone();
        move || {
            let t = db.r_txn();
            for (k, v) in t.in_order_iter() {
                let (_k, _v) = (black_box(k), black_box(v));
            }
        }
    });

    // Cleanup.
    sender.send(()).unwrap();
    background_thread.join().unwrap();
}
