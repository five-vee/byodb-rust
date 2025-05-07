use anyhow::{Context, Result};
use divan::{Bencher, black_box};
use rand::{
    SeedableRng,
    distr::{Alphabetic, SampleString},
    prelude::*,
};
use rand_chacha::ChaCha8Rng;
use tempfile::NamedTempFile;

use byodb_rust::{DB, consts};

const DEFAULT_SEED: u64 = 1;

fn main() {
    divan::main()
}

#[divan::bench(args = [1, 2, 4, 8, 16, 32])]
fn bench_fibonacci(n: u64) -> u64 {
    fibonacci(black_box(n))
}

fn fibonacci(n: u64) -> u64 {
    if n <= 1 {
        1
    } else {
        fibonacci(n - 2) + fibonacci(n - 1)
    }
}

fn new_test_db() -> (DB, NamedTempFile) {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    let db = DB::open_or_create(path).unwrap();
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

    fn seed(self, db: &DB) -> Result<()> {
        let mut t = db.rw_txn();
        for (i, (k, v)) in self.enumerate() {
            t.insert(k.as_bytes(), v.as_bytes())
                .with_context(|| format!("failed to insert {i}th ({k}, {v})"))?;
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

#[divan::bench(args = [1, 2])]
fn bench_single_reader(b: Bencher, n: usize) {
    let (db, _temp_file) = new_test_db();
    Seeder::new(n, DEFAULT_SEED).seed(&db).unwrap();
    let r = db.r_txn();
    b.counter(n).bench_local(move || {
        for (k, v) in black_box(r.in_order_iter()) {
            let (_k, _v) = (black_box(k), black_box(v));
        }
    });
}
