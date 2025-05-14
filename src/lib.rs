//! # Build Your Own Database from Scratch (in Rust)
//!
//! [https://build-your-own.org/database/](https://build-your-own.org/database/),
//! but instead of Go, use Rust. This a personal project to learn both database
//! internals and the Rust programming language.
//!
//! ## Example
//!
//! ```rust,no_run
//! # use byodb_rust::{DB, Result, RTxn, RWTxn};
//! # fn main() -> Result<()> {
//! let path = "/path/to/a/db/file";
//! let db: DB = DB::open_or_create(path)?;
//!
//! // Perform reads in a read transaction.
//! {
//!     let r_txn: RTxn<'_, '_> = db.r_txn();
//!     for (k, v) in r_txn.in_order_iter() {
//!         println!("key: {k:?}, val: {v:?}");
//!     }
//! } // read transaction is dropped at the end of scope.
//!
//! // Perform reads and writes in a read-write transaction.
//! {
//!     let mut rw_txn: RWTxn<'_, '_> = db.rw_txn();
//!     if rw_txn.get("some_key".as_bytes())?.is_some() {
//!         rw_txn.update("some_key".as_bytes(), "some_new_val".as_bytes())?;
//!     }
//!     // If rw_txn.commit() is not called b/c there was error in any of the
//!     // above steps, then when rw_txn is dropped, it is equivalent to doing
//!     // rw_txn.abort().
//!     rw_txn.commit();
//! }
//! # Ok(())
//! # }
//! ```
mod api;
mod core;
pub mod file_util;

pub use api::*;
