//! The `core` module contains the "core" data types and functions used for
//! manipulating the file-backed memory-mapped copy-on-write B+ tree.
//!
//! [`mmap`] is a file-backed memory-mapped region that serves as the
//! underlying data layer of the B+ [`tree`].
pub mod consts;
pub(crate) mod error;
mod header;
pub(crate) mod mmap;
pub(crate) mod tree;
