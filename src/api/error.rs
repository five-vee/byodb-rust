//! Errors returned by functions in this crate.
pub use crate::core::error::{MmapError, NodeError, TreeError};

/// An error that occurred during a transaction.
#[derive(thiserror::Error, Debug)]
pub enum TxnError {
    /// An error whose root cause is due to an erroneous B+ tree operation,
    /// e.g. deleting a non-existent key.
    #[error("Tree error: {0}")]
    Tree(#[from] TreeError),
    /// A system error with the memory map or its underlying file.
    #[error("Page error: {0}")]
    Mmap(#[from] MmapError),
}
