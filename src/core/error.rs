//! Errors returned by functions in the [`crate::core`] module.

use std::io;

/// An error encountered when interacting with a B+ tree.
pub type TreeError = NodeError;

/// An error encountered when interacting with a B+ tree node.
#[derive(thiserror::Error, Debug)]
pub enum NodeError {
    /// Key size exceeds maximum limit.
    #[error("Key size exceeds maximum limit: key length {0} exceeds MAX_KEY_SIZE")]
    MaxKeySize(usize), // usize is key length
    /// Value size exceeds maximum limit.
    #[error("Value size exceeds maximum limit: value length {0} exceeds MAX_VALUE_SIZE")]
    MaxValueSize(usize), // usize is value length
    /// Unexpected node type. This represents data corruption.
    #[error("Unexpected node type: {0:#b}")]
    UnexpectedNodeType(u16), // u16 is the node type
    /// Key already exists.
    #[error("Key already exists")]
    AlreadyExists,
    /// Key not found.
    #[error("Key not found")]
    KeyNotFound,
}

/// An error encountered when interacting with the memory map.
#[derive(thiserror::Error, Debug)]
pub enum MmapError {
    /// An IO error encountered when interacting with the memory-mapped file.
    #[error(transparent)]
    IOError(#[from] io::Error),
    /// The file is invalid and cannot be memory-mapped.
    #[error("Invalid file: {0}")]
    InvalidFile(String),
}
