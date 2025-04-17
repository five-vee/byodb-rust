use std::{io, rc::Rc};
use thiserror;

/// An error type for `mod tree`.
#[derive(thiserror::Error, Debug)]
pub enum TreeError {
    #[error("Node error: {0}")]
    Node(#[from] NodeError),
    #[error("Page store error: {0}")]
    Page(#[from] PageError),
    #[error("Key not found")]
    KeyNotFound,
}

/// An error type for `mod node`.
#[derive(thiserror::Error, Debug)]
pub enum NodeError {
    #[error("Key size exceeds maximum limit: key length {0} exceeds MAX_KEY_SIZE")]
    MaxKeySize(usize), // usize is key length
    #[error("Value size exceeds maximum limit: value length {0} exceeds MAX_VALUE_SIZE")]
    MaxValueSize(usize), // usize is value length
    #[error("Unexpected node type: {0:#b}")]
    UnexpectedNodeType(u16), // u16 is the node type
    #[error("Key already exists")]
    AlreadyExists,
    #[error("Key not found")]
    KeyNotFound,
    #[error("Page store error: {0}")]
    Page(#[from] PageError),
}

/// An error type for `mod page_store`.
#[derive(thiserror::Error, Debug)]
pub enum PageError {
    #[error("Page read error: {0}")]
    Read(Rc<str>),
    #[error("Page write error: {0}")]
    Write(Rc<str>),
    #[error("Page not found: {0}")]
    NotFound(usize),
    #[error(transparent)]
    IOError(#[from] io::Error),
    #[error(transparent)]
    MmapError(#[from] mmap_rs::Error),
    #[error("Invalid file: {0}")]
    InvalidFile(Rc<str>),
}
