pub use crate::core::error::{MmapError, NodeError, TreeError};

#[derive(thiserror::Error, Debug)]
pub enum TxnError {
    #[error("Tree error: {0}")]
    Tree(#[from] TreeError),
    #[error("Page error: {0}")]
    Mmap(#[from] MmapError),
}
