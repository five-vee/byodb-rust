//! The [`MetaNode`] is special node that is not part of the B+ tree.
//! Rather, it contains metadata about the B+ tree, including where
//! the root node is located, how many nodes/pages are currently
//! there (including those not yet reclaimed into the [`FreeList`]), etc.
//!
//! The meta node has the following format on disk:
//!
//! ```text
//! | root_page | num_pages | head_page | head_seq | tail_page | tail_seq |
//! |     8B    |     8B    |    8B     |    8B    |     8B    |    8B    |
//! ```
use crate::core::error::MmapError;
use std::{convert::TryFrom, ptr};

use super::free_list::FreeList;

/// Size of a meta node as stored on disk.
/// This MUST be at most the size of a disk sector to guarantee write atomicity
/// of a meta page without having to rely on double buffering.
pub(crate) const META_PAGE_SIZE: usize = 64;
const META_NODE_SIZE: usize = std::mem::size_of::<MetaNode>();

const _: () = {
    assert!(META_NODE_SIZE <= META_PAGE_SIZE);
};

type Result<T> = std::result::Result<T, MmapError>;

/// The meta node is special node that is not part of the B+ tree.
/// Rather, it contains metadata about the B+ tree, including where
/// the root node is located, how many nodes/pages are currently
/// there (including those not yet reclaimed into the free list), etc.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct MetaNode {
    /// Which page represents the root of the B+ tree.
    pub root_page: usize,

    /// How many pages are utilized by the memory map.
    /// Note that `num_pages` is NOT the number of nodes/pages in the B+ tree.
    /// Rather, it is a sort of watermark that monotonically increases over
    /// time with new write operations on the B+ tree, as it grows. If there
    /// are no free nodes in the free list, a new page can be allocated, and so
    /// will the `num_pages` counter.
    pub num_pages: usize,

    /// Which page represents the head node of the free list.
    pub head_page: usize,

    /// A monotonically increasing sequence number representing the index into
    /// the free list that represents the first free page pointed to by the
    /// head node.
    pub head_seq: usize,

    /// Which page represents the tail node of the free list.
    pub tail_page: usize,

    /// A monotonically increasing sequence number representing the index into
    /// the free list that represents the last free page pointed to by the
    /// tail node.
    pub tail_seq: usize,
}

impl MetaNode {
    /// Creates a new meta node from the supplied parameters.
    pub fn new(root_page: usize, num_pages: usize, fl: &FreeList) -> Self {
        MetaNode {
            root_page,
            num_pages,
            head_page: fl.head_page,
            head_seq: fl.head_seq,
            tail_page: fl.tail_page,
            tail_seq: fl.tail_seq,
        }
    }

    /// Writes a meta node to beginning of the slice.
    pub fn copy_to_slice(&self, slice: &mut [u8]) {
        let page: [u8; META_PAGE_SIZE] = self.into();
        slice[0..META_PAGE_SIZE].copy_from_slice(&page);
    }
}

impl Default for MetaNode {
    /// Creates a new meta node representing a completely empty mmap file that
    /// has 2 pages:
    /// 1. the B+ tree as a leaf node with no key-values
    /// 2. the initial empty free list node
    fn default() -> Self {
        Self::new(0, 2, &FreeList::default())
    }
}

impl TryFrom<&[u8]> for MetaNode {
    type Error = MmapError;

    fn try_from(value: &[u8]) -> Result<Self> {
        if value.len() < META_PAGE_SIZE {
            return Err(MmapError::InvalidFile(format!(
                "Input slice too small for MetaNode. Expected at least {} bytes, got {}",
                META_PAGE_SIZE,
                value.len()
            )));
        }

        let mut meta_node = MetaNode::default();

        // SAFETY: We've checked that value.len() >= node_size.
        // Pointers are valid and derived from slices/struct.
        // Copy the data from the slice into the struct.
        unsafe {
            ptr::copy_nonoverlapping(
                value.as_ptr(),
                &mut meta_node as *mut MetaNode as *mut u8,
                META_NODE_SIZE,
            );
        }
        Ok(meta_node)
    }
}

impl<'a> From<&'a MetaNode> for [u8; META_PAGE_SIZE] {
    fn from(node: &'a MetaNode) -> Self {
        let mut buffer = [0u8; META_PAGE_SIZE];

        // SAFETY: We've asserted that node_size <= META_SIZE.
        // Pointers are valid and derived from struct/buffer.
        // Copy the struct data to the beginning of the buffer.
        unsafe {
            ptr::copy_nonoverlapping(
                node as *const MetaNode as *const u8,
                buffer.as_mut_ptr(),
                META_NODE_SIZE,
            );
        }

        // The rest of the buffer (buffer[node_size..]) remains zeroed
        // as initialized, fulfilling the requirement to fill META_SIZE.
        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;

    #[test]
    fn test_meta_node_try_from_valid() {
        let original_meta = MetaNode {
            root_page: 1024,
            num_pages: 5,
            ..Default::default()
        };
        // Create the buffer using the From implementation
        let buffer: [u8; META_PAGE_SIZE] = (&original_meta).into();

        let result: Result<MetaNode> = buffer[..].try_into();
        assert!(result.is_ok());
        let converted_meta = result.unwrap();

        assert_eq!(converted_meta, original_meta);
    }

    #[test]
    fn test_try_from_too_small() {
        let buffer = [0u8; META_PAGE_SIZE - 1];
        assert!(matches!(
            MetaNode::try_from(&buffer[..]),
            Err(MmapError::InvalidFile(_))
        ));
    }

    #[test]
    fn test_meta_node_from_into_bytes() {
        let original_node = MetaNode::default();

        // Convert node to byte buffer
        let buffer: [u8; META_PAGE_SIZE] = (&original_node).into();

        // Check that the first META_NODE_SIZE bytes convert back correctly
        let converted_node: MetaNode = buffer[..META_PAGE_SIZE].try_into().unwrap();
        assert_eq!(converted_node, original_node);

        // Check that the remaining padding bytes are zero
        for &byte in buffer[META_NODE_SIZE..].iter() {
            assert_eq!(byte, 0, "Padding bytes should be zero");
        }
    }
}
