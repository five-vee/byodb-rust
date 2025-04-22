//! ```ignore
//! | sig | root_ptr | num_pages |
//! | 16B |    8B    |     8B    |
//! ```
use crate::error::PageError;
use std::{convert::TryFrom, ptr, rc::Rc, sync::OnceLock};

use crc::Crc;

static CRC_32_CKSUM: OnceLock<Crc<u32>> = OnceLock::new();

pub(crate) const META_PAGE_SIZE: usize = 64;
pub(crate) const META_OFFSET: usize = META_PAGE_SIZE * 2;
pub(crate) const DB_SIG: [u8; 16] = *b"BuildYourOwnDB06";
const META_NODE_SIZE: usize = std::mem::size_of::<MetaNode>();

const _: () = {
    assert!(META_NODE_SIZE <= META_PAGE_SIZE);
};

type Result<T> = std::result::Result<T, PageError>;

#[derive(Debug, PartialEq, Eq)]
pub enum Position {
    A,
    B,
}

impl Position {
    pub fn next(&self) -> Self {
        match self {
            Position::A => Position::B,
            _ => Position::A,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct MetaNode {
    pub signature: [u8; 16],
    pub root_ptr: usize,
    pub num_pages: usize,
    pub sequence: u64,

    // THIS MUST BE THE LAST FIELD.
    pub checksum: u32,
}

impl MetaNode {
    const CHECKSUM_FIELD_OFFSET: usize = memoffset::offset_of!(MetaNode, checksum);

    pub fn new() -> Self {
        let mut node = Self::default();
        node.signature = DB_SIG;
        node.num_pages = 1;
        node.checksum = node.checksum();
        node
    }

    /// Calculates the CRC32 checksum of the MetaNode's data,
    /// excluding the checksum field itself.
    pub fn checksum(&self) -> u32 {
        let node_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self as *const MetaNode as *const u8, META_NODE_SIZE)
        };
        let bytes_to_checksum = &node_bytes[0..Self::CHECKSUM_FIELD_OFFSET];
        CRC_32_CKSUM
            .get_or_init(|| Crc::<u32, crc::Table<1>>::new(&crc::CRC_32_CKSUM))
            .checksum(bytes_to_checksum)
    }

    pub fn valid(&self) -> bool {
        self.signature == DB_SIG && self.checksum == self.checksum()
    }

    pub fn read_last_valid_meta_node(slice: &[u8]) -> (Self, Position) {
        assert!(slice.len() >= META_OFFSET, "no valid meta node found");
        let node_a: MetaNode = slice[..META_PAGE_SIZE].try_into().unwrap();
        let node_b: MetaNode = slice[META_PAGE_SIZE..META_OFFSET].try_into().unwrap();
        let (node, pos) = match (node_a, node_b) {
            (a, b) if !a.valid() && !b.valid() => {
                panic!("no valid meta node found")
            }
            (a, b) if a.valid() && !b.valid() => (a, Position::A),
            (a, b) if !a.valid() && b.valid() => (b, Position::B),
            _ => {
                let a = node_a.sequence;
                let b = node_b.sequence;
                if a >= b {
                    (node_a, Position::A)
                } else {
                    (node_b, Position::B)
                }
            }
        };
        (node, pos)
    }

    pub fn copy_to_slice(&self, slice: &mut [u8], pos: Position) {
        let offset: usize = match pos {
            Position::A => 0,
            Position::B => META_PAGE_SIZE,
        };
        let page: [u8; META_PAGE_SIZE] = self.into();
        slice[offset..offset + META_PAGE_SIZE].copy_from_slice(&page);
    }
}

impl TryFrom<&[u8]> for MetaNode {
    type Error = PageError;

    fn try_from(value: &[u8]) -> Result<Self> {
        if value.len() < META_NODE_SIZE {
            return Err(PageError::InvalidFile(Rc::from(format!(
                "Input slice too small for MetaNode. Expected at least {} bytes, got {}",
                META_NODE_SIZE,
                value.len()
            ))));
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
            signature: DB_SIG,
            root_ptr: 1024,
            num_pages: 5,
            sequence: 2,
            checksum: 1, // dummy
        };
        // Create the buffer using the From implementation
        let buffer: [u8; META_PAGE_SIZE] = (&original_meta).into();

        let result: Result<MetaNode> = buffer[..].try_into();
        assert!(result.is_ok());
        let converted_meta = result.unwrap();

        assert_eq!(converted_meta, original_meta);
    }

    #[test]
    fn test_meta_node_new_is_valid() {
        let node = MetaNode::new();
        // Check default values and signature
        assert_eq!(node.signature, DB_SIG);
        assert_eq!(node.root_ptr, 0);
        assert_eq!(node.num_pages, 1);
        assert_eq!(node.sequence, 0);
        // Checksum should be calculated correctly, making the node valid
        assert!(node.valid());
    }

    #[test]
    fn test_meta_node_valid_true() {
        let node = MetaNode::new();
        assert!(node.valid());
    }

    #[test]
    fn test_meta_node_valid_false_sig() {
        let mut node = MetaNode::new();
        node.signature[0] = 0; // Invalidate signature
        assert!(!node.valid());
    }

    #[test]
    fn test_meta_node_valid_false_checksum() {
        let mut node = MetaNode::new();
        node.checksum = node.checksum.wrapping_add(1); // Invalidate checksum
        assert!(!node.valid());
    }

    // Helper to create a valid node with a specific sequence number
    fn create_valid_node(sequence: u64) -> MetaNode {
        let mut node = MetaNode {
            signature: DB_SIG,
            root_ptr: 1,
            num_pages: 1,
            sequence,
            checksum: 0, // Placeholder
        };
        node.checksum = node.checksum(); // Calculate correct checksum
        node
    }

    // Helper to create an invalid node (bad signature)
    fn create_invalid_node(sequence: u64) -> MetaNode {
        let mut node = create_valid_node(sequence);
        node.signature[0] = 0; // Invalidate signature
                               // Checksum is now wrong, but signature is the primary invalidation here
        node
    }

    #[test]
    #[should_panic]
    fn test_read_last_valid_meta_node_too_small() {
        let buffer = [0u8; META_OFFSET - 1];
        let _ = MetaNode::read_last_valid_meta_node(&buffer);
    }

    #[test]
    #[should_panic]
    fn test_read_last_valid_meta_node_none_valid() {
        let mut buffer = [0u8; META_OFFSET];
        let invalid_node_a = create_invalid_node(1);
        let invalid_node_b = create_invalid_node(2);
        buffer[..META_PAGE_SIZE].copy_from_slice(&<[u8; META_PAGE_SIZE]>::from(&invalid_node_a));
        buffer[META_PAGE_SIZE..META_OFFSET]
            .copy_from_slice(&<[u8; META_PAGE_SIZE]>::from(&invalid_node_b));

        let _ = MetaNode::read_last_valid_meta_node(&buffer);
    }

    #[test]
    fn test_read_last_valid_meta_node_only_a_valid() {
        let mut buffer = [0u8; META_OFFSET];
        let valid_node_a = create_valid_node(1);
        let invalid_node_b = create_invalid_node(2);
        buffer[..META_PAGE_SIZE].copy_from_slice(&<[u8; META_PAGE_SIZE]>::from(&valid_node_a));
        buffer[META_PAGE_SIZE..META_OFFSET]
            .copy_from_slice(&<[u8; META_PAGE_SIZE]>::from(&invalid_node_b));

        let result = MetaNode::read_last_valid_meta_node(&buffer);
        assert_eq!(result, (valid_node_a, Position::A));
    }

    #[test]
    fn test_read_last_valid_meta_node_only_b_valid() {
        let mut buffer = [0u8; META_OFFSET];
        let invalid_node_a = create_invalid_node(1);
        let valid_node_b = create_valid_node(2);
        buffer[..META_PAGE_SIZE].copy_from_slice(&<[u8; META_PAGE_SIZE]>::from(&invalid_node_a));
        buffer[META_PAGE_SIZE..META_OFFSET]
            .copy_from_slice(&<[u8; META_PAGE_SIZE]>::from(&valid_node_b));

        let result = MetaNode::read_last_valid_meta_node(&buffer);
        assert_eq!(result, (valid_node_b, Position::B));
    }

    #[test]
    fn test_read_last_valid_meta_node_both_valid_a_newer() {
        let mut buffer = [0u8; META_OFFSET];
        let valid_node_a = create_valid_node(5); // Newer sequence
        let valid_node_b = create_valid_node(2);
        buffer[..META_PAGE_SIZE].copy_from_slice(&<[u8; META_PAGE_SIZE]>::from(&valid_node_a));
        buffer[META_PAGE_SIZE..META_OFFSET]
            .copy_from_slice(&<[u8; META_PAGE_SIZE]>::from(&valid_node_b));

        let result = MetaNode::read_last_valid_meta_node(&buffer);
        assert_eq!(result, (valid_node_a, Position::A));
    }

    #[test]
    fn test_read_last_valid_meta_node_both_valid_b_newer() {
        let mut buffer = [0u8; META_OFFSET];
        let valid_node_a = create_valid_node(2);
        let valid_node_b = create_valid_node(5); // Newer sequence
        buffer[..META_PAGE_SIZE].copy_from_slice(&<[u8; META_PAGE_SIZE]>::from(&valid_node_a));
        buffer[META_PAGE_SIZE..META_OFFSET]
            .copy_from_slice(&<[u8; META_PAGE_SIZE]>::from(&valid_node_b));

        let result = MetaNode::read_last_valid_meta_node(&buffer);
        assert_eq!(result, (valid_node_b, Position::B));
    }

    #[test]
    fn test_read_last_valid_meta_node_both_valid_equal_seq() {
        let mut buffer = [0u8; META_OFFSET];
        let valid_node_a = create_valid_node(5); // Equal sequence
        let valid_node_b = create_valid_node(5); // Equal sequence
        buffer[..META_PAGE_SIZE].copy_from_slice(&<[u8; META_PAGE_SIZE]>::from(&valid_node_a));
        buffer[META_PAGE_SIZE..META_OFFSET]
            .copy_from_slice(&<[u8; META_PAGE_SIZE]>::from(&valid_node_b));

        let result = MetaNode::read_last_valid_meta_node(&buffer);
        // Node A should be preferred when sequences are equal
        assert_eq!(result, (valid_node_a, Position::A));
    }

    #[test]
    fn test_meta_node_from_into_bytes() {
        let original_node = create_valid_node(123);

        // Convert node to byte buffer
        let buffer: [u8; META_PAGE_SIZE] = (&original_node).into();

        // Check that the first META_NODE_SIZE bytes convert back correctly
        let converted_node_result: Result<MetaNode> = buffer[..META_NODE_SIZE].try_into();
        assert!(converted_node_result.is_ok());
        assert_eq!(converted_node_result.unwrap(), original_node);

        // Check that the remaining padding bytes are zero
        for &byte in buffer[META_NODE_SIZE..].iter() {
            assert_eq!(byte, 0, "Padding bytes should be zero");
        }
    }

    #[test]
    fn test_meta_node_checksum_calculation() {
        // Create a node with known data, checksum field doesn't matter here
        let node = MetaNode {
            signature: DB_SIG,
            root_ptr: 12345,
            num_pages: 678,
            sequence: 91011,
            checksum: 999, // This value is ignored by checksum()
        };

        // Manually calculate the expected checksum based on the fields
        let node_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(&node as *const MetaNode as *const u8, META_NODE_SIZE)
        };
        let bytes_to_checksum = &node_bytes[0..MetaNode::CHECKSUM_FIELD_OFFSET];
        let crc_instance = Crc::<u32, crc::Table<1>>::new(&crc::CRC_32_CKSUM);
        let expected_checksum = crc_instance.checksum(bytes_to_checksum);

        // Compare with the checksum calculated by the method
        assert_eq!(node.checksum(), expected_checksum);
    }
}
