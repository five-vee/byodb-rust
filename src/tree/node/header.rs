use crate::tree::error::NodeError;

type Result<T> = std::result::Result<T, NodeError>;

/// An enum representing a page's node type.
#[repr(u16)]
#[derive(Debug)]
pub enum NodeType {
    Leaf = 0b01u16,
    Internal = 0b10u16,
}

impl TryFrom<u16> for NodeType {
    type Error = NodeError;
    fn try_from(value: u16) -> Result<Self> {
        match value {
            0b01u16 => Ok(NodeType::Leaf),
            0b10u16 => Ok(NodeType::Internal),
            _ => Err(NodeError::UnexpectedNodeType(value)),
        }
    }
}


/// Sets the page header of a node's page buffer.
pub fn set_node_type(page: &mut [u8], node_type: NodeType) {
    page[0..2].copy_from_slice(&(node_type as u16).to_le_bytes());
}

pub fn get_node_type(page: &[u8]) -> Result<NodeType> {
    NodeType::try_from(u16::from_le_bytes([page[0], page[1]]))
}

/// Sets the number of keys in a node's page buffer.
pub fn set_num_keys(page: &mut [u8], n: usize) {
    page[2..4].copy_from_slice(&(n as u16).to_le_bytes());
}

/// Gets the number of keys in a node's page buffer.
pub fn get_num_keys(page: &[u8]) -> usize {
    u16::from_le_bytes([page[2], page[3]]) as usize
}
