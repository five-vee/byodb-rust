//! # Design a node format
//!
//! Here is our node format. The 2nd row is the encoded field size in bytes.
//!
//! ```ignore
//! | type | nkeys |  pointers* |  offsets   | key-values** | unused |
//! |  2B  |   2B  | nkeys × 8B | nkeys × 2B |     ...      |        |
//!
//! *  pointers is omitted for leaf nodes.
//! ** values are omitted for internal nodes.
//! ```
//!
//! The format starts with a 4-bytes header:
//!
//! * `type` is the node type (leaf or internal).
//! * `nkeys` is the number of keys (and the number of child pointers).
//!
//! Then an array of child pointers and the KV pairs follow. The pointer array
//! is omitted for leaf nodes.
//!
//! Each KV pair is prefixed by its size. For internal nodes,
//! the value size is 0.
//!
//! ```ignore
//! | key_size* | val_size* | key | val** |
//! |    2B     |    2B     | ... | ...   |
//!
//! *  key_size & val_size are omitted for internal nodes.
//! ** val is omitted for internal nodes.
//! ```
//!
//! The encoded KV pairs are concatenated. To find the `n`th KV pair, we have
//! to read all previous pairs. This is avoided by storing the offset of each
//! KV pair.
//!
//! For example, a leaf node `{"k1":"hi", "k3":"hello"}` is encoded as:
//!
//! ```ignore
//! | type | nkeys | offsets |            key-values           | unused |
//! |   2  |   2   | 8 19    | 2 2 "k1" "hi"  2 5 "k3" "hello" |        |
//! |  2B  |  2B   | 2×2B    | 4B + 2B + 2B + 4B + 2B + 5B     |        |
//! ```
//!
//! The offset of the first KV pair is always 0, so it’s not stored. To find the
//! position of the `n`-th pair, use the `offsets[n-1]`. In this example, 8 is
//! the offset of the 2nd pair, 19 is the offset past the end of the 2nd pair.
//!
//! # A range is divided into subranges by keys
//!
//! Keys in an internal node indicate the range of each child.
//! A root node’s range is `[−∞, +∞)`. The range is divided recursively from
//! the root to the leaves. To divide a range into `n` subranges, we need
//! `n − 1` keys. For example, node `["p", "q"]` divides its range `[a, z)`
//! into 3 subranges: `[a, p)`, `[p, q)`, `[q, z)`.
//!
//! However, our format uses `n` keys instead. Each key represents the start of
//! the subrange. For example, node `["p", "q"]` divides its range `[p, z)`
//! into 2 subranges: `[p, q)`, `[q, z)`.
//!
//! This makes the visualization easier and removes some edge cases, but the
//! 1st key in an internal node is redundant, because the range start is
//! inherited from the parent node.
//!
//! # KV size limit
//!
//! We’ll set the node size to 4KB, which is the typical OS page size. However,
//! keys and values can be arbitrarily large, exceeding a single node. There
//! should be a way to store large KVs outside of nodes, or to make the node
//! size variable. This is solvable, but not fundamental. So we’ll just limit
//! the KV size so that they always fit into a node.
//!
//! The key size limit also ensures that an internal node can at least host
//! 2 keys.
//!
//! # Page number
//!
//! An in-memory pointer is an integer address to the location of a byte.
//! For disk data, a pointer can mean a file offset.
//! Either way, it’s just an integer.
//!
//! In a disk-based B+tree, nodes are fixed-size pages, the entire file is an
//! array of fixed-size pages. So a node pointer needs not to address bytes,
//! but the index of pages, called the page number, which is the file offset
//! divided by the page size.
//!
//! # Summary of our B+tree node format
//!
//! * We will implement the database as an array of fixed-size pages.
//! * Each page contains a serialized B+tree node.
//! * A B+tree leaf node is a list of sorted KV pairs.
//! * A B+tree internal node is a list of sorted key-pointer pairs.
//!
//! The node format is just an implementation detail. The B+tree will work as
//! long as nodes contain the necessary information.

mod leaf;
mod internal;

pub use internal::ChildEntry;
pub use internal::Internal;
pub use leaf::Leaf;
use internal::InternalEffect;
use leaf::LeafEffect;
use super::error::NodeError;

type Result<T> = std::result::Result<T, NodeError>;

/// Size of a B+ tree node page.
pub const PAGE_SIZE: usize = 4096;
const MAX_KEY_SIZE: usize = 1000;
const MAX_VALUE_SIZE: usize = 3000;

const _: () = {
    assert!(PAGE_SIZE <= (1 << 16), "page size is within 16 bits");
    assert!(
        (PAGE_SIZE as isize)
            - 2 // type
            - 2 // nkeys
            // 3 keys + overhead
            - 3 * (8 + 2 + MAX_KEY_SIZE as isize)
            >= 0,
        "3 keys of max size cannot fit into an internal node page"
    );
    assert!(
        (PAGE_SIZE as isize)
            - 2 // type
            - 2 // nkeys
            // 1 key-value pair + overhead
            - 1 * (2 + 2 + 2 + MAX_KEY_SIZE as isize + MAX_VALUE_SIZE as isize)
            >= 0,
        "1 key-value pair of max size cannot fit into a leaf node page"
    );
};

/// An enum representing a page's node type.
#[repr(u16)]
enum NodeType {
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

/// An enum representing the type of B+ tree node.
#[derive(Clone, Debug)]
pub enum Node {
    /// A B+ tree leaf node.
    Leaf(Leaf),
    /// A B+ tree internal node.
    Internal(Internal),
}

impl Node {
    /// Gets the key at a specified node index.
    pub fn get_key(&self, i: usize) -> &[u8] {
        match self {
            Node::Leaf(leaf) => leaf.get_key(i),
            Node::Internal(internal) => internal.get_key(i),
        }
    }

    pub fn get_num_keys(&self) -> usize {
        match self {
            Node::Leaf(leaf) => leaf.get_num_keys(),
            Node::Internal(internal) => internal.get_num_keys(),
        }
    }
}

/// An enum representing the effect of a node operation.
#[derive(Debug)]
pub enum NodeEffect {
    /// A node without 0 keys after a delete was performed on it.
    /// This is a special-case of `Underflow` done to avoid unnecessary
    /// page allocations, since empty non-root nodes aren't allowed.
    Empty,
    /// A newly created node that remained  "intact", i.e. it did not split.
    Intact(Node),
    /// The left and right splits of a node that was created.
    ///
    /// The left and right nodes are the same type.
    Split { left: Node, right: Node },
}

impl From<LeafEffect> for NodeEffect {
    fn from(value: LeafEffect) -> Self {
        match value {
            LeafEffect::Empty => NodeEffect::Empty,
            LeafEffect::Intact(leaf) => NodeEffect::Intact(Node::Leaf(leaf)),
            LeafEffect::Split{left, right} => NodeEffect::Split { left: Node::Leaf(left), right: Node::Leaf(right) }
        }
    }
}

impl From<InternalEffect> for NodeEffect {
    fn from(value: InternalEffect) -> Self {
        match value {
            InternalEffect::Empty => NodeEffect::Empty,
            InternalEffect::Intact(internal) => NodeEffect::Intact(Node::Internal(internal)),
            InternalEffect::Split{left, right} => NodeEffect::Split { left: Node::Internal(left), right: Node::Internal(right) }
        }
    }
}

/// An enum representing the sufficiency of a node created or destroyed during a
/// deletion operation on a node.
pub enum Sufficiency {
    /// A node without 0 keys after a delete was performed on it.
    /// This is a special-case of `Underflow` done to avoid unnecessary
    /// page allocations, since empty non-root nodes aren't allowed.
    Empty,
    /// A node that is NOT sufficiently sized but is not empty
    /// (i.e. has 1 key).
    Underflow,
    /// A node that is sufficiently sized (i.e. has at least 2 keys) even after
    /// a delete was performed on it.
    Sufficient,
}

// Returns how sufficient a node is.
pub fn sufficiency(n: &Node) -> Sufficiency {
    match n.get_num_keys() {
        0 => Sufficiency::Empty,
        1 => Sufficiency::Underflow,
        _ => Sufficiency::Sufficient
    }
}

/// Merges `left` and `right` into a possibly-overflowed node and splits if
/// needed. This is modeled as a Deletion b/c it is (so far) only useful in the
/// context of deletion.
pub fn steal_or_merge(left: &Node, right: &Node) -> Result<NodeEffect> {
    match (left, right) {
        (Node::Leaf(left), Node::Leaf(right)) => Ok(Leaf::steal_or_merge(left, right)?.into()),
        (Node::Internal(left), Node::Internal(right)) => Ok(Internal::steal_or_merge(left, right)?.into()),
        _ => unreachable!("It is assumed that both are the same node type."),
    }
}

/// Sets the page header of a node's page buffer.
fn set_page_header(buf: &mut [u8], node_type: NodeType) {
    buf[0..2].copy_from_slice(&(node_type as u16).to_be_bytes());
}

/// Sets the number of keys in a node's page buffer.
fn set_num_keys(buf: &mut [u8], n: usize) {
    buf[2..4].copy_from_slice(&(n as u16).to_be_bytes());
}

/// Gets the number of keys in a node's page buffer.
fn get_num_keys(buf: &[u8]) -> usize {
    u16::from_be_bytes([buf[2], buf[3]]) as usize
}
