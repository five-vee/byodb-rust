//! [`Node`] is a enum that wraps either a [`Leaf`] or [`Internal`] node.
//!
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

mod internal;
mod leaf;

use std::rc::Rc;

use crate::core::error::NodeError;
use crate::core::mmap::{Guard, Page, ReadOnlyPage, Writer};
use crate::core::{consts, header};
use header::NodeType;
pub(crate) use internal::ChildEntry;
pub(crate) use internal::Internal;
use internal::InternalEffect;
pub(crate) use leaf::Leaf;
use leaf::LeafEffect;

#[allow(dead_code)]
type Result<T> = std::result::Result<T, NodeError>;

/// An enum representing the type of B+ tree node.
pub enum Node<'g, P: ReadOnlyPage<'g>> {
    /// A B+ tree leaf node.
    Leaf(Leaf<'g, P>),
    /// A B+ tree internal node.
    Internal(Internal<'g, P>),
}

impl<'g, P: ReadOnlyPage<'g>> Node<'g, P> {
    /// Reads the page at `page_num` and returns it represented as a `Node`.
    /// This is unsafe for the same reason [`Guard::read_page`] is unsafe.
    pub unsafe fn read<G: Guard<'g, P>>(guard: &'g G, page_num: usize) -> Node<'g, P> {
        let page = unsafe { guard.read_page(page_num) };
        match header::get_node_type(&page).unwrap() {
            NodeType::Leaf => Node::Leaf(Leaf::read(guard, page_num)),
            NodeType::Internal => Node::Internal(Internal::read(guard, page_num)),
        }
    }

    pub fn page_num(&self) -> usize {
        match self {
            Node::Leaf(leaf) => leaf.page_num(),
            Node::Internal(internal) => internal.page_num(),
        }
    }

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

    pub fn get_num_bytes(&self) -> usize {
        match self {
            Node::Leaf(leaf) => leaf.get_num_bytes(),
            Node::Internal(internal) => internal.get_num_bytes(),
        }
    }
}

/// An enum representing the effect of a node operation.
pub enum NodeEffect<'g> {
    /// A node without 0 keys after a delete was performed on it.
    /// This is a special-case of `Underflow` done to avoid unnecessary
    /// page allocations, since empty non-root nodes aren't allowed.
    Empty,
    /// A newly created node that remained  "intact", i.e. it did not split.
    Intact(Node<'g, Page<'g>>),
    /// The left and right splits of a node that was created.
    ///
    /// The left and right nodes are the same type.
    Split {
        left: Node<'g, Page<'g>>,
        right: Node<'g, Page<'g>>,
    },
}

impl NodeEffect<'_> {
    /// Converts the node(s) created during an operation into child
    /// entries of a B+ tree internal node.
    ///
    /// `i` is the index in the internal that the operation was performed on.
    pub fn child_entries(self, i: usize) -> Rc<[ChildEntry]> {
        match self {
            NodeEffect::Empty => Rc::new([]),
            NodeEffect::Intact(node) => [ChildEntry::Update {
                i,
                key: node.get_key(0).into(),
                page_num: node.page_num(),
            }]
            .into(),
            NodeEffect::Split { left, right } => [
                ChildEntry::Update {
                    i,
                    key: left.get_key(0).into(),
                    page_num: left.page_num(),
                },
                ChildEntry::Insert {
                    key: right.get_key(0).into(),
                    page_num: right.page_num(),
                },
            ]
            .into(),
        }
    }
}

impl<'a> From<LeafEffect<'a>> for NodeEffect<'a> {
    fn from(value: LeafEffect<'a>) -> Self {
        match value {
            LeafEffect::Empty => NodeEffect::Empty,
            LeafEffect::Intact(leaf) => NodeEffect::Intact(Node::Leaf(leaf)),
            LeafEffect::Split { left, right } => NodeEffect::Split {
                left: Node::Leaf(left),
                right: Node::Leaf(right),
            },
        }
    }
}

impl<'a> From<InternalEffect<'a>> for NodeEffect<'a> {
    fn from(value: InternalEffect<'a>) -> Self {
        match value {
            InternalEffect::Intact(internal) => NodeEffect::Intact(Node::Internal(internal)),
            InternalEffect::Split { left, right } => NodeEffect::Split {
                left: Node::Internal(left),
                right: Node::Internal(right),
            },
        }
    }
}

/// An enum representing the sufficiency of a node created or destroyed during a
/// deletion operation on a node.
pub enum Sufficiency {
    /// A node without 0 keys after a delete was performed on it.
    /// This is a special-case of [`Sufficiency::Underflow`] done to avoid
    /// unnecessary page allocations, since empty non-root nodes aren't
    /// allowed.
    Empty,
    /// A node that is NOT sufficiently sized but is not empty
    /// (i.e. has 1 key).
    Underflow,
    /// A node that is sufficiently sized (i.e. has at least 2 keys) even after
    /// a delete was performed on it.
    Sufficient,
}

// Returns how sufficient a node is.
pub fn sufficiency<'w>(n: &Node<'w, Page<'w>>) -> Sufficiency {
    match n.get_num_keys() {
        0 => Sufficiency::Empty,
        1 => Sufficiency::Underflow,
        _ => Sufficiency::Sufficient,
    }
}

/// Merges `left` and `right` into a possibly-overflowed node and splits if
/// needed. This is modeled as a Deletion b/c it is (so far) only useful in the
/// context of deletion.
pub fn steal_or_merge<'w>(
    left: Node<'w, Page<'w>>,
    right: Node<'w, Page<'w>>,
    writer: &'w Writer,
) -> NodeEffect<'w> {
    match (left, right) {
        (Node::Leaf(left), Node::Leaf(right)) => Leaf::steal_or_merge(left, right, writer).into(),
        (Node::Internal(left), Node::Internal(right)) => {
            Internal::steal_or_merge(left, right, writer).into()
        }
        _ => unreachable!("It is assumed that both are the same node type."),
    }
}

/// Checks whether `to` can steal a key from `from`.
/// `steal_end` determines whether to steal the end key of `from`,
/// otherwise the beginning key.
pub fn can_steal<'w>(from: &Node<'w, Page<'w>>, to: &Node<'w, Page<'w>>, steal_end: bool) -> bool {
    if from.get_num_keys() <= 2 {
        return false;
    }
    let i = if steal_end {
        from.get_num_keys() - 1
    } else {
        0
    };
    match (from, to) {
        (Node::Leaf(from), Node::Leaf(to)) => {
            2 + 4 + from.get_key(i).len() + from.get_value(i).len() + to.get_num_bytes()
                <= consts::PAGE_SIZE
        }
        (Node::Internal(from), Node::Internal(to)) => {
            8 + 2 + from.get_key(i).len() + to.get_num_bytes() <= consts::PAGE_SIZE
        }
        _ => unreachable!(),
    }
}

/// Checks whether the merging of `left` and `right` doesn't overflow.
pub fn can_merge<'w>(left: &Node<'w, Page<'w>>, right: &Node<'w, Page<'w>>) -> bool {
    left.get_num_bytes() + right.get_num_bytes() - 4 < consts::PAGE_SIZE
}
