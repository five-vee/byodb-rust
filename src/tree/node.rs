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

use std::rc::Rc;

/// Size of a B+ tree node page.
pub const PAGE_SIZE: usize = 4096;

pub type Result<T> = std::result::Result<T, ()>;

/// An enum representing the type of B+ tree node.
#[derive(Debug, Clone)]
pub enum Node {
    /// A B+ tree leaf node.
    Leaf(Leaf),
    /// A B+ tree internal node.
    Internal(Internal),
}

impl Node {
    /// Gets the key at a specified node index.
    pub fn get_key(&self, i: u16) -> &[u8] {
        match self {
            Node::Leaf(leaf) => leaf.get_key(i),
            Node::Internal(internal) => internal.get_key(i),
        }
    }
}

/// An enum representing the node(s) created during an insert or update
/// (aka an "upsert") operation on a tree.
pub enum Upsert {
    /// A newly created node that remained  "intact", i.e. it did not split
    /// after an upsert.
    Intact(Node),
    /// The left and right splits of a node that was created after an upsert.
    Split { left: Node, right: Node },
}

/// An enum representing the node(s), if any, created or destroyed during a
/// deletion operation on a tree.
#[derive(Debug)]
pub enum Deletion {
    /// A node without 0 keys after a delete was performed on it.
    /// This is a special-case of `Underflow` done to avoid unnecessary
    /// page allocations, since empty non-root nodes aren't allowed.
    Empty,
    /// A node that is sufficiently-sized (i.e. has at least 2 keys) even after
    /// a delete was performed on it.
    Sufficient(Node),
    /// A node that was split due to a delete operation. This can happen
    /// because the node had to delete a key and replace it with another key
    /// that was larger, pushing it beyond the page size,
    /// thus triggering a split.
    /// 
    /// Yes, this means the tree can grow in height despite
    /// the deletion operation.
    Split { left: Node, right: Node },
    /// A node that is NOT sufficiently-sized but is not empty
    /// (i.e. has 1 key).
    Underflow(Node),
}

/// Checks if after stealing entries,
/// both `from` and `into` are sufficiently sized.
pub fn can_steal(from: &Node, into: &Node) -> bool {
    unimplemented!()
}

/// Checks if after merging,
/// both `from` and `into` are sufficiently sized.
pub fn can_merge(from: &Node, into: &Node) -> bool {
    unimplemented!()
}

/// Steals entries from `from` and puts into `into`.
pub fn steal(from: &Node, into: &Node) -> Result<(Node, Node)> {
    unimplemented!();
}

/// Merges `from` into `into` into a merged node.
pub fn merge(from: &Node, into: &Node) -> Result<Node> {
    unimplemented!();
}

/// A delta representation of the effects of a deletion.
pub type DeletionDelta = Rc<[(u16, Option<u64>)]>;

/// A child entry to upsert into an internal node.
pub struct ChildEntry {
    /// An optional index into an internal node.
    /// If `None`, the index does not exist yet, meaning the child entry
    /// should be inserted into the internal node.
    /// Otherwise, the child entry should be updated at this index.
    pub maybe_i: Option<u16>,
    /// The key of the child entry.
    pub key: Rc<[u8]>,
    /// The page number (aka child pointer) of the child entry.
    pub page_num: u64,
}

/// A B+ tree leaf node.
#[derive(Debug)]
#[derive(Clone)]
pub struct Leaf {
    buf: Box<[u8]>,
}

/// A builder of a B+ tree leaf node.
struct LeafBuilder {
    buf: Box<[u8]>,
}

impl Default for LeafBuilder {
    fn default() -> Self {
        unimplemented!();
    }
}

impl LeafBuilder {
    /// Creates a leaf builder.
    fn new(len: usize) -> Self {
        unimplemented!();
    }

    /// Adds a key-value pair to the builder.
    fn add_key_value(mut self, key: &[u8], val: &[u8]) -> Self {
        unimplemented!();
    }

    /// Builds a leaf.
    fn build(self) -> Result<Leaf> {
        unimplemented!();
    }
}

impl Leaf {
    /// Create a new leaf node with keys and values.
    pub fn new(keys: &[&[u8]], vals: &[&[u8]]) -> Result<Self> {
        // Error if keys.len() != vals.len().
        // Error if resulting leaf is too large.
        // Build the leaf from keys + vals.
        unimplemented!();
    }

    /// Inserts a key-value pair.
    pub fn insert(&self, key: &[u8], val: &[u8]) -> Result<Upsert> {
        // If the insertion can cause a split, allocate via LeafBuilder::new.
        // Otherwise, allocate via LeafBuilder::default.
        // Build the new leaf from self, key, and val.
        // If overflowed, split into two leaves.
        unimplemented!();
    }

    /// Updates the value corresponding to a key.
    pub fn update(&self, key: &[u8], val: &[u8]) -> Result<Upsert> {
        // If the update can cause a split, allocate via LeafBuilder::new.
        // Otherwise, allocate via LeafBuilder::default.
        // Build the new leaf from self + key + val.
        // If overflowed, split into two leaves.
        unimplemented!();
    }

    /// Deletes a key and its corresponding value.
    pub fn delete(&self, key: &[u8]) -> Result<Deletion> {
        unimplemented!();
    }

    fn get_key(&self, i: u16) -> &[u8] {
        unimplemented!();
    }

    /// Finds the value corresponding to the queried key.
    pub fn find(&self, key: &[u8]) -> Option<&[u8]> {
        unimplemented!();
    }
}

/// A B+ tree internal node.
#[derive(Debug)]
#[derive(Clone)]
pub struct Internal {
    buf: Box<[u8]>,
}

/// A builder of a B+ tree internal node.
struct InternalBuilder {
    buf: Box<[u8]>,
}

impl InternalBuilder {
    /// Creates an internal builder.
    fn new(len: usize) -> Self {
        unimplemented!();
    }

    /// Adds a child entry to the builder.
    fn add_child(mut self, key: &[u8], pointer: u64) -> Self {
        unimplemented!();
    }

    /// Builds an internal node.
    fn build(self) -> Result<Internal> {
        unimplemented!();
    }
}

impl Internal {
    /// Create a new internal node with keys and child pointers.
    pub fn new(keys: &[&[u8]], child_pointers: &[u64]) -> Self {
        // Error if keys.len() != vals.len().
        // Error if resulting leaf is too large.
        // Build the internal from keys + vals.
        unimplemented!();
    }

    /// Inserts or updates child entries.
    pub fn upsert_child_entries(&self, entries: &[ChildEntry]) -> Result<Upsert> {
        // If the connection can cause a split, allocate via InternalBuilder::new.
        // Otherwise, allocate via LeafBuilder::default.
        // Build the internal from self + child_entries.
        // If overflowed, split into two internals.
        unimplemented!();
    }

    /// Finds the index corresponding to the key.
    pub fn find(&self, key: &[u8]) -> Option<u16> {
        unimplemented!();
    }

    /// Gets the child pointer at an index.
    pub fn get_child_pointer(&self, i: u16) -> Result<u64> {
        unimplemented!();
    }

    /// Deletes the child entry (i.e. key and child pointer) at an index.
    pub fn delete_child_entry(&self, i: u16) -> Result<Deletion> {
        // delete child entry @ i
        unimplemented!();
    }

    /// Updates the child entry at an index.
    pub fn update_child_entry(&self, i: u16, key: &[u8], page_num: u64) -> Result<Deletion> {
        // INVARIANT: internal is Sufficient.
        // update child page num @ i
        // If the update can cause a split, allocate via InternalBuilder::new.
        // Otherwise, allocate via InternalBuilder::default.
        // Build the new internal from self + key + val.
        // If overflowed, split into two internal.
        unimplemented!();
    }

    /// Merges a deletion delta into the internal node to simulate a deletion
    /// operation.
    pub fn merge_delta(&self, delta: DeletionDelta) -> Result<Deletion> {
        unimplemented!();
    }

    /// Gets the number of keys.
    pub fn get_num_keys(&self) -> u16 {
        unimplemented!();
    }

    fn get_key(&self, i: u16) -> &[u8] {
        unimplemented!();
    }
}
