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

pub use internal_util::ChildEntry;
use super::error::NodeError;
use internal_util::InternalBuilder;
use leaf_util::LeafBuilder;

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
#[derive(Debug, Clone)]
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

/// An enum representing the node(s) created during an insert or update
/// (aka an "upsert") operation on a tree.
pub enum Upsert {
    /// A newly created node that remained  "intact", i.e. it did not split
    /// after an upsert.
    Intact(Node),
    /// The left and right splits of a node that was created after an upsert.
    ///
    /// The left and right nodes are the same type.
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
    /// A node that is sufficiently sized (i.e. has at least 2 keys) even after
    /// a delete was performed on it.
    Sufficient(Node),
    /// A node that was split due to a delete operation. This can happen
    /// because the node had to delete a key and replace it with another key
    /// that was larger, pushing it beyond the page size,
    /// thus triggering a split.
    ///
    /// Yes, this means the tree can grow in height despite
    /// the deletion operation.
    ///
    /// The left and right nodes are the same type.
    Split { left: Node, right: Node },
    /// A node that is NOT sufficiently sized but is not empty
    /// (i.e. has 1 key).
    Underflow(Node),
}

/// Merges `left` and `right` into a possibly-overflowed node and splits if
/// needed. This is modeled as a Deletion b/c it is (so far) only useful in the
/// context of deletion.
pub fn steal_or_merge(left: &Node, right: &Node) -> Result<Deletion> {
    match (left, right) {
        (Node::Leaf(left), Node::Leaf(right)) => {
            let mut b =
                LeafBuilder::new(left.get_num_keys() + right.get_num_keys()).allow_overflow();
            for (key, val) in left.iter() {
                b = b.add_key_value(key, val)?;
            }
            for (key, val) in right.iter() {
                b = b.add_key_value(key, val)?;
            }
            b.build_deletion()
        }
        (Node::Internal(left), Node::Internal(right)) => {
            let mut b =
                InternalBuilder::new(left.get_num_keys() + right.get_num_keys()).allow_overflow();
            for (key, page_num) in left.iter() {
                b = b.add_child_entry(key, page_num)?;
            }
            for (key, page_num) in right.iter() {
                b = b.add_child_entry(key, page_num)?;
            }
            b.build_deletion()
        }
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

/// Leaf node utilities.
mod leaf_util {
    use super::{Deletion, Leaf, Node, NodeError, NodeType, Result, Upsert};

    /// Gets the `i`th key in a leaf page buffer.
    pub fn get_key(buf: &[u8], i: usize) -> &[u8] {
        let offset = get_offset(buf, i) as usize;
        let num_keys = super::get_num_keys(buf) as usize;
        let key_len = u16::from_be_bytes([
            buf[4 + num_keys * 2 + offset],
            buf[4 + num_keys * 2 + offset + 1],
        ]) as usize;
        &buf[4 + num_keys * 2 + offset + 4..4 + num_keys * 2 + offset + 4 + key_len]
    }

    /// Gets the `i`th value in a leaf page buffer.
    pub fn get_value(buf: &[u8], i: usize) -> &[u8] {
        let offset = get_offset(buf, i) as usize;
        let num_keys = super::get_num_keys(buf) as usize;
        let key_len = u16::from_be_bytes([
            buf[4 + num_keys * 2 + offset],
            buf[4 + num_keys * 2 + offset + 1],
        ]) as usize;
        let val_len = u16::from_be_bytes([
            buf[4 + num_keys * 2 + offset + 2],
            buf[4 + num_keys * 2 + offset + 3],
        ]) as usize;
        &buf[4 + num_keys * 2 + offset + 4 + key_len
            ..4 + num_keys * 2 + offset + 4 + key_len + val_len]
    }

    /// Gets the `i`th offset value.
    fn get_offset(buf: &[u8], i: usize) -> usize {
        if i == 0 {
            return 0;
        }
        u16::from_be_bytes([buf[4 + 2 * (i - 1)], buf[4 + 2 * i]]) as usize
    }

    /// Gets the number of bytes consumed by a page.
    pub fn get_num_bytes(buf: &[u8]) -> usize {
        let n = super::get_num_keys(buf);
        let offset = get_offset(buf, n);
        4 + (n * 2) + offset
    }

    /// Sets the next (i.e. `i+1`th) offset and returns the current offset.
    fn set_next_offset(buf: &mut [u8], i: usize, key: &[u8], val: &[u8]) -> usize {
        let curr_offset = get_offset(buf, i);
        let next_offset = curr_offset + 4 + key.len() + val.len();
        let next_i = i as usize + 1;
        buf[4 + 2 * (next_i - 1)..4 + 2 * next_i]
            .copy_from_slice(&(next_offset as u16).to_be_bytes());
        curr_offset
    }

    // A builder of a B+ tree leaf node.
    pub struct LeafBuilder {
        num_keys: usize,
        i: usize,
        buf: Option<Box<[u8]>>,
    }

    impl LeafBuilder {
        /// Creates a new leaf builder.
        pub fn new(num_keys: usize) -> Self {
            Self {
                num_keys,
                i: 0,
                buf: None,
            }
        }

        /// Allows the builder to overflow to two pages.
        pub fn allow_overflow(mut self) -> Self {
            assert!(
                self.buf.is_none(),
                "allow_overflow() must be called only once and before add_key_value()"
            );
            self.buf = Some([0; 2 * super::PAGE_SIZE - 4].into());
            self
        }

        /// Adds a key-value pair to the builder.
        pub fn add_key_value(mut self, key: &[u8], val: &[u8]) -> Result<Self> {
            assert!(
                self.i < self.num_keys,
                "add_key_value() called {} times, cannot be called more times than num_keys = {}",
                self.i,
                self.num_keys
            );
            if key.len() > super::MAX_KEY_SIZE {
                return Err(NodeError::MaxKeySize(key.len()));
            }
            if val.len() > super::MAX_VALUE_SIZE {
                return Err(NodeError::MaxValueSize(val.len()));
            }

            // Make sure buffer is initialized.
            if self.buf.is_none() {
                self.buf = Some(Self::new_buffer());
            }
            let mut buf = self.buf.take().unwrap();

            let n = self.num_keys;
            let offset = set_next_offset(&mut buf, self.i, key, val);
            let simulated_bytes = 4 + self.i * 2 + offset;
            assert!(simulated_bytes + 4 + key.len() + val.len() <= buf.len(),
            "builder unexpectedly overflowed; please call allow_overflow(), or don't add too many key-value pairs.");

            let pos = 4 + n * 2 + offset;
            buf[pos..pos + 2].copy_from_slice(&(key.len() as u16).to_be_bytes());
            buf[pos + 2..pos + 4].copy_from_slice(&(val.len() as u16).to_be_bytes());
            buf[pos + 4..pos + 4 + key.len()].copy_from_slice(key);
            buf[pos + 4 + key.len()..pos + 4 + key.len() + val.len()].copy_from_slice(val);

            self.i += 1;
            super::set_num_keys(&mut buf, self.i);
            self.buf = Some(buf);
            Ok(self)
        }

        /// Builds an Upsert.
        pub fn build_upsert(self) -> Result<Upsert> {
            assert!(
                self.i != self.num_keys,
                "build_upsert() must be called after calling add_key_value() num_keys = {} times",
                self.num_keys
            );
            Ok(Self::new_upsert(self.build_single_or_split()?))
        }

        /// Builds a Deletion.
        pub fn build_deletion(self) -> Result<Deletion> {
            assert!(
                self.i != self.num_keys,
                "build_deletion() must be called after calling add_key_value() num_keys = {} times",
                self.num_keys
            );
            Ok(Self::new_deletion(self.build_single_or_split()?))
        }

        /// Creates a new page-sized in-memory buffer.
        fn new_buffer() -> Box<[u8]> {
            let mut buf = [0; super::PAGE_SIZE];
            super::set_page_header(&mut buf, NodeType::Leaf);
            buf.into()
        }

        /// Creates a new Upsert from at least 1 leaf.
        fn new_upsert(build_result: (Leaf, Option<Leaf>)) -> Upsert {
            match build_result {
                (left, Some(right)) => Upsert::Split {
                    left: Node::Leaf(left),
                    right: Node::Leaf(right),
                },
                (leaf, None) => Upsert::Intact(Node::Leaf(leaf)),
            }
        }

        /// Creates a new Deletion from at least 1 leaf.
        fn new_deletion(build_result: (Leaf, Option<Leaf>)) -> Deletion {
            match build_result {
                (left, Some(right)) => Deletion::Split {
                    left: Node::Leaf(left),
                    right: Node::Leaf(right),
                },
                (leaf, None) => {
                    let n = super::get_num_keys(&leaf.buf);
                    if n == 0 {
                        Deletion::Empty
                    } else if n == 1 {
                        Deletion::Underflow(Node::Leaf(leaf))
                    } else {
                        Deletion::Sufficient(Node::Leaf(leaf))
                    }
                }
            }
        }

        /// Builds one leaf, or two due to splitting.
        fn build_single_or_split(mut self) -> Result<(Leaf, Option<Leaf>)> {
            if self.buf.is_none() {
                // Technically, an empty leaf is allowed.
                return Ok((Leaf::default(), None));
            }
            let buf = self.buf.take().unwrap();
            if get_num_bytes(&buf) <= super::PAGE_SIZE {
                return Ok((self.build_single(), None));
            }
            self.buf = Some(buf);
            let (left, right) = self.build_split()?;
            Ok((left, Some(right)))
        }

        /// Builds two splits of a leaf.
        fn build_split(mut self) -> Result<(Leaf, Leaf)> {
            let buf = self.buf.take().unwrap();
            let num_keys = self.num_keys as usize;
            let mut left_end: usize = 0;
            for i in 0..num_keys {
                // include i?
                let offset = get_offset(&buf, i + 1);
                if 4 + (i + 1) * 2 + offset <= super::PAGE_SIZE {
                    left_end = i + 1;
                }
            }

            let mut lb = Self::new(left_end);
            for i in 0..left_end {
                lb = lb.add_key_value(get_key(&buf, i), get_value(&buf, i))?;
            }
            let mut rb = Self::new(num_keys - left_end);
            for i in left_end..num_keys {
                rb = rb.add_key_value(get_key(&buf, i), get_value(&buf, i))?;
            }
            Ok((lb.build_single(), rb.build_single()))
        }

        /// Builds a leaf.
        fn build_single(mut self) -> Leaf {
            let buf = self.buf.take().unwrap();
            assert!(get_num_bytes(&buf) <= super::PAGE_SIZE);
            Leaf {
                buf: buf[0..super::PAGE_SIZE].into(),
            }
        }
    }
}

/// A B+ tree leaf node.
#[derive(Debug, Clone)]
pub struct Leaf {
    buf: Box<[u8]>,
}

impl Default for Leaf {
    fn default() -> Self {
        let mut buf = Box::new([0; PAGE_SIZE]);
        set_page_header(buf.as_mut(), NodeType::Leaf);
        return Self { buf };
    }
}

impl Leaf {
    /// Inserts a key-value pair.
    pub fn insert(&self, key: &[u8], val: &[u8]) -> Result<Upsert> {
        if self.find(key).is_some() {
            return Err(NodeError::AlreadyExists);
        }
        let mut b = LeafBuilder::new(self.get_num_keys() + 1).allow_overflow();
        let mut added = false;
        for (k, v) in self.iter() {
            if !added && key < k {
                added = true;
                b = b.add_key_value(key, val)?;
            }
            b = b.add_key_value(k, v)?;
        }
        b.build_upsert()
    }

    /// Updates the value corresponding to a key.
    pub fn update(&self, key: &[u8], val: &[u8]) -> Result<Upsert> {
        if self.find(key).is_none() {
            return Err(NodeError::KeyNotFound);
        }
        let mut b = LeafBuilder::new(self.get_num_keys()).allow_overflow();
        let mut added = false;
        for (k, v) in self.iter() {
            if !added && key == k {
                added = true;
                b = b.add_key_value(key, val)?;
                continue;
            }
            b = b.add_key_value(k, v)?;
        }
        b.build_upsert()
    }

    /// Deletes a key and its corresponding value.
    pub fn delete(&self, key: &[u8]) -> Result<Deletion> {
        if self.find(key).is_none() {
            return Err(NodeError::KeyNotFound);
        }
        // Optimization: avoid memory allocation and
        // just return Deletion::Empty if only 1 key.
        let n = self.get_num_keys();
        if n == 1 {
            return Ok(Deletion::Empty);
        }
        let mut b = LeafBuilder::new(n - 1).allow_overflow();
        let mut added = false;
        for (k, v) in self.iter() {
            if !added && key == k {
                added = true;
                continue;
            }
            b = b.add_key_value(k, v)?;
        }
        b.build_deletion()
    }

    /// Finds the value corresponding to the queried key.
    pub fn find(&self, key: &[u8]) -> Option<&[u8]> {
        self.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    pub fn iter<'a>(&'a self) -> LeafIterator<'a> {
        LeafIterator {
            node: self,
            i: 0,
            n: self.get_num_keys(),
        }
    }

    fn get_key(&self, i: usize) -> &[u8] {
        leaf_util::get_key(&self.buf, i)
    }

    fn get_value(&self, i: usize) -> &[u8] {
        leaf_util::get_value(&self.buf, i)
    }

    fn get_num_keys(&self) -> usize {
        get_num_keys(&self.buf)
    }
}

/// An key-value iterator for a leaf node.
pub struct LeafIterator<'a> {
    node: &'a Leaf,
    i: usize,
    n: usize,
}

impl<'a> Iterator for LeafIterator<'a> {
    type Item = (&'a [u8], &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.n {
            return None;
        }
        let item = Some((self.node.get_key(self.i), self.node.get_value(self.i)));
        self.i += 1;
        item
    }
}

/// Internal node utilities.
mod internal_util {
    use std::rc::Rc;

    use super::super::error::NodeError;
    use super::{Deletion, Internal, Node, NodeType, Result, Upsert};

    /// A child entry to insert, update, or delete in an internal node.
    pub enum ChildEntry {
        Insert {
            key: Rc<[u8]>,
            page_num: u64,
        },
        Update {
            i: usize,
            key: Rc<[u8]>,
            page_num: u64,
        },
        Delete {
            i: usize,
        },
    }

    /// Gets the `i`th key in an internal node's page buffer.
    pub fn get_key(buf: &[u8], i: usize) -> &[u8] {
        let offset = get_offset(buf, i) as usize;
        let num_keys = super::get_num_keys(buf) as usize;
        let key_len = u16::from_be_bytes([
            buf[4 + num_keys * 10 + offset],
            buf[4 + num_keys * 10 + offset + 1],
        ]) as usize;
        &buf[4 + num_keys * 10 + offset..4 + num_keys * 10 + offset + key_len]
    }

    /// Gets the `i`th child pointer in an internal node's page buffer.
    pub fn get_child_pointer(buf: &[u8], i: usize) -> u64 {
        u64::from_be_bytes([
            buf[4 + i * 8],
            buf[4 + i * 8 + 1],
            buf[4 + i * 8 + 2],
            buf[4 + i * 8 + 3],
            buf[4 + i * 8 + 4],
            buf[4 + i * 8 + 5],
            buf[4 + i * 8 + 6],
            buf[4 + i * 8 + 7],
        ])
    }

    /// Sets the `i`th child pointer in an internal node's page buffer.
    fn set_child_pointer(buf: &mut [u8], i: usize, page_num: u64) {
        buf[4 + i * 8..4 + (i + 1) * 8].copy_from_slice(&page_num.to_be_bytes());
    }

    /// Gets the `i`th offset value.
    fn get_offset(buf: &[u8], i: usize) -> usize {
        if i == 0 {
            return 0;
        }
        let n = super::get_num_keys(buf);
        u16::from_be_bytes([buf[4 + n * 8 + 2 * (i - 1)], buf[4 + n * 8 + 2 * i]]) as usize
    }

    /// Sets the next (i.e. `i+1`th) offset and returns the current offset.
    fn set_next_offset(buf: &mut [u8], i: usize, n: usize, key: &[u8]) -> usize {
        let curr_offset = get_offset(buf, i);
        let next_offset = curr_offset + key.len();
        let next_i = i as usize + 1;
        buf[4 + n * 8 + 2 * (next_i - 1)..4 + n * 8 + 2 * next_i]
            .copy_from_slice(&(next_offset as u16).to_be_bytes());
        curr_offset
    }

    /// Gets the number of bytes consumed by a page.
    pub fn get_num_bytes(buf: &[u8]) -> usize {
        let n = super::get_num_keys(buf);
        let offset = get_offset(buf, n);
        4 + (n * 10) + offset
    }

    /// A builder of a B+ tree internal node.
    pub struct InternalBuilder {
        num_keys: usize,
        i: usize,
        buf: Option<Box<[u8]>>,
    }

    impl InternalBuilder {
        /// Creates a new internal builder.
        pub fn new(num_keys: usize) -> Self {
            assert!(num_keys >= 2, "An internal node must have at least 2 keys.");
            Self {
                num_keys,
                i: 0,
                buf: None,
            }
        }

        /// Allows the builder to overflow to two pages.
        pub fn allow_overflow(mut self) -> Self {
            assert!(
                self.buf.is_none(),
                "allow_overflow() must be called only once and before add_child_entry()"
            );
            self.buf = Some([0; 2 * super::PAGE_SIZE - 4].into());
            self
        }

        /// Adds a child entry to the builder.
        pub fn add_child_entry(mut self, key: &[u8], page_num: u64) -> Result<Self> {
            assert!(
                self.i < self.num_keys,
                "add_child_entry() called {} times, cannot be called more times than num_keys = {}",
                self.i,
                self.num_keys
            );
            if key.len() > super::MAX_KEY_SIZE {
                return Err(NodeError::MaxKeySize(key.len()));
            }

            // Make sure buffer is initialized.
            if self.buf.is_none() {
                self.buf = Some(Self::new_buffer());
            }
            let mut buf = self.buf.take().unwrap();

            let n = self.num_keys;
            let offset = set_next_offset(&mut buf, self.i, n, key);
            set_child_pointer(&mut buf, self.i, page_num);
            let simulated_bytes = 4 + self.i * 10 + offset;
            assert!(
                simulated_bytes + key.len() <= buf.len(),
                "builder unexpectedly overflowed; please call allow_overflow(), or don't add too many key-value pairs.");

            let pos = 4 + n * 10 + offset;
            buf[pos..pos + key.len()].copy_from_slice(key);

            self.i += 1;
            super::set_num_keys(&mut buf, self.i);
            self.buf = Some(buf);
            Ok(self)
        }

        /// Builds an Upsert.
        pub fn build_upsert(self) -> Result<Upsert> {
            assert!(
                self.i == self.num_keys,
                "build_upsert() must be called after calling add_child_entry() num_keys = {} times",
                self.num_keys
            );
            Ok(Self::new_upsert(self.build_single_or_split()?))
        }

        /// Builds a Deletion.
        pub fn build_deletion(self) -> Result<Deletion> {
            assert!(
                self.i == self.num_keys,
                "build_deletion() must be called after calling add_child_entry() num_keys = {} times",
                self.num_keys
            );
            Ok(Self::new_deletion(self.build_single_or_split()?))
        }

        /// Builds an internal node.
        pub fn build_single(mut self) -> Internal {
            assert!(
                self.i == self.num_keys,
                "build_single() must be called after calling add_child_entry() num_keys = {} times",
                self.num_keys
            );
            let buf = self.buf.take().unwrap();
            assert!(get_num_bytes(&buf) <= super::PAGE_SIZE);
            Internal {
                buf: buf[0..super::PAGE_SIZE].into(),
            }
        }

        /// Builds one internal node, or two due to splitting.
        pub fn build_single_or_split(mut self) -> Result<(Internal, Option<Internal>)> {
            assert!(
                self.i == self.num_keys,
                "build_single_or_split() must be called after calling add_child_entry() num_keys = {} times",
                self.num_keys
            );
            let buf = self.buf.take().unwrap();
            if get_num_bytes(&buf) <= super::PAGE_SIZE {
                return Ok((self.build_single(), None));
            }
            let (left, right) = self.build_split()?;
            Ok((left, Some(right)))
        }

        fn new_buffer() -> Box<[u8]> {
            let mut buf = [0; super::PAGE_SIZE];
            super::set_page_header(&mut buf, NodeType::Internal);
            buf.into()
        }

        /// Creates a new Upsert from at least 1 internal node.
        fn new_upsert(build_result: (Internal, Option<Internal>)) -> Upsert {
            match build_result {
                (left, Some(right)) => Upsert::Split {
                    left: Node::Internal(left),
                    right: Node::Internal(right),
                },
                (internal, None) => Upsert::Intact(Node::Internal(internal)),
            }
        }

        /// Creates a new Deletion from at least 1 leaf.
        fn new_deletion(build_result: (Internal, Option<Internal>)) -> Deletion {
            match build_result {
                (left, Some(right)) => Deletion::Split {
                    left: Node::Internal(left),
                    right: Node::Internal(right),
                },
                (internal, None) => {
                    let n = super::get_num_keys(&internal.buf);
                    if n == 0 {
                        Deletion::Empty
                    } else if n == 1 {
                        Deletion::Underflow(Node::Internal(internal))
                    } else {
                        Deletion::Sufficient(Node::Internal(internal))
                    }
                }
            }
        }

        /// Builds two splits of an internal node.
        fn build_split(mut self) -> Result<(Internal, Internal)> {
            let buf = self.buf.take().unwrap();
            let num_keys = self.num_keys as usize;
            let mut left_end: usize = 0;
            for i in 0..num_keys {
                // include i?
                let offset = get_offset(&buf, i + 1);
                if 4 + (i + 1) * 2 + offset <= super::PAGE_SIZE {
                    left_end = i + 1;
                }
            }

            let mut lb = Self::new(left_end);
            for i in 0..left_end {
                lb = lb.add_child_entry(get_key(&buf, i), get_child_pointer(&buf, i))?;
            }
            let mut rb = Self::new(num_keys - left_end);
            for i in left_end..num_keys {
                rb = rb.add_child_entry(get_key(&buf, i), get_child_pointer(&buf, i))?;
            }
            Ok((lb.build_single(), rb.build_single()))
        }
    }
}

/// A B+ tree internal node.
#[derive(Debug, Clone)]
pub struct Internal {
    buf: Box<[u8]>,
}

impl Internal {
    /// Creates an internal node that is the parent of two splits.
    pub fn parent_of_split(keys: [&[u8]; 2], child_pointers: [u64; 2]) -> Result<Self> {
        let parent = InternalBuilder::new(2)
            .add_child_entry(keys[0], child_pointers[0])?
            .add_child_entry(keys[1], child_pointers[1])?
            .build_single();
        Ok(parent)
    }

    /// Inserts or updates child entries.
    pub fn merge_as_upsert(&self, entries: &[ChildEntry]) -> Result<Upsert> {
        let b = self.merge_child_entries(entries)?;
        b.build_upsert()
    }

    /// Updates or deletes child entries.
    pub fn merge_as_deletion(&self, entries: &[ChildEntry]) -> Result<Deletion> {
        let b = self.merge_child_entries(entries)?;
        b.build_deletion()
    }

    /// Finds the index of the child that contains the key.
    pub fn find(&self, key: &[u8]) -> Option<usize> {
        (self.get_num_keys() - 1..=0).find(|i| self.get_key(*i) <= key)
    }

    /// Gets the child pointer at an index.
    pub fn get_child_pointer(&self, i: usize) -> u64 {
        internal_util::get_child_pointer(&self.buf, i)
    }

    /// Gets the number of keys.
    pub fn get_num_keys(&self) -> usize {
        get_num_keys(&self.buf)
    }

    /// Creates an key-value iterator for the internal node.
    pub fn iter<'a>(&'a self) -> InternalIterator<'a> {
        InternalIterator {
            node: self,
            i: 0,
            n: self.get_num_keys(),
        }
    }

    /// Merges child entries into the internal node.
    /// Returns a builder so the user can decide on building
    /// an `Upsert` or a `Deletion`.
    fn merge_child_entries(&self, entries: &[ChildEntry]) -> Result<InternalBuilder> {
        let extra = entries
            .iter()
            .filter(|ce| {
                if let ChildEntry::Insert {
                    key: _,
                    page_num: _,
                } = *ce
                {
                    true
                } else {
                    false
                }
            })
            .count();
        let mut b = InternalBuilder::new(self.get_num_keys() + extra);
        if extra > 0 {
            b = b.allow_overflow();
        }
        let mut entries_iter = entries.iter();
        let mut next = entries_iter.next();
        for i in 0..self.get_num_keys() {
            if next.is_none() {
                b = b.add_child_entry(self.get_key(i), self.get_child_pointer(i))?;
                continue;
            }
            match next.unwrap() {
                ChildEntry::Insert { key, page_num } => {
                    b = b.add_child_entry(key, *page_num)?;
                    b = b.add_child_entry(self.get_key(i), self.get_child_pointer(i))?;
                    next = entries_iter.next();
                }
                ChildEntry::Update {
                    i: j,
                    key,
                    page_num,
                } if i == *j => {
                    b = b.add_child_entry(key, *page_num)?;
                    next = entries_iter.next();
                }
                ChildEntry::Delete { i: j } if i == *j => {
                    next = entries_iter.next();
                }
                _ => {
                    b = b.add_child_entry(self.get_key(i), self.get_child_pointer(i))?;
                }
            }
        }
        if let Some(ChildEntry::Insert { key, page_num }) = next {
            b = b.add_child_entry(key, *page_num)?;
        }
        Ok(b)
    }

    /// Gets the `i`th key in the internal buffer.
    fn get_key(&self, i: usize) -> &[u8] {
        internal_util::get_key(&self.buf, i)
    }
}

/// A key-value iterator of an internal node.
pub struct InternalIterator<'a> {
    node: &'a Internal,
    i: usize,
    n: usize,
}

impl<'a> Iterator for InternalIterator<'a> {
    type Item = (&'a [u8], u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.n {
            return None;
        }
        let item = Some((
            self.node.get_key(self.i),
            self.node.get_child_pointer(self.i),
        ));
        self.i += 1;
        item
    }
}
