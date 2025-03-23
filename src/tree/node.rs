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

use leaf_util::LeafBuilder;
pub use internal_util::{ChildEntry, DeletionDelta};

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
        "3 keys of max size cannot fit into a page"
    );
    assert!(
        (PAGE_SIZE as isize)
            - 2 // type
            - 2 // nkeys
            // 1 key-value pair + overhead
            - 1 * (2 + 2 + 2 + MAX_KEY_SIZE as isize + MAX_VALUE_SIZE as isize)
            >= 0,
        "1 key-value pair of max size cannot fit into a page"
    );
};

#[repr(u16)]
enum NodeType {
    Leaf = 0b01u16,
    Internal = 0b10u16,
}

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
    pub fn get_key(&self, i: usize) -> &[u8] {
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
    Split { left: Node, right: Node },
    /// A node that is NOT sufficiently sized but is not empty
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

fn set_page_header(buf: &mut [u8], node_type: NodeType) {
    buf[0..2].copy_from_slice(&(node_type as u16).to_be_bytes());
}

fn set_num_keys(buf: &mut [u8], n: usize) {
    buf[2..4].copy_from_slice(&(n as u16).to_be_bytes());
}

fn get_num_keys(buf: &[u8]) -> usize {
    u16::from_be_bytes([buf[2], buf[3]]) as usize
}

mod leaf_util {
    use super::{Deletion, Leaf, Node, NodeType, Result, Upsert};

    /// Gets the `i`th offset value.
    pub fn get_offset(buf: &[u8], i: usize) -> usize {
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

    pub fn get_key(buf: &[u8], i: usize) -> &[u8] {
        let offset = get_offset(buf, i) as usize;
        let num_keys = super::get_num_keys(buf) as usize;
        let key_len = u16::from_be_bytes([
            buf[4 + num_keys * 2 + offset],
            buf[4 + num_keys * 2 + offset + 1],
        ]) as usize;
        &buf[4 + num_keys * 2 + offset + 4..4 + num_keys * 2 + offset + 4 + key_len]
    }

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

    /// Sets the next (i.e. `i+1`th) offset and returns the current offset.
    fn set_next_offset(buf: &mut [u8], i: usize, key: &[u8], val: &[u8]) -> usize {
        let curr_offset = get_offset(buf, i);
        let next_offset = curr_offset + 4 + (key.len() + val.len());
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

        fn new_buffer() -> Box<[u8]> {
            let mut buf = [0; super::PAGE_SIZE];
            super::set_page_header(&mut buf, NodeType::Leaf);
            buf.into()
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
            if self.i >= self.num_keys {
                return Err(());
            }
            if key.len() > super::MAX_KEY_SIZE {
                return Err(());
            }
            if val.len() > super::MAX_VALUE_SIZE {
                return Err(());
            }

            // Make sure buffer is initialized.
            if self.buf.is_none() {
                self.buf = Some(Self::new_buffer());
            }
            let mut buf = self.buf.take().unwrap();

            assert!(get_num_bytes(&buf) + 6 + key.len() + val.len() <= buf.len(),
                "builder unexpectedly overflowed; please call allow_overflow(), or don't add too many key-value pairs.");

            let offset = set_next_offset(&mut buf, self.i, key, val);
            let n = self.num_keys;
            buf[4 + n * 2 + offset..4 + n * 2 + offset + 2]
                .copy_from_slice(&(key.len() as u16).to_be_bytes());
            buf[4 + n * 2 + offset + 2..4 + n * 2 + offset + 4]
                .copy_from_slice(&(val.len() as u16).to_be_bytes());
            buf[4 + n * 2 + offset + 4..4 + n * 2 + offset + 4 + key.len()].copy_from_slice(key);
            buf[4 + n * 2 + offset + 4 + key.len()..4 + n * 2 + offset + 4 + key.len() + val.len()]
                .copy_from_slice(val);

            self.i += 1;
            super::set_num_keys(&mut buf, self.i);
            self.buf = Some(buf);
            Ok(self)
        }

        /// Builds an Upsert.
        pub fn build_upsert(mut self) -> Result<Upsert> {
            Ok(Self::new_upsert(self.build_single_or_split()?))
        }

        /// Builds a Deletion.
        pub fn build_deletion(mut self) -> Result<Deletion> {
            Ok(Self::new_deletion(self.build_single_or_split()?))
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

        /// Builds one leaf, or two leaves due to splitting.
        fn build_single_or_split(mut self) -> Result<(Leaf, Option<Leaf>)> {
            if self.i != self.num_keys {
                return Err(());
            }
            if self.buf.is_none() {
                return Err(());
            }
            let buf = self.buf.take().unwrap();
            if get_num_bytes(&buf) <= super::PAGE_SIZE {
                return Ok((self.build_single(), None));
            }
            let (left, right) = self.build_split()?;
            Ok((left, Some(right)))
        }

        /// Builds two splits of a leaf.
        fn build_split(mut self) -> Result<(Leaf, Leaf)> {
            if self.buf.is_none() {
                return Err(());
            }
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

        /// Builds a leaf. Errors if the builder built more than one leaf due
        /// to splitting.
        fn build_single(mut self) -> Leaf {
            assert!(self.buf.is_some());
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
        return Self {
            buf: Box::new([0; PAGE_SIZE]),
        };
    }
}

impl Leaf {
    /// Inserts a key-value pair.
    pub fn insert(&self, key: &[u8], val: &[u8]) -> Result<Upsert> {
        if self.find(key).is_some() {
            return Err(());
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
            return Err(());
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
            return Err(());
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
        unimplemented!();
    }

    pub fn iter<'a>(&'a self) -> LeafIterator<'a> {
        LeafIterator {
            leaf: self,
            i: 0,
            n: get_num_keys(&self.buf),
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

pub struct LeafIterator<'a> {
    leaf: &'a Leaf,
    i: usize,
    n: usize,
}

impl<'a> Iterator for LeafIterator<'a> {
    type Item = (&'a [u8], &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.n {
            return None;
        }
        let item = Some((self.leaf.get_key(self.i), self.leaf.get_value(self.i)));
        self.i += 1;
        item
    }
}

mod internal_util {
    use std::rc::Rc;

    /// A delta representation of the effects of a deletion.
    pub type DeletionDelta = Rc<[(usize, Option<u64>)]>;

    /// A child entry to upsert into an internal node.
    pub struct ChildEntry {
        /// An optional index into an internal node.
        /// If `None`, the index does not exist yet, meaning the child entry
        /// should be inserted into the internal node.
        /// Otherwise, the child entry should be updated at this index.
        pub maybe_i: Option<usize>,
        /// The key of the child entry.
        pub key: Rc<[u8]>,
        /// The page number (aka child pointer) of the child entry.
        pub page_num: u64,
    }

    /// A builder of a B+ tree internal node.
    pub struct InternalBuilder {
        buf: Box<[u8]>,
    }
}

/// A B+ tree internal node.
#[derive(Debug, Clone)]
pub struct Internal {
    buf: Box<[u8]>,
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
    pub fn find(&self, key: &[u8]) -> Option<usize> {
        unimplemented!();
    }

    /// Gets the child pointer at an index.
    pub fn get_child_pointer(&self, i: usize) -> Result<u64> {
        unimplemented!();
    }

    /// Deletes the child entry (i.e. key and child pointer) at an index.
    pub fn delete_child_entry(&self, i: usize) -> Result<Deletion> {
        // delete child entry @ i
        unimplemented!();
    }

    /// Updates the child entry at an index.
    pub fn update_child_entry(&self, i: usize, key: &[u8], page_num: u64) -> Result<Deletion> {
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

    fn get_key(&self, i: usize) -> &[u8] {
        unimplemented!();
    }
}
