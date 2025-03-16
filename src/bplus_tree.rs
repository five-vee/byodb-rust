//! # Design a node format
//! 
//! Here is our node format. The 2nd row is the encoded field size in bytes.
//! 
//! ```
//! | type | nkeys |  pointers  |  offsets   | key-values | unused |
//! |  2B  |   2B  | nkeys × 8B | nkeys × 2B |     ...    |        |
//! ```
//! 
//! The format starts with a 4-bytes header:
//! 
//! * `type` is the node type (leaf or internal).
//! * `nkeys` is the number of keys (and the number of child pointers).
//! 
//! Then an array of child pointers and the KV pairs follow. The format is the
//! same for both leaf and internal nodes, so the pointer array is simply
//! unused for leaf nodes.
//! 
//! Each KV pair is prefixed by its size. For internal nodes,
//! the value size is 0.
//! 
//! ```
//! | key_size | val_size | key | val |
//! |    2B    |    2B    | ... | ... |
//! ```
//! 
//! The encoded KV pairs are concatenated. To find the `n`th KV pair, we have
//! to read all previous pairs. This is avoided by storing the offset of each
//! KV pair.
//! 
//! For example, a leaf node `{"k1":"hi", "k3":"hello"}` is encoded as:
//! 
//! ```
//! | type | nkeys | pointers | offsets |            key-values           | unused |
//! |   2  |   2   | nil nil  |  8 19   | 2 2 "k1" "hi"  2 5 "k3" "hello" |        |
//! |  2B  |  2B   |   2×8B   |  2×4B   | 4B + 2B + 2B + 4B + 2B + 5B     |        |
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
//! We’ll set the node size to 4K, which is the typical OS page size. However,
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

/// The B+ Tree node type of the page.
enum NodeType {
    Leaf,
    Internal,
}

impl From<u16> for NodeType {
    fn from(value: u16) -> Self {
        match value {
            0b01 => NodeType::Leaf,
            0b10 => NodeType::Internal,
            _ => panic!("Invalid NodeType value: {:b}", value),
        }
    }
}

/// The on-disk buffer representation of a B+ Tree node.
struct BNode {
    buf: [u8; Self::BTREE_PAGE_SIZE]
}

impl BNode {
    const BTREE_PAGE_SIZE: usize = 4096;
    const BTREE_MAX_KEY_SIZE: usize = 1000;
    const BTREE_MAX_VAL_SIZE: usize = 3000;

    /// Gets the node type of the page.
    fn get_node_type(&self) -> NodeType {
        u16::from_le_bytes([self.buf[0], self.buf[1]]).into()
    }

    /// Gets the number of keys in the page.
    fn get_num_keys(&self) -> u16 {
        u16::from_le_bytes([self.buf[2], self.buf[3]]).into()
    }

    /// Sets the page header.
    fn set_header(&mut self, node_type: NodeType, num_keys: u16) {
        self.buf[0..2].copy_from_slice(&(node_type as u16).to_le_bytes());
        self.buf[2..4].copy_from_slice(&num_keys.to_le_bytes());
    }

    /// Gets the `i`th child pointer.
    fn get_child_pointer(&self, i: u16) -> u64 {
        assert!(i < self.get_num_keys());
        let pos = usize::from(4 + 8*i);
        let sub: [u8; 8] = std::array::from_fn(|j| self.buf[pos+j]);
        u64::from_le_bytes(sub)
    }

    /// Sets the `i`th child pointer.
    fn set_child_pointer(&mut self, i: u16, val: u64) {
        assert!(i < self.get_num_keys());
        let pos = usize::from(4 + 8*i);
        self.buf[pos..pos+8].copy_from_slice(&val.to_le_bytes());
    }

    /// Gets the `i`th offset.
    fn get_offset(&self, i: u16) -> u16 {
        assert!(i <= self.get_num_keys());
        if i == 0 {
            return 0
        }
        let pos = usize::from(4 + 8*self.get_num_keys() + 2*(i - 1));
        u16::from_le_bytes([self.buf[pos], self.buf[pos+1]])
    }

    /// Sets the `i`th offset, where `0 < i <= n`.
    fn set_offset(&mut self, i: u16, offset: u16) {
        assert!(0 < i && i <= self.get_num_keys());
        let pos = usize::from(4 + 8*self.get_num_keys() + 2*(i - 1));
        self.buf[pos..pos+2].copy_from_slice(&offset.to_le_bytes());
    }

    /// Gets the position of the `i`th key-value pair.
    /// If `i == n`, then it returns the byte position right after the last
    /// key-value pair.
    fn get_key_value_position(&self, i: u16) -> usize {
        assert!(i <= self.get_num_keys());
        usize::from(4 + 10*self.get_num_keys() + self.get_offset(i))
    }

    /// Gets the `i`th key.
    fn get_key(&self, i: u16) -> &[u8] {
        assert!(i < self.get_num_keys());
        let pos = self.get_key_value_position(i);
        let key_len = u16::from_le_bytes([self.buf[pos], self.buf[pos + 1]]);
        &self.buf[pos + 4 .. pos + 4 + usize::from(key_len)]
    }

    /// Gets the `i`th value.
    fn get_value(&self, i: u16) -> &[u8] {
        assert!(i < self.get_num_keys());
        let pos = self.get_key_value_position(i);
        let key_len = u16::from_le_bytes([self.buf[pos], self.buf[pos + 1]]);
        let val_len = u16::from_le_bytes([self.buf[pos + 2], self.buf[pos + 3]]);
        &self.buf[pos + 4 + usize::from(key_len) .. pos + 4 + usize::from(key_len) + usize::from(val_len)]
    }

    /// Sets the `i`th key and value.
    fn set_key_value(&mut self, i: u16, k: &[u8], v: &[u8]) {
        assert!(i < self.get_num_keys());
        assert!(k.len() <= Self::BTREE_MAX_KEY_SIZE);
        assert!(v.len() <= Self::BTREE_MAX_VAL_SIZE);
        let key_len = k.len() as u16;
        let val_len = v.len() as u16;

        // Set the next offset.
        self.set_offset(i + 1, (self.get_key_value_position(i) as u16) + 4 + key_len + val_len);

        // Fill in the key-value pair.
        {
            let pos = self.get_key_value_position(i);
            self.buf[pos..pos+2].copy_from_slice(&key_len.to_le_bytes());
            self.buf[pos+2..pos+4].copy_from_slice(&val_len.to_le_bytes());
            self.buf[pos+4..pos+4+usize::from(key_len)].copy_from_slice(k);
            self.buf[pos+4+usize::from(key_len)..pos+4+usize::from(key_len)+usize::from(val_len)].copy_from_slice(v);
        }

        assert!(self.num_bytes() <= Self::BTREE_PAGE_SIZE);
    }

    /// Gets the number of bytes taken up by the B+ Tree node.
    fn num_bytes(&self) -> usize {
        self.get_key_value_position(self.get_num_keys())
    }
}
