//! # Design a node format
//! 
//! Here is our node format. The 2nd row is the encoded field size in bytes.
//! 
//! ```ignore
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
//! ```ignore
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
//! ```ignore
//! | type | nkeys | pointers | offsets |            key-values           | unused |
//! |   2  |   2   | nil nil  |  8 19   | 2 2 "k1" "hi"  2 5 "k3" "hello" |        |
//! |  2B  |  2B   |   2×8B   |  2×2B   | 4B + 2B + 2B + 4B + 2B + 5B     |        |
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
#[derive(Debug)]
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
#[derive(Clone, Copy)]
struct BNode {
    buf: [u8; Self::BTREE_PAGE_SIZE]
}

impl Default for BNode {
    fn default() -> Self {
        BNode{buf: [0; Self::BTREE_PAGE_SIZE]}
    }
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

    /// Sets the node type of the page.
    fn set_node_type(&mut self, node_type: NodeType) {
        self.buf[0..2].copy_from_slice(&(node_type as u16).to_le_bytes());
    }

    /// Sets the number of keys in the page.
    fn set_num_keys(&mut self, num_keys: u16) {
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
    /// If not careful, this can overwrite the key-values for `i+1` onwards;
    /// please use `copy_on_insert`/`copy_on_update`
    /// or `inplace_insert`/`inplace_update` instead.
    fn set_key_value(&mut self, i: u16, k: &[u8], v: &[u8]) {
        assert!(i < self.get_num_keys());
        assert!(k.len() <= Self::BTREE_MAX_KEY_SIZE);
        assert!(v.len() <= Self::BTREE_MAX_VAL_SIZE);
        let key_len = k.len() as u16;
        let val_len = v.len() as u16;

        // Set the next offset.
        self.set_offset(i + 1, (self.get_key_value_position(i) as u16) + 4 + key_len + val_len);

        // Fill in the key-value pair.
        let pos = self.get_key_value_position(i);
        self.buf[pos..pos+2].copy_from_slice(&key_len.to_le_bytes());
        self.buf[pos+2..pos+4].copy_from_slice(&val_len.to_le_bytes());
        self.buf[pos+4..pos+4+usize::from(key_len)].copy_from_slice(k);
        self.buf[pos+4+usize::from(key_len)..pos+4+usize::from(key_len)+usize::from(val_len)].copy_from_slice(v);
    }

    /// Gets the number of bytes taken up by the B+ Tree node.
    fn get_num_bytes(&self) -> usize {
        self.get_key_value_position(self.get_num_keys())
    }

    /// Creates a copy of the node, but with a newly inserted key-value pair at
    /// position `i`.
    fn copy_on_insert(&self, i: u16, k: &[u8], v: &[u8]) -> Self {
        // For simplicity, we re-use inplace_insert.
        let mut new_node = self.clone();
        new_node.inplace_insert(i, k, v);
        new_node
    }

    /// Creates a copy of the node, but with an updated key-value pair at
    /// position `i`.
    fn copy_on_update(&self, i: u16, k: &[u8], v: &[u8]) -> Self {
        // For simplicity, we re-use inplace_update.
        let mut new_node = self.clone();
        new_node.inplace_update(i, k, v);
        new_node
    }

    /// Find the index of the (lexicographically) largest key
    /// less-than-or-equal (i.e. lte) to the query.
    /// If no such key exists, retuns None.
    fn lookup_lte(&self, query_key: &[u8]) -> Option<usize> {
        // Linear search for simplicity.
        let mut result = None::<usize>;
        for i in 0..self.get_num_keys() {
            if self.get_key(i) <= query_key {
                result = Some(i as usize);
            }
        }
        result
    }

    /// Splits a node to stay within the page size limit.
    fn copy_on_split(&self) -> (Self, Self) {
        let old_num_keys = self.get_num_keys();
        assert!(old_num_keys > 1);
        // Lookup the largest index at which the right node
        // will have >= half the page size.
        //
        // The right node can bigger than the left b/c by convention, we
        // insert/update the left node after split.
        //
        // Use linear search for simplicity.
        let mut right_i: u16 = 1;
        for i in 2..old_num_keys {
            let left_num_bytes = self.get_key_value_position(i);
            let right_num_bytes = self.get_num_bytes() - left_num_bytes;
            if right_num_bytes < Self::BTREE_PAGE_SIZE / 2 {
                break;
            }
            right_i = i;
        }
        let mut left_node = BNode::default();
        let mut right_node = BNode::default();
        left_node.set_node_type(self.get_node_type());
        left_node.set_num_keys(right_i);
        for j in 0..right_i {
            left_node.set_child_pointer(j, self.get_child_pointer(j));
            left_node.set_key_value(j, self.get_key(j), self.get_value(j));
        }
        right_node.set_node_type(self.get_node_type());
        right_node.set_num_keys(old_num_keys - right_i);
        for j in right_i..old_num_keys {
            right_node.set_child_pointer(j - right_i, self.get_child_pointer(j));
            right_node.set_key_value(j - right_i, self.get_key(j), self.get_value(j));
        }
        (left_node, right_node)
    }

    /// Inserts a key-value pair in-place at position `i`.
    fn inplace_insert(&mut self, i: u16, k: &[u8], v: &[u8]) {
        // Shift key-values to the right:
        // 8 for child pointers array
        // 2 for offsets array
        // 2 for key_size                (only for i <= j < n)
        // 2 for val_size                (only for i <= j < n)
        // k.len()                       (only for i <= j < n)
        // v.len()                       (only for i <= j < n)
        //
        // total: 10                     (only for 0 <= j < i)
        // total: 14 + k.len() + v.len() (only for i <= j < n)
        {
            let src_start = self.get_key_value_position(i);
            let src_end = self.get_num_bytes();
            let shift = 14 + k.len() + v.len();
            self.buf.copy_within(src_start..src_end, src_start + shift);
        }
        if i > 0 {
            let src_start = self.get_key_value_position(0);
            let src_end = src_start + self.get_key_value_position(i);
            self.buf.copy_within(src_start..src_end, src_start+10);
        }

        // Shift offsets array to the right:
        // 8 for child pointers array
        // 2 for offsets array (only for i <= j < n)
        //
        // total: 8            (only for 0 <= j < i)
        // total: 10           (only for i <= j < n)
        //
        // Furthermore, increase the offset values (only for i <= j < n):
        // 2 for key_size
        // 2 for val_size
        // k.len()
        // v.len()
        //
        // total: 4 + k.len() + v.len()
        {
            let old_num_keys = self.get_num_keys();
            let src_start = usize::from(4 + 8*old_num_keys + i*2);
            let src_end = src_start + usize::from(old_num_keys-i)*2;
            self.buf.copy_within(src_start..src_end, src_start + 10);
            let offset_shift = 4 + (k.len() as u16) + (v.len() as u16);
            for j in i..old_num_keys {
                self.set_offset(j+1, self.get_offset(j+1) + offset_shift);
            }
        }
        if i > 0 {
            let src_start = usize::from(4 + 8*self.get_num_keys());
            let src_end = src_start + usize::from(i)*2;
            self.buf.copy_within(src_start..src_end, src_start+8);
        }

        // Shift pointers array by 8 only for i <= j < n:
        {
            let src_start = 4 + usize::from(8*i);
            let src_end = src_start + usize::from(self.get_num_keys() - i)*8;
            self.buf.copy_within(src_start..src_end, src_start+8);
        }

        self.set_num_keys(self.get_num_keys()+1);
        self.set_child_pointer(i, 0 /* safe default */);
        self.set_key_value(i, k, v);
    }

    /// Updates a key-value pair in-place at position `i`.
    fn inplace_update(&mut self, i: u16, k: &[u8], v: &[u8]) {
        // Short-cuts.
        if k.len() == self.get_key(i).len() && v.len() == self.get_value(i).len() {
            self.set_key_value(i, k, v);
            return
        }
        if i == self.get_num_keys() - 1 {
            self.set_key_value(i, k, v);
            return
        }

        // Shift key-values (only for i < j < n):
        // k.len()
        // v.len()
        // -old_k.len()
        // -old_v.len()
        {
            let src_start = self.get_key_value_position(i + 1);
            let src_end = self.get_num_bytes();
            let shift = k.len() + v.len() - self.get_key(i).len() - self.get_value(i).len();
            self.buf.copy_within(src_start..src_end, src_start + shift);
        }

        // Increase the offset values (only for i < j < n):
        // k.len()
        // v.len()
        // -old_k.len()
        // -old_v.len()
        {
            let offset_shift = (k.len() + v.len() - self.get_key(i).len() - self.get_value(i).len()) as u16;
            for j in i+1..self.get_num_keys() {
                self.set_offset(j+1, self.get_offset(j+1) + offset_shift);
            }
        }

        self.set_key_value(i, k, v);
    }
}
