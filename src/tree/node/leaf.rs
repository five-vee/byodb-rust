use crate::tree::buffer_store::{BufferStore, PooledBuf};
use crate::tree::error::NodeError;
use crate::tree::node::{self, NodeType, Result};

/// An enum representing the effect of a leaf node operation.
#[derive(Debug)]
pub enum LeafEffect {
    /// A leaf with 0 keys after a delete was performed on it.
    /// This is a special-case of `Underflow` done to avoid unnecessary
    /// page allocations, since empty non-root nodes aren't allowed.
    Empty,
    /// A newly created leaf that remained  "intact", i.e. it did not split.
    Intact(Leaf),
    /// The left and right splits of a leaf that was created.
    Split { left: Leaf, right: Leaf },
}

impl LeafEffect {
    fn take_intact(self) -> Leaf {
        match self {
            LeafEffect::Intact(leaf) => leaf,
            _ => panic!("{self:?} is not LeafEffect::Intact"),
        }
    }

    fn take_split(self) -> (Leaf, Leaf) {
        match self {
            LeafEffect::Split { left, right } => (left, right),
            _ => panic!("{self:?} is not LeafEffect::Split"),
        }
    }
}

    /// Gets the `i`th key in a leaf page buffer.
    pub fn get_key(buf: &[u8], i: usize) -> &[u8] {
        let offset = get_offset(buf, i) as usize;
        let num_keys = node::get_num_keys(buf) as usize;
        let key_len = u16::from_be_bytes([
            buf[4 + num_keys * 2 + offset],
            buf[4 + num_keys * 2 + offset + 1],
        ]) as usize;
        &buf[4 + num_keys * 2 + offset + 4..4 + num_keys * 2 + offset + 4 + key_len]
    }

    /// Gets the `i`th value in a leaf page buffer.
    pub fn get_value(buf: &[u8], i: usize) -> &[u8] {
        let offset = get_offset(buf, i) as usize;
        let num_keys = node::get_num_keys(buf) as usize;
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
        u16::from_be_bytes([buf[4 + 2 * (i - 1)], buf[4 + 2 * i - 1]]) as usize
    }

    /// Gets the number of bytes consumed by a page.
    pub fn get_num_bytes(buf: &[u8]) -> usize {
        let n = node::get_num_keys(buf);
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
    pub struct LeafBuilder<'a, S: BufferStore> {
        num_keys: usize,
        i: usize,
        cap: usize,
        store: &'a S,
        buf: PooledBuf,
    }

    impl<'a, S: BufferStore> LeafBuilder<'a, S> {
        /// Creates a new leaf builder.
        pub fn new(num_keys: usize, store: &'a S, allow_overflow: bool) -> Self {
            let (mut buf, cap) = if allow_overflow {
                (store.get_buf(node::PAGE_SIZE * 2), 2 * node::PAGE_SIZE - 4)
            } else {
                (store.get_buf(node::PAGE_SIZE), node::PAGE_SIZE)
            };
            node::set_page_header(&mut buf, NodeType::Leaf);
            Self {
                num_keys,
                i: 0,
                cap,
                store,
                buf,
            }
        }

        /// Adds a key-value pair to the builder.
        pub fn add_key_value(mut self, key: &[u8], val: &[u8]) -> Result<Self> {
            assert!(
                self.i < self.num_keys,
                "add_key_value() called {} times, cannot be called more times than num_keys = {}",
                self.i,
                self.num_keys
            );
            assert!(key.len() <= node::MAX_KEY_SIZE);
            assert!(val.len() <= node::MAX_VALUE_SIZE);

            let n = self.num_keys;
            let offset = set_next_offset(&mut self.buf, self.i, key, val);
            let pos = 4 + n * 2 + offset;
            assert!(pos + 4 + key.len() + val.len() <= self.cap,
                "builder unexpectedly overflowed; please call allow_overflow(), or don't add too many key-value pairs.");

            self.buf[pos..pos + 2].copy_from_slice(&(key.len() as u16).to_be_bytes());
            self.buf[pos + 2..pos + 4].copy_from_slice(&(val.len() as u16).to_be_bytes());
            self.buf[pos + 4..pos + 4 + key.len()].copy_from_slice(key);
            self.buf[pos + 4 + key.len()..pos + 4 + key.len() + val.len()].copy_from_slice(val);

            self.i += 1;
            node::set_num_keys(&mut self.buf, self.i);
            Ok(self)
        }

        /// Builds a leaf and optionally splits it if overflowed.
        pub fn build(self) -> Result<LeafEffect> {
            assert!(
                self.i == self.num_keys,
                "build() called after calling add_key_value() {} times < num_keys = {}",
                self.i,
                self.num_keys
            );
            if self.buf.len() == 0 {
                // Technically, an empty leaf is allowed.
                return Ok(LeafEffect::Empty);
            }
            if get_num_bytes(&self.buf) <= node::PAGE_SIZE {
                return Ok(LeafEffect::Intact(self.build_single()));
            }
            let (left, right) = self.build_split()?;
            Ok(LeafEffect::Split {
                left: left,
                right: right,
            })
        }

        /// Builds two splits of a leaf.
        fn build_split(self) -> Result<(Leaf, Leaf)> {
            let num_keys = self.num_keys as usize;
            let mut left_end: usize = 0;
            for i in 0..num_keys {
                // include i?
                let offset = get_offset(&self.buf, i + 1);
                if 4 + (i + 1) * 2 + offset <= node::PAGE_SIZE {
                    left_end = i + 1;
                }
            }

            let mut lb = Self::new(left_end, &self.store, false);
            for i in 0..left_end {
                lb = lb.add_key_value(get_key(&self.buf, i), get_value(&self.buf, i))?;
            }
            let mut rb = Self::new(num_keys - left_end, &self.store, false);
            for i in left_end..num_keys {
                rb = rb.add_key_value(get_key(&self.buf, i), get_value(&self.buf, i))?;
            }
            Ok((lb.build_single(), rb.build_single()))
        }

        /// Builds a leaf.
        fn build_single(self) -> Leaf {
            assert!(get_num_bytes(&self.buf) <= node::PAGE_SIZE);
            Leaf { buf: self.buf }
        }
    }

/// A B+ tree leaf node.
#[derive(Debug)]
pub struct Leaf {
    buf: PooledBuf,
}

impl Leaf {
    pub fn new<S: BufferStore>(store: &S) -> Self {
        let mut buf = store.get_buf(node::PAGE_SIZE);
        node::set_page_header(&mut buf, NodeType::Leaf);
        Self { buf }
    }

    /// Inserts a key-value pair.
    pub fn insert(&self, key: &[u8], val: &[u8]) -> Result<LeafEffect> {
        if key.len() > node::MAX_KEY_SIZE {
            return Err(NodeError::MaxKeySize(key.len()));
        }
        if val.len() > node::MAX_VALUE_SIZE {
            return Err(NodeError::MaxValueSize(val.len()));
        }
        if self.find(key).is_some() {
            return Err(NodeError::AlreadyExists);
        }
        let mut b = LeafBuilder::new(
            self.get_num_keys() + 1,
            &self.buf.pool,
            self.get_num_bytes() + 6 + key.len() + val.len() > node::PAGE_SIZE,
        );
        let mut added = false;
        for (k, v) in self.iter() {
            if !added && key < k {
                added = true;
                b = b.add_key_value(key, val)?;
            }
            b = b.add_key_value(k, v)?;
        }
        if !added {
            b = b.add_key_value(key, val)?;
        }
        b.build()
    }

    /// Updates the value corresponding to a key.
    pub fn update(&self, key: &[u8], val: &[u8]) -> Result<LeafEffect> {
        if key.len() > node::MAX_KEY_SIZE {
            return Err(NodeError::MaxKeySize(key.len()));
        }
        if val.len() > node::MAX_VALUE_SIZE {
            return Err(NodeError::MaxValueSize(val.len()));
        }
        let old_val = self.find(key);
        if old_val.is_none() {
            return Err(NodeError::KeyNotFound);
        }
        let old_val = old_val.unwrap();
        let mut b = LeafBuilder::new(self.get_num_keys(), &self.buf.pool, self.get_num_bytes() - old_val.len() + val.len() > node::PAGE_SIZE);
        let mut added = false;
        for (k, v) in self.iter() {
            if !added && key == k {
                added = true;
                b = b.add_key_value(key, val)?;
                continue;
            }
            b = b.add_key_value(k, v)?;
        }
        b.build()
    }

    /// Deletes a key and its corresponding value.
    pub fn delete(&self, key: &[u8]) -> Result<LeafEffect> {
        if key.len() > node::MAX_KEY_SIZE {
            return Err(NodeError::MaxKeySize(key.len()));
        }
        if self.find(key).is_none() {
            return Err(NodeError::KeyNotFound);
        }
        // Optimization: avoid memory allocation and
        // just return Deletion::Empty if only 1 key.
        let n = self.get_num_keys();
        if n == 1 {
            return Ok(LeafEffect::Empty);
        }
        let mut b = LeafBuilder::new(n - 1, &self.buf.pool, false);
        let mut added = false;
        for (k, v) in self.iter() {
            if !added && key == k {
                added = true;
                continue;
            }
            b = b.add_key_value(k, v)?;
        }
        b.build()
    }

    /// Finds the value corresponding to the queried key.
    pub fn find(&self, key: &[u8]) -> Option<&[u8]> {
        self.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    pub fn steal_or_merge(left: &Leaf, right: &Leaf) -> Result<LeafEffect> {
        // TODO: Actually determine if it really needs overflow.
        let allow_overflow = true;
        let mut b = LeafBuilder::new(left.get_num_keys() + right.get_num_keys(), &left.buf.pool, allow_overflow);
        for (key, val) in left.iter() {
            b = b.add_key_value(key, val)?;
        }
        for (key, val) in right.iter() {
            b = b.add_key_value(key, val)?;
        }
        b.build()
    }

    pub fn get_key(&self, i: usize) -> &[u8] {
        get_key(&self.buf, i)
    }

    pub fn get_num_keys(&self) -> usize {
        node::get_num_keys(&self.buf)
    }

    pub fn iter<'a>(&'a self) -> LeafIterator<'a> {
        LeafIterator {
            node: self,
            i: 0,
            n: self.get_num_keys(),
        }
    }

    fn get_value(&self, i: usize) -> &[u8] {
        get_value(&self.buf, i)
    }

    fn get_num_bytes(&self) -> usize {
        get_num_bytes(&self.buf)
    }
}

impl Clone for Leaf {
    fn clone(&self) -> Self {
        let buf = self.buf.pool.get_buf(self.buf.len());
        Self { buf }
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

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;
    use crate::tree::buffer_store::BufferPool;
    use crate::tree::node;
    use super::*;

    static TEST_ARENA_POOL: OnceLock<BufferPool> = OnceLock::new();

    fn pool() -> BufferPool {
        TEST_ARENA_POOL.get_or_init(|| BufferPool::new(8)).clone()
    }

    #[test]
    fn insert_intact() {
        let leaf = Leaf::new(&pool())
            .insert("hello".as_bytes(), "world".as_bytes())
            .unwrap()
            .take_intact();
        assert_eq!(leaf.find("hello".as_bytes()).unwrap(), "world".as_bytes());
    }

    #[test]
    fn insert_max_key_size() {
        let key = &[0u8; node::MAX_KEY_SIZE + 1];
        let result = Leaf::new(&pool()).insert(key, "val".as_bytes());
        assert!(matches!(result, Err(NodeError::MaxKeySize(x)) if x == node::MAX_KEY_SIZE + 1));
    }

    #[test]
    fn insert_max_value_size() {
        let val = &[0u8; node::MAX_VALUE_SIZE + 1];
        let result = Leaf::new(&pool()).insert("key".as_bytes(), val);
        assert!(matches!(result, Err(NodeError::MaxValueSize(x)) if x == node::MAX_VALUE_SIZE + 1));
    }

    #[test]
    fn insert_split() {
        // Insert 1 huge key-value.
        let key1 = &[0u8; node::MAX_KEY_SIZE];
        let val1 = &[0u8; node::MAX_VALUE_SIZE];
        let result = Leaf::new(&pool()).insert(key1, val1);
        assert!(matches!(result, Ok(LeafEffect::Intact(_))), "1st insert result: {result:?}");
        let leaf = result.unwrap().take_intact();

        // Insert another huge key-value to trigger splitting.
        let key2 = &[1u8; node::MAX_KEY_SIZE];
        let val2 = &[1u8; node::MAX_VALUE_SIZE];
        let result = leaf.insert(key2, val2);
        assert!(matches!(result, Ok(LeafEffect::Split{..})), "2nd insert result: {result:?}");
        let (left, right) = result.unwrap().take_split();
        drop(leaf);
        assert_eq!(left.get_num_keys(), 1);
        assert_eq!(right.get_num_keys(), 1);
        assert_eq!(left.find(key1).unwrap(), val1);
        assert_eq!(right.find(key2).unwrap(), val2);
    }
}
