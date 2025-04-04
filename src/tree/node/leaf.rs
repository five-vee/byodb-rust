use crate::tree::buffer_store::BufferStore;
use crate::tree::error::NodeError;
use crate::tree::node::{self, NodeType, Result};

/// An enum representing the effect of a leaf node operation.
#[derive(Debug)]
pub enum LeafEffect<B: BufferStore> {
    /// A leaf with 0 keys after a delete was performed on it.
    /// This is a special-case of `Underflow` done to avoid unnecessary
    /// page allocations, since empty non-root nodes aren't allowed.
    Empty,
    /// A newly created leaf that remained  "intact", i.e. it did not split.
    Intact(Leaf<B>),
    /// The left and right splits of a leaf that was created.
    Split { left: Leaf<B>, right: Leaf<B> },
}

impl<B: BufferStore> LeafEffect<B> {
    #[allow(dead_code)]
    fn take_intact(self) -> Leaf<B> {
        match self {
            LeafEffect::Intact(leaf) => leaf,
            _ => panic!("{self:?} is not LeafEffect::Intact"),
        }
    }

    #[allow(dead_code)]
    fn take_split(self) -> (Leaf<B>, Leaf<B>) {
        match self {
            LeafEffect::Split { left, right } => (left, right),
            _ => panic!("{self:?} is not LeafEffect::Split"),
        }
    }
}

/// Gets the `i`th key in a leaf page buffer.
pub fn get_key(buf: &[u8], i: usize) -> &[u8] {
    let offset = get_offset(buf, i);
    let num_keys = node::get_num_keys(buf);
    let key_len = u16::from_be_bytes([
        buf[4 + num_keys * 2 + offset],
        buf[4 + num_keys * 2 + offset + 1],
    ]) as usize;
    &buf[4 + num_keys * 2 + offset + 4..4 + num_keys * 2 + offset + 4 + key_len]
}

/// Gets the `i`th value in a leaf page buffer.
pub fn get_value(buf: &[u8], i: usize) -> &[u8] {
    let offset = get_offset(buf, i);
    let num_keys = node::get_num_keys(buf);
    let key_len = u16::from_be_bytes([
        buf[4 + num_keys * 2 + offset],
        buf[4 + num_keys * 2 + offset + 1],
    ]) as usize;
    let val_len = u16::from_be_bytes([
        buf[4 + num_keys * 2 + offset + 2],
        buf[4 + num_keys * 2 + offset + 3],
    ]) as usize;
    &buf[4 + num_keys * 2 + offset + 4 + key_len..4 + num_keys * 2 + offset + 4 + key_len + val_len]
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
    let next_i = i + 1;
    buf[4 + 2 * (next_i - 1)..4 + 2 * next_i].copy_from_slice(&(next_offset as u16).to_be_bytes());
    curr_offset
}

// A builder of a B+ tree leaf node.
pub struct LeafBuilder<'a, B: BufferStore> {
    i: usize,
    cap: usize,
    store: &'a B,
    buf: B::B,
}

impl<'a, B: BufferStore> LeafBuilder<'a, B> {
    /// Creates a new leaf builder.
    pub fn new(num_keys: usize, store: &'a B, allow_overflow: bool) -> Self {
        let (mut buf, cap) = if allow_overflow {
            (store.get_buf(node::PAGE_SIZE * 2), 2 * node::PAGE_SIZE - 4)
        } else {
            (store.get_buf(node::PAGE_SIZE), node::PAGE_SIZE)
        };
        node::set_page_header(&mut buf, NodeType::Leaf);
        node::set_num_keys(&mut buf, num_keys);
        Self {
            i: 0,
            cap,
            store,
            buf,
        }
    }

    /// Adds a key-value pair to the builder.
    pub fn add_key_value(mut self, key: &[u8], val: &[u8]) -> Result<Self> {
        let n = node::get_num_keys(&self.buf);
        assert!(
            self.i < n,
            "add_key_value() called {} times, cannot be called more times than num_keys = {}",
            self.i,
            n
        );
        assert!(key.len() <= node::MAX_KEY_SIZE);
        assert!(val.len() <= node::MAX_VALUE_SIZE);

        let offset = set_next_offset(&mut self.buf, self.i, key, val);
        let pos = 4 + n * 2 + offset;
        assert!(
            pos + 4 + key.len() + val.len() <= self.cap,
            "builder unexpectedly overflowed: i = {}, n = {}",
            self.i,
            n
        );

        self.buf[pos..pos + 2].copy_from_slice(&(key.len() as u16).to_be_bytes());
        self.buf[pos + 2..pos + 4].copy_from_slice(&(val.len() as u16).to_be_bytes());
        self.buf[pos + 4..pos + 4 + key.len()].copy_from_slice(key);
        self.buf[pos + 4 + key.len()..pos + 4 + key.len() + val.len()].copy_from_slice(val);

        self.i += 1;
        Ok(self)
    }

    /// Builds a leaf and optionally splits it if overflowed.
    pub fn build(self) -> Result<LeafEffect<B>> {
        let n = node::get_num_keys(&self.buf);
        assert!(
            self.i == n,
            "build() called after calling add_key_value() {} times < num_keys = {}",
            self.i,
            n
        );
        if self.buf.len() == 0 {
            // Technically, an empty leaf is allowed.
            return Ok(LeafEffect::Empty);
        }
        if get_num_bytes(&self.buf) <= node::PAGE_SIZE {
            return Ok(LeafEffect::Intact(self.build_single()));
        }
        let (left, right) = self.build_split()?;
        Ok(LeafEffect::Split { left, right })
    }

    /// Builds two splits of a leaf.
    fn build_split(self) -> Result<(Leaf<B>, Leaf<B>)> {
        let n = node::get_num_keys(&self.buf);
        // Try to have at least 2 keys in each to be sufficient
        // in each of left and right.
        let mut left_end = (2..=n - 2).rev().find(|i| {
            let next_offset = get_offset(&self.buf, *i);
            4 + *i * 2 + next_offset <= node::PAGE_SIZE
        });
        if left_end.is_none() {
            // Relax the sufficiency requirement by just making sure
            // each left and right fits within a page.
            left_end = (0..n).rev().find(|i| {
                let next_offset = get_offset(&self.buf, *i);
                4 + *i + 2 + next_offset <= node::PAGE_SIZE
            })
        }
        let left_end = left_end.unwrap();

        let mut lb = Self::new(left_end, self.store, false);
        for i in 0..left_end {
            lb = lb.add_key_value(get_key(&self.buf, i), get_value(&self.buf, i))?;
        }
        let mut rb = Self::new(n - left_end, self.store, false);
        for i in left_end..n {
            rb = rb.add_key_value(get_key(&self.buf, i), get_value(&self.buf, i))?;
        }
        Ok((lb.build_single(), rb.build_single()))
    }

    /// Builds a leaf.
    fn build_single(self) -> Leaf<B> {
        assert!(get_num_bytes(&self.buf) <= node::PAGE_SIZE);
        Leaf {
            buf: self.buf,
            store: self.store.clone(),
        }
    }
}

/// A B+ tree leaf node.
#[derive(Debug)]
pub struct Leaf<B: BufferStore> {
    buf: B::B,
    store: B,
}

impl<B: BufferStore> Leaf<B> {
    pub fn new(store: &B) -> Self {
        let mut buf = store.get_buf(node::PAGE_SIZE);
        node::set_page_header(&mut buf, NodeType::Leaf);
        Self {
            buf,
            store: store.clone(),
        }
    }

    /// Inserts a key-value pair.
    pub fn insert(&self, key: &[u8], val: &[u8]) -> Result<LeafEffect<B>> {
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
            &self.store,
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
    pub fn update(&self, key: &[u8], val: &[u8]) -> Result<LeafEffect<B>> {
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
        let mut b = LeafBuilder::new(
            self.get_num_keys(),
            &self.store,
            self.get_num_bytes() - old_val.len() + val.len() > node::PAGE_SIZE,
        );
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
    pub fn delete(&self, key: &[u8]) -> Result<LeafEffect<B>> {
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
        let mut b = LeafBuilder::new(n - 1, &self.store, false);
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

    pub fn steal_or_merge(left: &Leaf<B>, right: &Leaf<B>) -> Result<LeafEffect<B>> {
        // TODO: Actually determine if it really needs overflow.
        let allow_overflow = true;
        let mut b = LeafBuilder::new(
            left.get_num_keys() + right.get_num_keys(),
            &left.store,
            allow_overflow,
        );
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

    pub fn get_value(&self, i: usize) -> &[u8] {
        get_value(&self.buf, i)
    }

    pub fn get_num_keys(&self) -> usize {
        node::get_num_keys(&self.buf)
    }

    pub fn iter(&self) -> LeafIterator<B> {
        LeafIterator {
            node: self,
            i: 0,
            n: self.get_num_keys(),
        }
    }

    pub fn get_num_bytes(&self) -> usize {
        get_num_bytes(&self.buf)
    }
}

impl<B: BufferStore> Clone for Leaf<B> {
    fn clone(&self) -> Self {
        let mut buf = self.store.get_buf(self.buf.len());
        buf.copy_from_slice(&self.buf);
        Self {
            buf,
            store: self.store.clone(),
        }
    }
}

/// An key-value iterator for a leaf node.
pub struct LeafIterator<'a, B: BufferStore> {
    node: &'a Leaf<B>,
    i: usize,
    n: usize,
}

impl<'a, B: BufferStore> Iterator for LeafIterator<'a, B> {
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
    use super::*;
    use crate::tree::buffer_store::Heap;
    use crate::tree::node;

    static TEST_HEAP_STORE: Heap = Heap {};

    #[test]
    fn insert_intact() {
        let leaf = Leaf::new(&TEST_HEAP_STORE)
            .insert("hello".as_bytes(), "world".as_bytes())
            .unwrap()
            .take_intact();
        assert_eq!(
            leaf.iter().collect::<Vec<_>>(),
            vec![("hello".as_bytes(), "world".as_bytes())]
        );
        assert_eq!(leaf.find("hello".as_bytes()).unwrap(), "world".as_bytes());
    }

    #[test]
    fn insert_max_key_size() {
        let key = &[0u8; node::MAX_KEY_SIZE + 1];
        let result = Leaf::new(&TEST_HEAP_STORE).insert(key, "val".as_bytes());
        assert!(matches!(result, Err(NodeError::MaxKeySize(x)) if x == node::MAX_KEY_SIZE + 1));
    }

    #[test]
    fn insert_max_value_size() {
        let val = &[0u8; node::MAX_VALUE_SIZE + 1];
        let result = Leaf::new(&TEST_HEAP_STORE).insert("key".as_bytes(), val);
        assert!(matches!(result, Err(NodeError::MaxValueSize(x)) if x == node::MAX_VALUE_SIZE + 1));
    }

    #[test]
    fn insert_split() {
        // Insert 1 huge key-value.
        let key1 = &[0u8; node::MAX_KEY_SIZE];
        let val1 = &[0u8; node::MAX_VALUE_SIZE];
        let result = Leaf::new(&TEST_HEAP_STORE).insert(key1, val1);
        assert!(
            matches!(result, Ok(LeafEffect::Intact(_))),
            "1st insert result: {result:?}"
        );
        let leaf = result.unwrap().take_intact();

        // Insert another huge key-value to trigger splitting.
        let key2 = &[1u8; node::MAX_KEY_SIZE];
        let val2 = &[1u8; node::MAX_VALUE_SIZE];
        let result = leaf.insert(key2, val2);
        assert!(
            matches!(result, Ok(LeafEffect::Split { .. })),
            "2nd insert result: {result:?}"
        );
        let (left, right) = result.unwrap().take_split();
        drop(leaf);
        assert_eq!(left.get_num_keys(), 1);
        assert_eq!(right.get_num_keys(), 1);
        assert_eq!(left.find(key1).unwrap(), val1);
        assert_eq!(right.find(key2).unwrap(), val2);
    }

    #[test]
    fn find_some() {
        let leaf = LeafBuilder::new(1, &TEST_HEAP_STORE, false)
            .add_key_value("key".as_bytes(), "val".as_bytes())
            .unwrap()
            .build()
            .unwrap()
            .take_intact();
        assert!(matches!(leaf.find("key".as_bytes()), Some(v) if v == "val".as_bytes()));
    }

    #[test]
    fn find_none() {
        let leaf = Leaf::new(&TEST_HEAP_STORE);
        assert!(leaf.find("key".as_bytes()).is_none())
    }

    #[test]
    fn iter() {
        let leaf = LeafBuilder::new(2, &TEST_HEAP_STORE, false)
            .add_key_value("key1".as_bytes(), "val1".as_bytes())
            .unwrap()
            .add_key_value("key2".as_bytes(), "val2".as_bytes())
            .unwrap()
            .build()
            .unwrap()
            .take_intact();
        let got = leaf.iter().collect::<Vec<_>>();
        assert_eq!(
            got,
            vec![
                ("key1".as_bytes(), "val1".as_bytes()),
                ("key2".as_bytes(), "val2".as_bytes())
            ]
        );
    }

    #[test]
    fn iter_empty() {
        let leaf = Leaf::new(&TEST_HEAP_STORE);
        assert_eq!(leaf.iter().count(), 0);
    }

    #[test]
    fn update_intact() {
        let leaf = LeafBuilder::new(2, &TEST_HEAP_STORE, false)
            .add_key_value("key1".as_bytes(), "val1".as_bytes())
            .unwrap()
            .add_key_value("key2".as_bytes(), "val2".as_bytes())
            .unwrap()
            .build()
            .unwrap()
            .take_intact();

        let leaf = leaf
            .update("key1".as_bytes(), "val1_new".as_bytes())
            .unwrap()
            .take_intact();

        assert_eq!(
            leaf.iter().collect::<Vec<_>>(),
            vec![
                ("key1".as_bytes(), "val1_new".as_bytes()),
                ("key2".as_bytes(), "val2".as_bytes())
            ]
        );
        assert_eq!(leaf.find("key1".as_bytes()).unwrap(), "val1_new".as_bytes());
    }

    #[test]
    fn update_split() {
        let leaf = LeafBuilder::new(2, &TEST_HEAP_STORE, false)
            .add_key_value(&[0u8; node::MAX_KEY_SIZE], &[0u8; node::MAX_VALUE_SIZE])
            .unwrap()
            .add_key_value("1".as_bytes(), "1".as_bytes())
            .unwrap()
            .build()
            .unwrap()
            .take_intact();

        // Update with a huge value to trigger splitting.
        let (left, right) = leaf
            .update("1".as_bytes(), &[1u8; node::MAX_VALUE_SIZE])
            .unwrap()
            .take_split();
        drop(leaf);
        assert_eq!(left.get_num_keys(), 1);
        assert_eq!(right.get_num_keys(), 1);
        assert_eq!(
            left.find(&[0u8; node::MAX_KEY_SIZE]).unwrap(),
            &[0u8; node::MAX_VALUE_SIZE]
        );
        assert_eq!(
            right.find("1".as_bytes()).unwrap(),
            &[1u8; node::MAX_VALUE_SIZE]
        );
    }

    #[test]
    fn update_max_key_size() {
        let key = &[0u8; node::MAX_KEY_SIZE + 1];
        let result = Leaf::new(&TEST_HEAP_STORE).update(key, "val".as_bytes());
        assert!(matches!(result, Err(NodeError::MaxKeySize(x)) if x == node::MAX_KEY_SIZE + 1));
    }

    #[test]
    fn update_max_value_size() {
        let leaf = LeafBuilder::new(1, &TEST_HEAP_STORE, false)
            .add_key_value("key".as_bytes(), "val".as_bytes())
            .unwrap()
            .build()
            .unwrap()
            .take_intact();
        let val = &[0u8; node::MAX_VALUE_SIZE + 1];
        let result = leaf.update("key".as_bytes(), val);
        assert!(matches!(result, Err(NodeError::MaxValueSize(x)) if x == node::MAX_VALUE_SIZE + 1));
    }

    #[test]
    fn update_non_existent() {
        let result = Leaf::new(&TEST_HEAP_STORE).update("key".as_bytes(), "val".as_bytes());
        assert!(matches!(result, Err(NodeError::KeyNotFound)));
    }

    #[test]
    fn delete_intact() {
        let leaf = LeafBuilder::new(2, &TEST_HEAP_STORE, false)
            .add_key_value("key1".as_bytes(), "val1".as_bytes())
            .unwrap()
            .add_key_value("key2".as_bytes(), "val2".as_bytes())
            .unwrap()
            .build()
            .unwrap()
            .take_intact();

        let leaf = leaf.delete("key1".as_bytes()).unwrap().take_intact();

        assert_eq!(
            leaf.iter().collect::<Vec<_>>(),
            vec![("key2".as_bytes(), "val2".as_bytes())]
        );
        assert!(leaf.find("key1".as_bytes()).is_none());
    }

    #[test]
    fn delete_empty() {
        let leaf = LeafBuilder::new(1, &TEST_HEAP_STORE, false)
            .add_key_value("key".as_bytes(), "val".as_bytes())
            .unwrap()
            .build()
            .unwrap()
            .take_intact();
        let effect = leaf.delete("key".as_bytes()).unwrap();
        assert!(matches!(effect, LeafEffect::Empty));
    }

    #[test]
    fn delete_non_existent() {
        let result = Leaf::new(&TEST_HEAP_STORE).delete("key".as_bytes());
        assert!(matches!(result, Err(NodeError::KeyNotFound)));
    }

    #[test]
    fn steal_or_merge_steal() {
        let left = LeafBuilder::new(1, &TEST_HEAP_STORE, false)
            .add_key_value(&[1; node::MAX_KEY_SIZE], &[1; node::MAX_VALUE_SIZE])
            .unwrap()
            .build()
            .unwrap()
            .take_intact();

        let right = LeafBuilder::new(3, &TEST_HEAP_STORE, false)
            .add_key_value(&[2], &[2])
            .unwrap()
            .add_key_value(&[3], &[3])
            .unwrap()
            .add_key_value(&[4; node::MAX_KEY_SIZE], &[4; node::MAX_VALUE_SIZE])
            .unwrap()
            .build()
            .unwrap()
            .take_intact();

        let (left, right) = Leaf::steal_or_merge(&left, &right).unwrap().take_split();
        assert!(left.get_num_keys() >= 2);
        assert!(right.get_num_keys() >= 2);
        assert!(right.get_num_keys() < 3);
        let chained = left.iter().chain(right.iter()).collect::<Vec<_>>();
        assert_eq!(
            chained,
            vec![
                (&[1; node::MAX_KEY_SIZE][..], &[1; node::MAX_VALUE_SIZE][..]),
                (&[2], &[2]),
                (&[3], &[3]),
                (&[4; node::MAX_KEY_SIZE], &[4; node::MAX_VALUE_SIZE]),
            ]
        );
    }

    #[test]
    fn steal_or_merge_merge() {
        let left = LeafBuilder::new(1, &TEST_HEAP_STORE, false)
            .add_key_value(&[1], &[1])
            .unwrap()
            .build()
            .unwrap()
            .take_intact();

        let right = LeafBuilder::new(2, &TEST_HEAP_STORE, false)
            .add_key_value(&[2], &[2])
            .unwrap()
            .add_key_value(&[3], &[3])
            .unwrap()
            .build()
            .unwrap()
            .take_intact();

        let merged = Leaf::steal_or_merge(&left, &right).unwrap().take_intact();
        assert_eq!(
            merged.iter().collect::<Vec<_>>(),
            vec![(&[1][..], &[1][..]), (&[2], &[2]), (&[3], &[3]),]
        );
    }
}
