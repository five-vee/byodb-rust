use std::iter::Peekable;

use crate::tree::consts;
use crate::tree::error::NodeError;
use crate::tree::node::{self, NodeType, Result};
use crate::tree::page_store::{PageStore, ReadOnlyPage};

/// An enum representing the effect of a leaf node operation.
pub enum LeafEffect<P: PageStore> {
    /// A leaf with 0 keys after a delete was performed on it.
    /// This is a special-case of `Underflow` done to avoid unnecessary
    /// page allocations, since empty non-root nodes aren't allowed.
    Empty,
    /// A newly created leaf that remained  "intact", i.e. it did not split.
    Intact(Leaf<P>),
    /// The left and right splits of a leaf that was created.
    Split { left: Leaf<P>, right: Leaf<P> },
}

impl<P: PageStore> LeafEffect<P> {
    #[allow(dead_code)]
    fn take_intact(self) -> Leaf<P> {
        match self {
            LeafEffect::Intact(leaf) => leaf,
            _ => panic!("is not LeafEffect::Intact"),
        }
    }

    #[allow(dead_code)]
    fn take_split(self) -> (Leaf<P>, Leaf<P>) {
        match self {
            LeafEffect::Split { left, right } => (left, right),
            _ => panic!("is not LeafEffect::Split"),
        }
    }
}

/// Gets the `i`th key in a leaf page buffer.
fn get_key(page: &[u8], i: usize) -> &[u8] {
    let offset = get_offset(page, i);
    let num_keys = node::get_num_keys(page);
    let key_len = u16::from_be_bytes([
        page[4 + num_keys * 2 + offset],
        page[4 + num_keys * 2 + offset + 1],
    ]) as usize;
    &page[4 + num_keys * 2 + offset + 4..4 + num_keys * 2 + offset + 4 + key_len]
}

/// Gets the `i`th value in a leaf page buffer.
fn get_value(page: &[u8], i: usize) -> &[u8] {
    let offset = get_offset(page, i);
    let num_keys = node::get_num_keys(page);
    let key_len = u16::from_be_bytes([
        page[4 + num_keys * 2 + offset],
        page[4 + num_keys * 2 + offset + 1],
    ]) as usize;
    let val_len = u16::from_be_bytes([
        page[4 + num_keys * 2 + offset + 2],
        page[4 + num_keys * 2 + offset + 3],
    ]) as usize;
    &page
        [4 + num_keys * 2 + offset + 4 + key_len..4 + num_keys * 2 + offset + 4 + key_len + val_len]
}

/// Gets the `i`th offset value.
fn get_offset(page: &[u8], i: usize) -> usize {
    if i == 0 {
        return 0;
    }
    u16::from_be_bytes([page[4 + 2 * (i - 1)], page[4 + 2 * i - 1]]) as usize
}

/// Gets the number of bytes consumed by a page.
fn get_num_bytes(page: &[u8]) -> usize {
    let n = node::get_num_keys(page);
    let offset = get_offset(page, n);
    4 + (n * 2) + offset
}

/// Sets the next (i.e. `i+1`th) offset and returns the current offset.
fn set_next_offset(page: &mut [u8], i: usize, key: &[u8], val: &[u8]) -> usize {
    let curr_offset = get_offset(page, i);
    let next_offset = curr_offset + 4 + key.len() + val.len();
    let next_i = i + 1;
    page[4 + 2 * (next_i - 1)..4 + 2 * next_i].copy_from_slice(&(next_offset as u16).to_be_bytes());
    curr_offset
}

fn find_split<'a, F, I>(itr_func: F, num_keys: usize) -> usize
where
    F: Fn() -> I,
    I: Iterator<Item = (&'a [u8], &'a [u8])>,
{
    assert!(num_keys >= 2);

    // Try to split such that both splits are sufficient
    // (i.e. have at least 2 keys).
    if num_keys < 4 {
        // Relax the sufficiency requirement if impossible to meet.
        return itr_func()
            .scan(4usize, |size, (k, v)| {
                *size += 6 + k.len() + v.len();
                if *size > consts::PAGE_SIZE {
                    return None;
                }
                Some(())
            })
            .count();
    }
    itr_func()
        .enumerate()
        .scan(4usize, |size, (i, (k, v))| {
            *size += 6 + k.len() + v.len();
            if i < 2 {
                return Some(());
            }
            if *size > consts::PAGE_SIZE || i >= num_keys - 2 {
                return None;
            }
            Some(())
        })
        .count()
}

fn build_split<'a, P, F, I>(store: P, itr_func: &F, num_keys: usize) -> Result<LeafEffect<P>>
where
    P: PageStore + 'a,
    F: Fn() -> I,
    I: Iterator<Item = (&'a [u8], &'a [u8])>,
{
    let split_at = find_split(itr_func, num_keys);
    let itr = itr_func();
    let (mut lb, mut rb) = (
        Builder::new(split_at, store.clone())?,
        Builder::new(num_keys - split_at, store.clone())?,
    );
    for (i, (k, v)) in itr.enumerate() {
        if i < split_at {
            lb = lb.add_key_value(k, v);
        } else {
            rb = rb.add_key_value(k, v);
        }
    }
    let (left, right) = (lb.build(), rb.build());
    Ok(LeafEffect::Split { left, right })
}

fn build<'a, P, F, I>(store: P, itr_func: F, num_keys: usize) -> Result<LeafEffect<P>>
where
    P: PageStore + 'a,
    F: FnOnce() -> I,
    I: Iterator<Item = (&'a [u8], &'a [u8])>,
{
    let itr = itr_func();
    let mut b = Builder::new(num_keys, store)?;
    for (k, v) in itr {
        b = b.add_key_value(k, v);
    }
    Ok(LeafEffect::Intact(b.build()))
}

// A builder of a B+ tree leaf node.
struct Builder<P: PageStore> {
    i: usize,
    store: P,
    page: P::Page,
}

impl<P: PageStore> Builder<P> {
    /// Creates a new leaf builder.
    fn new(num_keys: usize, store: P) -> Result<Self> {
        let mut page = store.new_page()?;
        node::set_node_type(&mut page, NodeType::Leaf);
        node::set_num_keys(&mut page, num_keys);
        Ok(Self { i: 0, store, page })
    }

    /// Adds a key-value pair to the builder.
    fn add_key_value(mut self, key: &[u8], val: &[u8]) -> Self {
        let n = node::get_num_keys(&self.page);
        assert!(
            self.i < n,
            "add_key_value() called {} times, cannot be called more times than num_keys = {}",
            self.i + 1,
            n
        );
        assert!(key.len() <= consts::MAX_KEY_SIZE);
        assert!(val.len() <= consts::MAX_VALUE_SIZE);

        let offset = set_next_offset(&mut self.page, self.i, key, val);
        let pos = 4 + n * 2 + offset;
        assert!(
            pos + 4 + key.len() + val.len() <= consts::PAGE_SIZE,
            "builder unexpectedly overflowed: i = {}, n = {}",
            self.i,
            n
        );

        self.page[pos..pos + 2].copy_from_slice(&(key.len() as u16).to_be_bytes());
        self.page[pos + 2..pos + 4].copy_from_slice(&(val.len() as u16).to_be_bytes());
        self.page[pos + 4..pos + 4 + key.len()].copy_from_slice(key);
        self.page[pos + 4 + key.len()..pos + 4 + key.len() + val.len()].copy_from_slice(val);

        self.i += 1;
        self
    }

    /// Builds a leaf.
    fn build(self) -> Leaf<P> {
        let n = node::get_num_keys(&self.page);
        assert!(
            self.i == n,
            "build() called after calling add_key_value() {} times < num_keys = {}",
            self.i,
            n
        );
        assert_ne!(n, 0, "This case should be handled by Leaf::delete instead.");
        Leaf {
            page: self.store.write_page(self.page),
            store: self.store.clone(),
        }
    }
}

/// A B+ tree leaf node.
#[derive(Debug)]
pub struct Leaf<P: PageStore> {
    page: P::ReadOnlyPage,
    store: P,
}

impl<P: PageStore> Leaf<P> {
    pub fn new(store: P) -> Result<Self> {
        let mut page = store.new_page()?;
        node::set_node_type(&mut page, NodeType::Leaf);
        Ok(Self {
            page: store.write_page(page),
            store,
        })
    }

    pub fn from_page(store: P, page: P::ReadOnlyPage) -> Self {
        Leaf { page, store }
    }

    pub fn page_num(&self) -> usize {
        self.page.page_num()
    }

    /// Inserts a key-value pair.
    pub fn insert(&self, key: &[u8], val: &[u8]) -> Result<LeafEffect<P>> {
        if key.len() > consts::MAX_KEY_SIZE {
            return Err(NodeError::MaxKeySize(key.len()));
        }
        if val.len() > consts::MAX_VALUE_SIZE {
            return Err(NodeError::MaxValueSize(val.len()));
        }
        if self.find(key).is_some() {
            return Err(NodeError::AlreadyExists);
        }
        let itr_func = || self.insert_iter(key, val);
        if self.get_num_bytes() + 6 + key.len() + val.len() > consts::PAGE_SIZE {
            return build_split(self.store.clone(), &itr_func, self.get_num_keys() + 1);
        }
        build(self.store.clone(), itr_func, self.get_num_keys() + 1)
    }

    /// Updates the value corresponding to a key.
    pub fn update(&self, key: &[u8], val: &[u8]) -> Result<LeafEffect<P>> {
        if key.len() > consts::MAX_KEY_SIZE {
            return Err(NodeError::MaxKeySize(key.len()));
        }
        if val.len() > consts::MAX_VALUE_SIZE {
            return Err(NodeError::MaxValueSize(val.len()));
        }
        let old_val = self.find(key);
        if old_val.is_none() {
            return Err(NodeError::KeyNotFound);
        }
        let old_val = old_val.unwrap();
        let itr_func = || self.update_iter(key, val);
        if self.get_num_bytes() - old_val.len() + val.len() > consts::PAGE_SIZE {
            return build_split(self.store.clone(), &itr_func, self.get_num_keys());
        }
        build(self.store.clone(), itr_func, self.get_num_keys())
    }

    /// Deletes a key and its corresponding value.
    pub fn delete(&self, key: &[u8]) -> Result<LeafEffect<P>> {
        if key.len() > consts::MAX_KEY_SIZE {
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
        let mut b = Builder::new(n - 1, self.store.clone())?;
        let mut added = false;
        for (k, v) in self.iter() {
            if !added && key == k {
                added = true;
                continue;
            }
            b = b.add_key_value(k, v);
        }
        Ok(LeafEffect::Intact(b.build()))
    }

    /// Finds the value corresponding to the queried key.
    pub fn find(&self, key: &[u8]) -> Option<&[u8]> {
        self.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    pub fn steal_or_merge(left: &Leaf<P>, right: &Leaf<P>) -> Result<LeafEffect<P>> {
        let itr_func = || left.iter().chain(right.iter());
        let total_num_keys = left.get_num_keys() + right.get_num_keys();
        let overflow = left.get_num_bytes() + right.get_num_bytes() - 4 > consts::PAGE_SIZE;
        if overflow {
            // Steal
            return build_split(left.store.clone(), &itr_func, total_num_keys);
        }
        // Merge
        build(left.store.clone(), itr_func, total_num_keys)
    }

    pub fn get_key(&self, i: usize) -> &[u8] {
        get_key(&self.page, i)
    }

    pub fn get_value(&self, i: usize) -> &[u8] {
        get_value(&self.page, i)
    }

    pub fn get_num_keys(&self) -> usize {
        node::get_num_keys(&self.page)
    }

    pub fn iter(&self) -> LeafIterator<P> {
        LeafIterator {
            node: self,
            i: 0,
            n: self.get_num_keys(),
        }
    }

    fn insert_iter<'a>(&'a self, key: &'a [u8], val: &'a [u8]) -> InsertIterator<'a, P> {
        InsertIterator {
            leaf_itr: self.iter().peekable(),
            key,
            val,
            added: false,
        }
    }

    fn update_iter<'a>(&'a self, key: &'a [u8], val: &'a [u8]) -> UpdateIterator<'a, P> {
        UpdateIterator {
            leaf_itr: self.iter().peekable(),
            key,
            val,
            skip: false,
        }
    }

    pub fn get_num_bytes(&self) -> usize {
        get_num_bytes(&self.page)
    }
}

/// An key-value iterator for a leaf node.
pub struct LeafIterator<'a, P: PageStore> {
    node: &'a Leaf<P>,
    i: usize,
    n: usize,
}

impl<'a, P: PageStore> Iterator for LeafIterator<'a, P> {
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

struct InsertIterator<'a, P: PageStore> {
    leaf_itr: Peekable<LeafIterator<'a, P>>,
    key: &'a [u8],
    val: &'a [u8],
    added: bool,
}

impl<'a, P: PageStore> Iterator for InsertIterator<'a, P> {
    type Item = (&'a [u8], &'a [u8]);
    fn next(&mut self) -> Option<Self::Item> {
        if self.added {
            return self.leaf_itr.next();
        }
        match self.leaf_itr.peek() {
            None => {
                self.added = true;
                Some((self.key, self.val))
            }
            Some(&(leaf_key, _)) => {
                if self.key < leaf_key {
                    self.added = true;
                    Some((self.key, self.val))
                } else {
                    self.leaf_itr.next()
                }
            }
        }
    }
}

struct UpdateIterator<'a, P: PageStore> {
    leaf_itr: Peekable<LeafIterator<'a, P>>,
    key: &'a [u8],
    val: &'a [u8],
    skip: bool,
}

impl<'a, P: PageStore> Iterator for UpdateIterator<'a, P> {
    type Item = (&'a [u8], &'a [u8]);
    fn next(&mut self) -> Option<Self::Item> {
        match self.leaf_itr.peek() {
            None => None,
            Some(&(leaf_key, _)) => {
                if self.skip {
                    self.skip = false;
                    self.leaf_itr.next();
                    return self.leaf_itr.next();
                }
                if self.key == leaf_key {
                    self.skip = true;
                    return Some((self.key, self.val));
                }
                self.leaf_itr.next()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tree::page_store::InMemory;

    use super::*;

    fn test_store() -> InMemory {
        InMemory::new()
    }

    #[test]
    fn insert_intact() {
        let leaf = Leaf::new(test_store())
            .unwrap()
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
        let key = &[0u8; consts::MAX_KEY_SIZE + 1];
        let result = Leaf::new(test_store())
            .unwrap()
            .insert(key, "val".as_bytes());
        assert!(matches!(result, Err(NodeError::MaxKeySize(x)) if x == consts::MAX_KEY_SIZE + 1));
    }

    #[test]
    fn insert_max_value_size() {
        let val = &[0u8; consts::MAX_VALUE_SIZE + 1];
        let result = Leaf::new(test_store())
            .unwrap()
            .insert("key".as_bytes(), val);
        assert!(
            matches!(result, Err(NodeError::MaxValueSize(x)) if x == consts::MAX_VALUE_SIZE + 1)
        );
    }

    #[test]
    fn insert_split() {
        // Insert 1 huge key-value.
        let key1 = &[1u8; consts::MAX_KEY_SIZE];
        let val1 = &[1u8; consts::MAX_VALUE_SIZE];
        let result = Leaf::new(test_store()).unwrap().insert(key1, val1);
        assert!(matches!(result, Ok(LeafEffect::Intact(_))));
        let leaf = result.unwrap().take_intact();

        // Insert another huge key-value to trigger splitting.
        let key0 = &[0u8; consts::MAX_KEY_SIZE];
        let val0 = &[0u8; consts::MAX_VALUE_SIZE];
        let result = leaf.insert(key0, val0);
        assert!(matches!(result, Ok(LeafEffect::Split { .. })),);
        let (left, right) = result.unwrap().take_split();
        drop(leaf);
        assert_eq!(left.get_num_keys(), 1);
        assert_eq!(right.get_num_keys(), 1);
        assert_eq!(left.find(key0).unwrap(), val0);
        assert_eq!(right.find(key1).unwrap(), val1);
    }

    #[test]
    fn find_some() {
        let leaf = Builder::new(1, test_store())
            .unwrap()
            .add_key_value("key".as_bytes(), "val".as_bytes())
            .build();
        assert!(matches!(leaf.find("key".as_bytes()), Some(v) if v == "val".as_bytes()));
    }

    #[test]
    fn find_none() {
        let leaf = Leaf::new(test_store()).unwrap();
        assert!(leaf.find("key".as_bytes()).is_none())
    }

    #[test]
    fn iter() {
        let leaf = Builder::new(2, test_store())
            .unwrap()
            .add_key_value("key1".as_bytes(), "val1".as_bytes())
            .add_key_value("key2".as_bytes(), "val2".as_bytes())
            .build();
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
        let leaf = Leaf::new(test_store()).unwrap();
        assert_eq!(leaf.iter().count(), 0);
    }

    #[test]
    fn update_intact() {
        let leaf = Builder::new(2, test_store())
            .unwrap()
            .add_key_value("key1".as_bytes(), "val1".as_bytes())
            .add_key_value("key2".as_bytes(), "val2".as_bytes())
            .build();

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
        let leaf = Builder::new(2, test_store())
            .unwrap()
            .add_key_value(&[0u8; consts::MAX_KEY_SIZE], &[0u8; consts::MAX_VALUE_SIZE])
            .add_key_value("1".as_bytes(), "1".as_bytes())
            .build();

        // Update with a huge value to trigger splitting.
        let (left, right) = leaf
            .update("1".as_bytes(), &[1u8; consts::MAX_VALUE_SIZE])
            .unwrap()
            .take_split();
        drop(leaf);
        assert_eq!(left.get_num_keys(), 1);
        assert_eq!(right.get_num_keys(), 1);
        assert_eq!(
            left.find(&[0u8; consts::MAX_KEY_SIZE]).unwrap(),
            &[0u8; consts::MAX_VALUE_SIZE]
        );
        assert_eq!(
            right.find("1".as_bytes()).unwrap(),
            &[1u8; consts::MAX_VALUE_SIZE]
        );
    }

    #[test]
    fn update_max_key_size() {
        let key = &[0u8; consts::MAX_KEY_SIZE + 1];
        let result = Leaf::new(test_store())
            .unwrap()
            .update(key, "val".as_bytes());
        assert!(matches!(result, Err(NodeError::MaxKeySize(x)) if x == consts::MAX_KEY_SIZE + 1));
    }

    #[test]
    fn update_max_value_size() {
        let leaf = Builder::new(1, test_store())
            .unwrap()
            .add_key_value("key".as_bytes(), "val".as_bytes())
            .build();
        let val = &[0u8; consts::MAX_VALUE_SIZE + 1];
        let result = leaf.update("key".as_bytes(), val);
        assert!(
            matches!(result, Err(NodeError::MaxValueSize(x)) if x == consts::MAX_VALUE_SIZE + 1)
        );
    }

    #[test]
    fn update_non_existent() {
        let result = Leaf::new(test_store())
            .unwrap()
            .update("key".as_bytes(), "val".as_bytes());
        assert!(matches!(result, Err(NodeError::KeyNotFound)));
    }

    #[test]
    fn delete_intact() {
        let leaf = Builder::new(2, test_store())
            .unwrap()
            .add_key_value("key1".as_bytes(), "val1".as_bytes())
            .add_key_value("key2".as_bytes(), "val2".as_bytes())
            .build();

        let leaf = leaf.delete("key1".as_bytes()).unwrap().take_intact();

        assert_eq!(
            leaf.iter().collect::<Vec<_>>(),
            vec![("key2".as_bytes(), "val2".as_bytes())]
        );
        assert!(leaf.find("key1".as_bytes()).is_none());
    }

    #[test]
    fn delete_empty() {
        let leaf = Builder::new(1, test_store())
            .unwrap()
            .add_key_value("key".as_bytes(), "val".as_bytes())
            .build();
        let effect = leaf.delete("key".as_bytes()).unwrap();
        assert!(matches!(effect, LeafEffect::Empty));
    }

    #[test]
    fn delete_non_existent() {
        let result = Leaf::new(test_store()).unwrap().delete("key".as_bytes());
        assert!(matches!(result, Err(NodeError::KeyNotFound)));
    }

    #[test]
    fn steal_or_merge_steal() {
        let left = Builder::new(1, test_store())
            .unwrap()
            .add_key_value(&[1; consts::MAX_KEY_SIZE], &[1; consts::MAX_VALUE_SIZE])
            .build();

        let right = Builder::new(3, test_store())
            .unwrap()
            .add_key_value(&[2], &[2])
            .add_key_value(&[3], &[3])
            .add_key_value(&[4; consts::MAX_KEY_SIZE], &[4; consts::MAX_VALUE_SIZE])
            .build();

        let (left, right) = Leaf::steal_or_merge(&left, &right).unwrap().take_split();
        assert!(left.get_num_keys() >= 2);
        assert!(right.get_num_keys() >= 2);
        assert!(right.get_num_keys() < 3);
        let chained = left.iter().chain(right.iter()).collect::<Vec<_>>();
        assert_eq!(
            chained,
            vec![
                (
                    &[1; consts::MAX_KEY_SIZE][..],
                    &[1; consts::MAX_VALUE_SIZE][..]
                ),
                (&[2], &[2]),
                (&[3], &[3]),
                (&[4; consts::MAX_KEY_SIZE], &[4; consts::MAX_VALUE_SIZE]),
            ]
        );
    }

    #[test]
    fn steal_or_merge_merge() {
        let left = Builder::new(1, test_store())
            .unwrap()
            .add_key_value(&[1], &[1])
            .build();

        let right = Builder::new(2, test_store())
            .unwrap()
            .add_key_value(&[2], &[2])
            .add_key_value(&[3], &[3])
            .build();

        let merged = Leaf::steal_or_merge(&left, &right).unwrap().take_intact();
        assert_eq!(
            merged.iter().collect::<Vec<_>>(),
            vec![(&[1][..], &[1][..]), (&[2], &[2]), (&[3], &[3]),]
        );
    }
}
