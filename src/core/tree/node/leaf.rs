use std::iter::Peekable;
use std::ops::Deref as _;

use crate::core::consts;
use crate::core::error::NodeError;
use crate::core::mmap::{Page, ReadOnlyPage, Writer};

#[cfg(test)]
use crate::core::mmap::Guard;

use super::header::{self, NodeType};

type Result<T> = std::result::Result<T, NodeError>;

/// A B+ tree leaf node.
pub struct Leaf<'a> {
    page: ReadOnlyPage<'a>,
}

impl<'a> Leaf<'a> {
    /// Inserts a key-value pair.
    pub fn insert(self, writer: &'a Writer, key: &[u8], val: &[u8]) -> Result<LeafEffect<'a>> {
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
        let num_keys = self.get_num_keys() + 1;
        let overflow = self.get_num_bytes() + 6 + key.len() + val.len() > consts::PAGE_SIZE;
        Ok(self.build_then_free(writer, itr_func, num_keys, overflow))
    }

    /// Updates the value corresponding to a key.
    pub fn update(self, writer: &'a Writer, key: &[u8], val: &[u8]) -> Result<LeafEffect<'a>> {
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
        let num_keys = self.get_num_keys();
        let overflow = self.get_num_bytes() - old_val.len() + val.len() > consts::PAGE_SIZE;
        Ok(self.build_then_free(writer, itr_func, num_keys, overflow))
    }

    /// Deletes a key and its corresponding value.
    pub fn delete<'w>(self, writer: &'w Writer, key: &[u8]) -> Result<LeafEffect<'w>> {
        let page_num = self.page_num();
        let effect = self.delete_inner(writer, key);
        if effect.is_ok() {
            // Now that self is no longer used, free it.
            writer.mark_free(page_num);
        }
        effect
    }

    fn delete_inner<'w>(self, writer: &'w Writer, key: &[u8]) -> Result<LeafEffect<'w>> {
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
        let mut b = Builder::new(writer, n - 1);
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
    pub fn find(&self, key: &[u8]) -> Option<&'a [u8]> {
        self.iter().find(|&(k, _)| k == key).map(|(_, v)| v)
    }

    /// Creates the leaf (or leaves) resulting from either left stealing from
    /// right, or left merging with right.
    pub fn steal_or_merge(left: Leaf, right: Leaf, writer: &'a Writer) -> LeafEffect<'a> {
        let itr_func = || left.iter().chain(right.iter());
        let num_keys = left.get_num_keys() + right.get_num_keys();
        let overflow = left.get_num_bytes() + right.get_num_bytes() - 4 > consts::PAGE_SIZE;
        let effect = if overflow {
            // Steal
            build_split(writer, &itr_func, num_keys)
        } else {
            // Merge
            build(writer, itr_func(), num_keys)
        };
        // Now that left and right are no longer used, free them.
        writer.mark_free(left.page_num());
        writer.mark_free(right.page_num());

        effect
    }

    /// Gets the `i`th key.
    pub fn get_key(&self, i: usize) -> &'a [u8] {
        get_key(&self.page, i)
    }

    /// Gets the `i`th value.
    pub fn get_value(&self, i: usize) -> &'a [u8] {
        get_value(&self.page, i)
    }

    /// Gets the number of keys in the leaf.
    pub fn get_num_keys(&self) -> usize {
        header::get_num_keys(&self.page)
    }

    /// Gets the number of bytes taken up by the leaf.
    pub fn get_num_bytes(&self) -> usize {
        get_num_bytes(&self.page)
    }

    /// Gets the page number associated to the leaf node.
    pub fn page_num(&self) -> usize {
        self.page.page_num()
    }

    /// Returns a key-value iterator of the leaf.
    pub fn iter(&self) -> LeafIterator<'_, 'a> {
        LeafIterator {
            node: self,
            i: 0,
            n: self.get_num_keys(),
        }
    }

    /// Returns a key-value insert-iterator of the leaf.
    fn insert_iter<'l>(&'l self, key: &'a [u8], val: &'a [u8]) -> InsertIterator<'l, 'a> {
        InsertIterator {
            leaf_itr: self.iter().peekable(),
            key,
            val,
            added: false,
        }
    }

    /// Returns a key-value update-iterator of the leaf.
    fn update_iter<'l>(&'l self, key: &'a [u8], val: &'a [u8]) -> UpdateIterator<'l, 'a> {
        UpdateIterator {
            leaf_itr: self.iter().peekable(),
            key,
            val,
            skip: false,
        }
    }

    /// Builds an [`LeafEffect`], then frees self back to the store.
    fn build_then_free<'l, I, F>(
        &'l self,
        writer: &'a Writer,
        itr_func: F,
        num_keys: usize,
        overflow: bool,
    ) -> LeafEffect<'a>
    where
        I: Iterator<Item = (&'l [u8], &'l [u8])>,
        F: Fn() -> I,
    {
        let effect = if overflow {
            build_split(writer, &itr_func, num_keys)
        } else {
            build(writer, itr_func(), num_keys)
        };
        writer.mark_free(self.page_num());
        effect
    }
}

#[cfg(test)]
impl<'g> Leaf<'g> {
    /// Reads a page as a leaf node type.
    fn read<G: Guard>(guard: &'g G, page_num: usize) -> Leaf<'g> {
        let page = unsafe { guard.read_page(page_num) };
        let node_type = header::get_node_type(page.as_ref()).unwrap();
        assert_eq!(node_type, NodeType::Leaf);
        Leaf { page }
    }
}

impl<'a> TryFrom<ReadOnlyPage<'a>> for Leaf<'a> {
    type Error = NodeError;
    fn try_from(page: ReadOnlyPage<'a>) -> Result<Self> {
        let node_type = header::get_node_type(page.deref())?;
        if node_type != NodeType::Leaf {
            return Err(NodeError::UnexpectedNodeType(node_type as u16));
        }
        Ok(Leaf { page })
    }
}

/// An enum representing the effect of a leaf node operation.
pub enum LeafEffect<'r> {
    /// A leaf with 0 keys after a delete was performed on it.
    /// This is a special-case of `Underflow` done to avoid unnecessary
    /// page allocations, since empty non-root nodes aren't allowed.
    Empty,
    /// A newly created leaf that remained  "intact", i.e. it did not split.
    Intact(Leaf<'r>),
    /// The left and right splits of a leaf that was created.
    Split { left: Leaf<'r>, right: Leaf<'r> },
}

impl<'r> LeafEffect<'r> {
    #[allow(dead_code)]
    fn take_intact(self) -> Leaf<'r> {
        match self {
            LeafEffect::Intact(leaf) => leaf,
            _ => panic!("is not LeafEffect::Intact"),
        }
    }

    #[allow(dead_code)]
    fn take_split(self) -> (Leaf<'r>, Leaf<'r>) {
        match self {
            LeafEffect::Split { left, right } => (left, right),
            _ => panic!("is not LeafEffect::Split"),
        }
    }
}

// A builder of a B+ tree leaf node.
struct Builder<'w> {
    i: usize,
    page: Page<'w>,
}

impl<'w> Builder<'w> {
    /// Creates a new leaf builder.
    fn new(writer: &'w Writer, num_keys: usize) -> Self {
        let mut page = writer.new_page();
        header::set_node_type(&mut page, NodeType::Leaf);
        header::set_num_keys(&mut page, num_keys);
        Self { i: 0, page }
    }

    /// Adds a key-value pair to the builder.
    fn add_key_value(mut self, key: &[u8], val: &[u8]) -> Self {
        let n = header::get_num_keys(&self.page);
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

        self.page[pos..pos + 2].copy_from_slice(&(key.len() as u16).to_le_bytes());
        self.page[pos + 2..pos + 4].copy_from_slice(&(val.len() as u16).to_le_bytes());
        self.page[pos + 4..pos + 4 + key.len()].copy_from_slice(key);
        self.page[pos + 4 + key.len()..pos + 4 + key.len() + val.len()].copy_from_slice(val);

        self.i += 1;
        self
    }

    /// Builds a leaf.
    fn build(self) -> Leaf<'w> {
        let n = header::get_num_keys(&self.page);
        assert!(
            self.i == n,
            "build() called after calling add_key_value() {} times < num_keys = {}",
            self.i,
            n
        );
        assert_ne!(n, 0, "This case should be handled by Leaf::delete instead.");
        self.page.read_only().try_into().unwrap()
    }
}

/// Finds the split point of an overflow leaf node that is accessed via
/// an iterator of key-value pairs.
fn find_split<'l, I>(itr: I, num_keys: usize) -> usize
where
    I: Iterator<Item = (&'l [u8], &'l [u8])>,
{
    assert!(num_keys >= 2);

    // Try to split such that both splits are sufficient
    // (i.e. have at least 2 keys).
    if num_keys < 4 {
        // Relax the sufficiency requirement if impossible to meet.
        return itr
            .scan(4usize, |size, (k, v)| {
                *size += 6 + k.len() + v.len();
                if *size > consts::PAGE_SIZE {
                    return None;
                }
                Some(())
            })
            .count();
    }
    itr.enumerate()
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

/// Builds a new leaf from the provided iterator of key-value pairs.
fn build<'l, 'w, I>(writer: &'w Writer, itr: I, num_keys: usize) -> LeafEffect<'w>
where
    I: Iterator<Item = (&'l [u8], &'l [u8])>,
{
    let mut b = Builder::new(writer, num_keys);
    for (k, v) in itr {
        b = b.add_key_value(k, v);
    }
    LeafEffect::Intact(b.build())
}

/// Builds two leaves by finding the split point from the provided iterator of
/// key-value pairs.
fn build_split<'l, 'w, I, F>(writer: &'w Writer, itr_func: &F, num_keys: usize) -> LeafEffect<'w>
where
    I: Iterator<Item = (&'l [u8], &'l [u8])>,
    F: Fn() -> I,
{
    let split_at = find_split(itr_func(), num_keys);
    let itr = itr_func();
    let (mut lb, mut rb) = (
        Builder::new(writer, split_at),
        Builder::new(writer, num_keys - split_at),
    );
    for (i, (k, v)) in itr.enumerate() {
        if i < split_at {
            lb = lb.add_key_value(k, v);
        } else {
            rb = rb.add_key_value(k, v);
        }
    }
    let (left, right) = (lb.build(), rb.build());
    LeafEffect::Split { left, right }
}

/// A key-value iterator for a leaf node.
pub struct LeafIterator<'l, 'a> {
    node: &'l Leaf<'a>,
    i: usize,
    n: usize,
}

impl<'a> Iterator for LeafIterator<'_, 'a> {
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

/// A key-value iterator for a leaf node that has inserted a new key-value.
struct InsertIterator<'l, 'a> {
    leaf_itr: Peekable<LeafIterator<'l, 'a>>,
    key: &'a [u8],
    val: &'a [u8],
    added: bool,
}

impl<'a> Iterator for InsertIterator<'_, 'a> {
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

/// A key-value iterator for a leaf node that has updated a key-value.
struct UpdateIterator<'l, 'a> {
    leaf_itr: Peekable<LeafIterator<'l, 'a>>,
    key: &'a [u8],
    val: &'a [u8],
    skip: bool,
}

impl<'a> Iterator for UpdateIterator<'_, 'a> {
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

/// Gets the `i`th key in a leaf page buffer.
fn get_key<'a>(page: &ReadOnlyPage<'a>, i: usize) -> &'a [u8] {
    let offset = get_offset(page, i);
    let num_keys = header::get_num_keys(page);
    let key_len = u16::from_le_bytes([
        page[4 + num_keys * 2 + offset],
        page[4 + num_keys * 2 + offset + 1],
    ]) as usize;
    let key = &page[4 + num_keys * 2 + offset + 4..4 + num_keys * 2 + offset + 4 + key_len];
    // Safety: key borrows from page,
    // which itself borrows from a Reader/Writer (with lifetime 'a),
    // which itself has a reference to a Store,
    // which itself is modeled as a slice of bytes.
    // So long as the reader/writer is alive,
    // the Store cannot be dropped,
    // meaning the underlying slice of bytes cannot either.
    // Thus, casting the borrow lifetime from 'l
    // (the lifetime of leaf node)
    // to 'a (the lifetime of the reader/writer) is safe.
    unsafe { std::slice::from_raw_parts(key.as_ptr(), key.len()) }
}

/// Gets the `i`th value in a leaf page buffer.
fn get_value<'a>(page: &ReadOnlyPage<'a>, i: usize) -> &'a [u8] {
    let offset = get_offset(page, i);
    let num_keys = header::get_num_keys(page);
    let key_len = u16::from_le_bytes([
        page[4 + num_keys * 2 + offset],
        page[4 + num_keys * 2 + offset + 1],
    ]) as usize;
    let val_len = u16::from_le_bytes([
        page[4 + num_keys * 2 + offset + 2],
        page[4 + num_keys * 2 + offset + 3],
    ]) as usize;
    let val = &page[4 + num_keys * 2 + offset + 4 + key_len
        ..4 + num_keys * 2 + offset + 4 + key_len + val_len];
    // Safety: val borrows from page,
    // which itself borrows from a Reader/Writer (with lifetime 'a),
    // which itself has a reference to a Store,
    // which itself is modeled as a slice of bytes.
    // So long as the reader/writer is alive,
    // the Store cannot be dropped,
    // meaning the underlying slice of bytes cannot either.
    // Thus, casting the borrow lifetime from 'l
    // (the lifetime of leaf node)
    // to 'a (the lifetime of the reader/writer) is safe.
    unsafe { std::slice::from_raw_parts(val.as_ptr(), val.len()) }
}

/// Gets the `i`th offset value.
fn get_offset(page: &[u8], i: usize) -> usize {
    if i == 0 {
        return 0;
    }
    u16::from_le_bytes([page[4 + 2 * (i - 1)], page[4 + 2 * i - 1]]) as usize
}

/// Gets the number of bytes consumed by a page.
fn get_num_bytes(page: &[u8]) -> usize {
    let n = header::get_num_keys(page);
    let offset = get_offset(page, n);
    4 + (n * 2) + offset
}

/// Sets the next (i.e. `i+1`th) offset and returns the current offset.
fn set_next_offset(page: &mut [u8], i: usize, key: &[u8], val: &[u8]) -> usize {
    let curr_offset = get_offset(page, i);
    let next_offset = curr_offset + 4 + key.len() + val.len();
    let next_i = i + 1;
    page[4 + 2 * (next_i - 1)..4 + 2 * next_i].copy_from_slice(&(next_offset as u16).to_le_bytes());
    curr_offset
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::NamedTempFile;

    use crate::core::mmap::{Mmap, Store};

    use super::*;

    fn new_test_store() -> (Arc<Store>, NamedTempFile, usize) {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        println!("Created temporary file {path:?}");
        let mmap = Mmap::open_or_create(path).unwrap();
        let store = Arc::new(Store::new(mmap));
        (store, temp_file, 0)
    }

    #[test]
    fn test_insert_intact() {
        let (store, _temp_file, root_ptr) = new_test_store();
        let reader = store.reader();
        let writer = store.writer();

        let leaf = Leaf::read(&reader, root_ptr)
            .insert(&writer, "hello".as_bytes(), "world".as_bytes())
            .unwrap()
            .take_intact();
        assert_eq!(
            leaf.iter().collect::<Vec<_>>(),
            vec![("hello".as_bytes(), "world".as_bytes())]
        );
        assert_eq!(leaf.find("hello".as_bytes()).unwrap(), "world".as_bytes());
    }

    #[test]
    fn test_insert_max_key_size() {
        let (store, _temp_file, root_ptr) = new_test_store();
        let reader = store.reader();
        let writer = store.writer();

        let key = &[0u8; consts::MAX_KEY_SIZE + 1];
        let result = Leaf::read(&reader, root_ptr).insert(&writer, key, "val".as_bytes());
        assert!(matches!(result, Err(NodeError::MaxKeySize(x)) if x == consts::MAX_KEY_SIZE + 1));
    }

    #[test]
    fn test_insert_max_value_size() {
        let (store, _temp_file, root_ptr) = new_test_store();
        let reader = store.reader();
        let writer = store.writer();

        let val = &[0u8; consts::MAX_VALUE_SIZE + 1];
        let result = Leaf::read(&reader, root_ptr).insert(&writer, "key".as_bytes(), val);
        assert!(
            matches!(result, Err(NodeError::MaxValueSize(x)) if x == consts::MAX_VALUE_SIZE + 1)
        );
    }

    #[test]
    fn test_insert_split() {
        let (store, _temp_file, root_ptr) = new_test_store();
        let reader = store.reader();
        let writer = store.writer();

        // Insert 1 huge key-value.
        let key1 = &[1u8; consts::MAX_KEY_SIZE];
        let val1 = &[1u8; consts::MAX_VALUE_SIZE];
        let result = Leaf::read(&reader, root_ptr).insert(&writer, key1, val1);
        assert!(matches!(result, Ok(LeafEffect::Intact(_))));
        let leaf = result.unwrap().take_intact();

        // Insert another huge key-value to trigger splitting.
        let key0 = &[0u8; consts::MAX_KEY_SIZE];
        let val0 = &[0u8; consts::MAX_VALUE_SIZE];
        let result = leaf.insert(&writer, key0, val0);
        assert!(matches!(result, Ok(LeafEffect::Split { .. })),);
        let (left, right) = result.unwrap().take_split();
        assert_eq!(left.get_num_keys(), 1);
        assert_eq!(right.get_num_keys(), 1);
        assert_eq!(left.find(key0).unwrap(), val0);
        assert_eq!(right.find(key1).unwrap(), val1);
    }

    #[test]
    fn test_find_some() {
        let (store, _temp_file, _root_ptr) = new_test_store();
        let writer = store.writer();

        let leaf = Builder::new(&writer, 1)
            .add_key_value("key".as_bytes(), "val".as_bytes())
            .build();
        assert!(matches!(leaf.find("key".as_bytes()), Some(v) if v == "val".as_bytes()));
    }

    #[test]
    fn test_find_none() {
        let (store, _temp_file, root_ptr) = new_test_store();
        let reader = store.reader();

        let leaf = Leaf::read(&reader, root_ptr);
        assert!(leaf.find("key".as_bytes()).is_none())
    }

    #[test]
    fn test_iter() {
        let (store, _temp_file, _root_ptr) = new_test_store();
        let writer = store.writer();

        let leaf = Builder::new(&writer, 2)
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
    fn test_iter_empty() {
        let (store, _temp_file, root_ptr) = new_test_store();
        let reader = store.reader();

        let leaf = Leaf::read(&reader, root_ptr);
        assert_eq!(leaf.iter().count(), 0);
    }

    #[test]
    fn test_update_intact() {
        let (store, _temp_file, _root_ptr) = new_test_store();
        let writer = store.writer();

        let leaf = Builder::new(&writer, 2)
            .add_key_value("key1".as_bytes(), "val1".as_bytes())
            .add_key_value("key2".as_bytes(), "val2".as_bytes())
            .build();

        let leaf = leaf
            .update(&writer, "key1".as_bytes(), "val1_new".as_bytes())
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
    fn test_update_split() {
        let (store, _temp_file, _root_ptr) = new_test_store();
        let writer = store.writer();

        let leaf = Builder::new(&writer, 2)
            .add_key_value(&[0u8; consts::MAX_KEY_SIZE], &[0u8; consts::MAX_VALUE_SIZE])
            .add_key_value("1".as_bytes(), "1".as_bytes())
            .build();

        // Update with a huge value to trigger splitting.
        let (left, right) = leaf
            .update(&writer, "1".as_bytes(), &[1u8; consts::MAX_VALUE_SIZE])
            .unwrap()
            .take_split();
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
    fn test_update_max_key_size() {
        let (store, _temp_file, root_ptr) = new_test_store();
        let reader = store.reader();
        let writer = store.writer();

        let key = &[0u8; consts::MAX_KEY_SIZE + 1];
        let result = Leaf::read(&reader, root_ptr).update(&writer, key, "val".as_bytes());
        assert!(matches!(result, Err(NodeError::MaxKeySize(x)) if x == consts::MAX_KEY_SIZE + 1));
    }

    #[test]
    fn test_update_max_value_size() {
        let (store, _temp_file, _root_ptr) = new_test_store();
        let writer = store.writer();

        let leaf = Builder::new(&writer, 1)
            .add_key_value("key".as_bytes(), "val".as_bytes())
            .build();

        let val = &[0u8; consts::MAX_VALUE_SIZE + 1];
        let result = leaf.update(&writer, "key".as_bytes(), val);
        assert!(
            matches!(result, Err(NodeError::MaxValueSize(x)) if x == consts::MAX_VALUE_SIZE + 1)
        );
    }

    #[test]
    fn test_update_non_existent() {
        let (store, _temp_file, root_ptr) = new_test_store();
        let reader = store.reader();
        let writer = store.writer();

        let result =
            Leaf::read(&reader, root_ptr).update(&writer, "key".as_bytes(), "val".as_bytes());
        assert!(matches!(result, Err(NodeError::KeyNotFound)));
    }

    #[test]
    fn test_delete_intact() {
        let (store, _temp_file, _root_ptr) = new_test_store();
        let writer = store.writer();

        let leaf = Builder::new(&writer, 2)
            .add_key_value("key1".as_bytes(), "val1".as_bytes())
            .add_key_value("key2".as_bytes(), "val2".as_bytes())
            .build();

        let leaf = leaf
            .delete(&writer, "key1".as_bytes())
            .unwrap()
            .take_intact();

        assert_eq!(
            leaf.iter().collect::<Vec<_>>(),
            vec![("key2".as_bytes(), "val2".as_bytes())]
        );
        assert!(leaf.find("key1".as_bytes()).is_none());
    }

    #[test]
    fn test_delete_empty() {
        let (store, _temp_file, _root_ptr) = new_test_store();
        let writer = store.writer();

        let leaf = Builder::new(&writer, 1)
            .add_key_value("key".as_bytes(), "val".as_bytes())
            .build();

        let effect = leaf.delete(&writer, "key".as_bytes()).unwrap();
        assert!(matches!(effect, LeafEffect::Empty));
    }

    #[test]
    fn test_delete_non_existent() {
        let (store, _temp_file, root_ptr) = new_test_store();
        let reader = store.reader();
        let writer = store.writer();

        let result = Leaf::read(&reader, root_ptr).delete(&writer, "key".as_bytes());
        assert!(matches!(result, Err(NodeError::KeyNotFound)));
    }

    #[test]
    fn test_steal_or_merge_steal() {
        let (store, _temp_file, _root_ptr) = new_test_store();
        let writer = store.writer();

        let left = Builder::new(&writer, 1)
            .add_key_value(&[1; consts::MAX_KEY_SIZE], &[1; consts::MAX_VALUE_SIZE])
            .build();

        let right = Builder::new(&writer, 3)
            .add_key_value(&[2], &[2])
            .add_key_value(&[3], &[3])
            .add_key_value(&[4; consts::MAX_KEY_SIZE], &[4; consts::MAX_VALUE_SIZE])
            .build();

        let (left, right) = Leaf::steal_or_merge(left, right, &writer).take_split();
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
    fn test_steal_or_merge_merge() {
        let (store, _temp_file, _root_ptr) = new_test_store();
        let writer = store.writer();

        let left = Builder::new(&writer, 1).add_key_value(&[1], &[1]).build();

        let right = Builder::new(&writer, 2)
            .add_key_value(&[2], &[2])
            .add_key_value(&[3], &[3])
            .build();

        let merged = Leaf::steal_or_merge(left, right, &writer).take_intact();
        assert_eq!(
            merged.iter().collect::<Vec<_>>(),
            vec![(&[1][..], &[1][..]), (&[2], &[2]), (&[3], &[3]),]
        );
    }
}
