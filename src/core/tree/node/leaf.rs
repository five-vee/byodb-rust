//! A [`Leaf`] node holds a sub-range of key-value pairs of the B+ tree.
//! All key-values of the tree are stored in one or more leaf nodes.
//!
//! The tree is initialized as an empty leaf node.
use std::iter::Peekable;
use std::marker::PhantomData;

use crate::core::consts;
use crate::core::error::NodeError;
use crate::core::header::{self, NodeType};
use crate::core::mmap::{Guard, ImmutablePage, Writer, WriterPage};

type Result<T> = std::result::Result<T, NodeError>;

/// A B+ tree leaf node.
pub struct Leaf<'g, P: ImmutablePage<'g>> {
    _phantom: PhantomData<&'g ()>,
    page: P,
    num_keys: usize,
}

impl<'g, P: ImmutablePage<'g>> Leaf<'g, P> {
    /// Reads a page as a leaf node type.
    pub fn read<G: Guard<'g, P>>(guard: &'g G, page_num: usize) -> Leaf<'g, P> {
        let page = unsafe { guard.read_page(page_num) };
        let node_type = header::get_node_type(page.as_ref()).unwrap();
        let num_keys = header::get_num_keys(&page);
        debug_assert_eq!(node_type, NodeType::Leaf);
        Leaf {
            _phantom: PhantomData,
            page,
            num_keys,
        }
    }

    /// Gets the value corresponding to the queried key.
    pub fn get(&self, key: &[u8]) -> Option<&'g [u8]> {
        self.iter().find(|&(k, _)| k == key).map(|(_, v)| v)
    }

    /// Gets the `i`th key.
    #[inline]
    pub fn get_key(&self, i: usize) -> &'g [u8] {
        get_key(&self.page, i, self.num_keys)
    }

    /// Gets the `i`th value.
    #[inline]
    pub fn get_value(&self, i: usize) -> &'g [u8] {
        get_value(&self.page, i, self.num_keys)
    }

    /// Gets the number of keys in the leaf.
    #[inline]
    pub fn get_num_keys(&self) -> usize {
        self.num_keys
    }

    /// Gets the number of bytes taken up by the leaf.
    #[inline]
    pub fn get_num_bytes(&self) -> usize {
        get_num_bytes(&self.page, self.num_keys)
    }

    /// Gets the page number associated to the leaf node.
    #[inline]
    pub fn page_num(&self) -> usize {
        self.page.page_num()
    }

    /// Returns a key-value iterator of the leaf.
    pub fn iter(&self) -> LeafIterator<'_, 'g, P> {
        LeafIterator {
            node: self,
            i: 0,
            n: self.get_num_keys(),
        }
    }
}

impl<'w, 's> Leaf<'w, WriterPage<'w, 's>> {
    /// Inserts a key-value pair.
    pub fn insert(
        self,
        writer: &'w Writer<'s>,
        key: &[u8],
        val: &[u8],
    ) -> Result<LeafEffect<'w, 's>> {
        if key.len() > consts::MAX_KEY_SIZE {
            return Err(NodeError::MaxKeySize(key.len()));
        }
        if val.len() > consts::MAX_VALUE_SIZE {
            return Err(NodeError::MaxValueSize(val.len()));
        }
        if self.get(key).is_some() {
            return Err(NodeError::AlreadyExists);
        }
        let itr_func = || self.insert_iter(key, val);
        let num_keys = self.get_num_keys() + 1;
        let overflow = self.get_num_bytes() + 6 + key.len() + val.len() > consts::PAGE_SIZE;
        Ok(self.build_then_free(writer, itr_func, num_keys, overflow))
    }

    /// Updates the value corresponding to a key.
    pub fn update(
        self,
        writer: &'w Writer<'s>,
        key: &[u8],
        val: &[u8],
    ) -> Result<LeafEffect<'w, 's>> {
        if key.len() > consts::MAX_KEY_SIZE {
            return Err(NodeError::MaxKeySize(key.len()));
        }
        if val.len() > consts::MAX_VALUE_SIZE {
            return Err(NodeError::MaxValueSize(val.len()));
        }
        let old_val = self.get(key);
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
    pub fn delete(self, writer: &'w Writer<'s>, key: &[u8]) -> Result<LeafEffect<'w, 's>> {
        let page_num = self.page_num();
        let effect = self.delete_inner(writer, key);
        if effect.is_ok() {
            // Now that self is no longer used, free it.
            writer.mark_free(page_num);
        }
        effect
    }

    fn delete_inner(self, writer: &'w Writer<'s>, key: &[u8]) -> Result<LeafEffect<'w, 's>> {
        if key.len() > consts::MAX_KEY_SIZE {
            return Err(NodeError::MaxKeySize(key.len()));
        }
        if self.get(key).is_none() {
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

    /// Creates the leaf (or leaves) resulting from either `left` stealing from
    /// `right`, or `left` merging with `right`.
    pub fn steal_or_merge(
        left: Leaf<'w, WriterPage<'w, '_>>,
        right: Leaf<'w, WriterPage<'w, '_>>,
        writer: &'w Writer<'s>,
    ) -> LeafEffect<'w, 's> {
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

    /// Returns a key-value insert-iterator of the leaf.
    fn insert_iter<'l>(&'l self, key: &'w [u8], val: &'w [u8]) -> InsertIterator<'l, 'w, 's> {
        InsertIterator {
            leaf_itr: self.iter().peekable(),
            key,
            val,
            added: false,
        }
    }

    /// Returns a key-value update-iterator of the leaf.
    fn update_iter<'l>(&'l self, key: &'w [u8], val: &'w [u8]) -> UpdateIterator<'l, 'w, 's> {
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
        writer: &'w Writer<'s>,
        itr_func: F,
        num_keys: usize,
        overflow: bool,
    ) -> LeafEffect<'w, 's>
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

/// An enum representing the effect of a leaf node operation.
pub enum LeafEffect<'w, 's> {
    /// A leaf with 0 keys after a delete was performed on it.
    /// This is a special-case of [`super::Sufficiency::Underflow`] done to
    /// avoid unnecessary page allocations, since empty non-root nodes aren't
    /// allowed.
    Empty,
    /// A newly created leaf that remained  "intact", i.e. it did not split.
    Intact(Leaf<'w, WriterPage<'w, 's>>),
    /// The `left` and `right` splits of a leaf that was created.
    Split {
        left: Leaf<'w, WriterPage<'w, 's>>,
        right: Leaf<'w, WriterPage<'w, 's>>,
    },
}

impl<'w, 's> LeafEffect<'w, 's> {
    #[allow(dead_code)]
    fn take_intact(self) -> Leaf<'w, WriterPage<'w, 's>> {
        match self {
            LeafEffect::Intact(leaf) => leaf,
            _ => panic!("is not LeafEffect::Intact"),
        }
    }

    #[allow(dead_code)]
    fn take_split(self) -> (Leaf<'w, WriterPage<'w, 's>>, Leaf<'w, WriterPage<'w, 's>>) {
        match self {
            LeafEffect::Split { left, right } => (left, right),
            _ => panic!("is not LeafEffect::Split"),
        }
    }
}

// A builder of a B+ tree leaf node.
struct Builder<'w, 's> {
    i: usize,
    n: usize,
    page: WriterPage<'w, 's>,
}

impl<'w, 's> Builder<'w, 's> {
    /// Creates a new leaf builder.
    fn new(writer: &'w Writer<'s>, num_keys: usize) -> Self {
        let mut page = writer.new_page();
        header::set_node_type(&mut page, NodeType::Leaf);
        header::set_num_keys(&mut page, num_keys);
        Self {
            i: 0,
            n: num_keys,
            page,
        }
    }

    /// Adds a key-value pair to the builder.
    fn add_key_value(mut self, key: &[u8], val: &[u8]) -> Self {
        debug_assert!(
            self.i < self.n,
            "add_key_value() called {} times, cannot be called more times than num_keys = {}",
            self.i + 1,
            self.n
        );
        debug_assert!(key.len() <= consts::MAX_KEY_SIZE);
        debug_assert!(val.len() <= consts::MAX_VALUE_SIZE);

        let offset = set_next_offset(&mut self.page, self.i, key, val);
        let pos = 4 + self.n * 2 + offset;
        debug_assert!(
            pos + 4 + key.len() + val.len() <= consts::PAGE_SIZE,
            "builder unexpectedly overflowed: i = {}, n = {}",
            self.i,
            self.n
        );

        self.page[pos..pos + 2].copy_from_slice(&(key.len() as u16).to_le_bytes());
        self.page[pos + 2..pos + 4].copy_from_slice(&(val.len() as u16).to_le_bytes());
        self.page[pos + 4..pos + 4 + key.len()].copy_from_slice(key);
        self.page[pos + 4 + key.len()..pos + 4 + key.len() + val.len()].copy_from_slice(val);

        self.i += 1;
        self
    }

    /// Builds a leaf.
    fn build(self) -> Leaf<'w, WriterPage<'w, 's>> {
        debug_assert!(
            self.i == self.n,
            "build() called after calling add_key_value() {} times < num_keys = {}",
            self.i,
            self.n
        );
        debug_assert_ne!(
            self.n, 0,
            "This case should be handled by Leaf::delete instead."
        );
        Leaf {
            _phantom: PhantomData,
            page: self.page.read_only(),
            num_keys: self.n,
        }
    }
}

/// Finds the split point of an overflow leaf node that is accessed via
/// an iterator of key-value pairs.
fn find_split<'l, I>(itr: I, num_keys: usize) -> usize
where
    I: Iterator<Item = (&'l [u8], &'l [u8])>,
{
    debug_assert!(num_keys >= 2);

    itr.scan(4usize, |size, (k, v)| {
        *size += 6 + k.len() + v.len();
        if *size > consts::PAGE_SIZE {
            return None;
        }
        Some(())
    })
    .count()
}

/// Builds a new leaf from the provided iterator of key-value pairs.
fn build<'l, 'w, 's, I>(writer: &'w Writer<'s>, itr: I, num_keys: usize) -> LeafEffect<'w, 's>
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
fn build_split<'l, 'w, 's, I, F>(
    writer: &'w Writer<'s>,
    itr_func: &F,
    num_keys: usize,
) -> LeafEffect<'w, 's>
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
pub struct LeafIterator<'l, 'w, P: ImmutablePage<'w>> {
    node: &'l Leaf<'w, P>,
    i: usize,
    n: usize,
}

impl<'w, P: ImmutablePage<'w>> Iterator for LeafIterator<'_, 'w, P> {
    type Item = (&'w [u8], &'w [u8]);

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
struct InsertIterator<'l, 'w, 's> {
    leaf_itr: Peekable<LeafIterator<'l, 'w, WriterPage<'w, 's>>>,
    key: &'w [u8],
    val: &'w [u8],
    added: bool,
}

impl<'w> Iterator for InsertIterator<'_, 'w, '_> {
    type Item = (&'w [u8], &'w [u8]);
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
struct UpdateIterator<'l, 'w, 's> {
    leaf_itr: Peekable<LeafIterator<'l, 'w, WriterPage<'w, 's>>>,
    key: &'w [u8],
    val: &'w [u8],
    skip: bool,
}

impl<'w> Iterator for UpdateIterator<'_, 'w, '_> {
    type Item = (&'w [u8], &'w [u8]);
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
fn get_key<'g, P: ImmutablePage<'g>>(page: &P, i: usize, n: usize) -> &'g [u8] {
    let offset = get_offset(page, i);
    let key_len =
        u16::from_le_bytes([page[4 + n * 2 + offset], page[4 + n * 2 + offset + 1]]) as usize;
    let key = &page[4 + n * 2 + offset + 4..4 + n * 2 + offset + 4 + key_len];
    // Safety: key borrows from page,
    // which itself borrows from a Reader/Writer (with lifetime 'g),
    // which itself has a reference to a Store,
    // which itself is modeled as a slice of bytes.
    // So long as the reader/writer is alive,
    // the Store cannot be dropped,
    // meaning the underlying slice of bytes cannot either.
    // Thus, casting the borrow lifetime from 'l
    // (the lifetime of leaf node)
    // to 'g (the lifetime of the reader/writer) is safe.
    unsafe { std::slice::from_raw_parts(key.as_ptr(), key.len()) }
}

/// Gets the `i`th value in a leaf page buffer.
fn get_value<'g, P: ImmutablePage<'g>>(page: &P, i: usize, n: usize) -> &'g [u8] {
    let offset = get_offset(page, i);
    let key_len =
        u16::from_le_bytes([page[4 + n * 2 + offset], page[4 + n * 2 + offset + 1]]) as usize;
    let val_len =
        u16::from_le_bytes([page[4 + n * 2 + offset + 2], page[4 + n * 2 + offset + 3]]) as usize;
    let val = &page[4 + n * 2 + offset + 4 + key_len..4 + n * 2 + offset + 4 + key_len + val_len];
    // Safety: val borrows from page,
    // which itself borrows from a Reader/Writer (with lifetime 'g),
    // which itself has a reference to a Store,
    // which itself is modeled as a slice of bytes.
    // So long as the reader/writer is alive,
    // the Store cannot be dropped,
    // meaning the underlying slice of bytes cannot either.
    // Thus, casting the borrow lifetime from 'l
    // (the lifetime of leaf node)
    // to 'g (the lifetime of the reader/writer) is safe.
    unsafe { std::slice::from_raw_parts(val.as_ptr(), val.len()) }
}

/// Gets the `i`th offset value.
#[inline]
fn get_offset(page: &[u8], i: usize) -> usize {
    if i == 0 {
        return 0;
    }
    u16::from_le_bytes([page[4 + 2 * (i - 1)], page[4 + 2 * i - 1]]) as usize
}

/// Gets the number of bytes consumed by a page.
fn get_num_bytes(page: &[u8], n: usize) -> usize {
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

    use seize::Collector;
    use tempfile::NamedTempFile;

    use crate::core::mmap::{DEFAULT_MIN_FILE_GROWTH_SIZE, Mmap, Store};

    use super::*;

    fn new_test_store() -> (Arc<Store>, NamedTempFile, usize) {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        println!("Created temporary file {path:?}");
        let mmap = Mmap::open_or_create(path, DEFAULT_MIN_FILE_GROWTH_SIZE).unwrap();

        // Use batch size of 1 to trigger garbage collection ASAP.
        let collector = Collector::new().batch_size(1);

        let store = Arc::new(Store::new(mmap, collector));
        (store, temp_file, 0)
    }

    #[test]
    fn test_insert_intact() {
        let (store, _temp_file, root_ptr) = new_test_store();
        let writer = store.writer();

        let leaf = Leaf::read(&writer, root_ptr)
            .insert(&writer, "hello".as_bytes(), "world".as_bytes())
            .unwrap()
            .take_intact();
        assert_eq!(
            leaf.iter().collect::<Vec<_>>(),
            vec![("hello".as_bytes(), "world".as_bytes())]
        );
        assert_eq!(leaf.get("hello".as_bytes()).unwrap(), "world".as_bytes());
    }

    #[test]
    fn test_insert_max_key_size() {
        let (store, _temp_file, root_ptr) = new_test_store();
        let writer = store.writer();

        let key = &[0u8; consts::MAX_KEY_SIZE + 1];
        let result = Leaf::read(&writer, root_ptr).insert(&writer, key, "val".as_bytes());
        assert!(matches!(result, Err(NodeError::MaxKeySize(x)) if x == consts::MAX_KEY_SIZE + 1));
    }

    #[test]
    fn test_insert_max_value_size() {
        let (store, _temp_file, root_ptr) = new_test_store();
        let writer = store.writer();

        let val = &[0u8; consts::MAX_VALUE_SIZE + 1];
        let result = Leaf::read(&writer, root_ptr).insert(&writer, "key".as_bytes(), val);
        assert!(
            matches!(result, Err(NodeError::MaxValueSize(x)) if x == consts::MAX_VALUE_SIZE + 1)
        );
    }

    #[test]
    fn test_find_some() {
        let (store, _temp_file, _root_ptr) = new_test_store();
        let writer = store.writer();

        let leaf = Builder::new(&writer, 1)
            .add_key_value("key".as_bytes(), "val".as_bytes())
            .build();
        assert!(matches!(leaf.get("key".as_bytes()), Some(v) if v == "val".as_bytes()));
    }

    #[test]
    fn test_find_none() {
        let (store, _temp_file, root_ptr) = new_test_store();
        let reader = store.reader();

        let leaf = Leaf::read(&reader, root_ptr);
        assert!(leaf.get("key".as_bytes()).is_none())
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
        assert_eq!(leaf.get("key1".as_bytes()).unwrap(), "val1_new".as_bytes());
    }

    #[test]
    fn test_update_max_key_size() {
        let (store, _temp_file, root_ptr) = new_test_store();
        let writer = store.writer();

        let key = &[0u8; consts::MAX_KEY_SIZE + 1];
        let result = Leaf::read(&writer, root_ptr).update(&writer, key, "val".as_bytes());
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
        let writer = store.writer();

        let result =
            Leaf::read(&writer, root_ptr).update(&writer, "key".as_bytes(), "val".as_bytes());
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
        assert!(leaf.get("key1".as_bytes()).is_none());
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
        let writer = store.writer();

        let result = Leaf::read(&writer, root_ptr).delete(&writer, "key".as_bytes());
        assert!(matches!(result, Err(NodeError::KeyNotFound)));
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
