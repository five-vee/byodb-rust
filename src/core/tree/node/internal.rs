//! An [`Internal`] node is a non-leaf node that points to multiple
//! [`crate::core::tree::node::leaf::Leaf`] children nodes.
//!
//! The internal node will always have at least 2 keys, and therefore at least
//! 2 children.
use std::{iter::Peekable, marker::PhantomData, rc::Rc};

use crate::core::{
    consts,
    error::NodeError,
    mmap::{Guard, ImmutablePage, Writer, WriterPage},
};

use super::header::{self, NodeType};

#[allow(dead_code)]
type Result<T> = std::result::Result<T, NodeError>;

/// A B+ tree internal node.
pub struct Internal<'g, P: ImmutablePage<'g>> {
    _phantom: PhantomData<&'g ()>,
    page: P,
    num_keys: usize,
}

impl<'g, P: ImmutablePage<'g>> Internal<'g, P> {
    pub fn read<G: Guard<'g, P>>(guard: &'g G, page_num: usize) -> Internal<'g, P> {
        let page = unsafe { guard.read_page(page_num) };
        let node_type = header::get_node_type(page.deref()).unwrap();
        let num_keys = header::get_num_keys(&page);
        assert_eq!(node_type, NodeType::Internal);
        Internal {
            _phantom: PhantomData,
            page,
            num_keys,
        }
    }

    /// Finds the index of the child that contains the key.
    pub fn find(&self, key: &[u8]) -> usize {
        let n = self.get_num_keys();
        assert_ne!(n, 0);
        (1..n).rev().find(|&i| self.get_key(i) <= key).unwrap_or(0)
    }

    /// Gets the child pointer at an index.
    #[inline]
    pub fn get_child_pointer(&self, i: usize) -> usize {
        get_child_pointer(&self.page, i)
    }

    /// Gets the number of keys.
    #[inline]
    pub fn get_num_keys(&self) -> usize {
        self.num_keys
    }

    /// Gets the `i`th key in the internal buffer.
    #[inline]
    pub fn get_key(&self, i: usize) -> &'g [u8] {
        get_key(&self.page, i, self.num_keys)
    }

    #[inline]
    pub fn get_num_bytes(&self) -> usize {
        get_num_bytes(&self.page, self.num_keys)
    }

    /// Gets the page number associated to the internal node.
    #[inline]
    pub fn page_num(&self) -> usize {
        self.page.page_num()
    }

    /// Creates a child-entry iterator for the internal node.
    pub fn iter(&self) -> InternalIterator<'_, 'g, P> {
        InternalIterator {
            node: self,
            i: 0,
            n: self.get_num_keys(),
        }
    }
}

impl<'w, 's> Internal<'w, WriterPage<'w, 's>> {
    /// Creates an internal node that is the parent of two splits.
    pub fn parent_of_split(
        writer: &'w Writer<'s>,
        keys: [&[u8]; 2],
        child_pointers: [usize; 2],
    ) -> Self {
        Builder::new(writer, 2)
            .add_child_entry(keys[0], child_pointers[0])
            .add_child_entry(keys[1], child_pointers[1])
            .build()
    }

    /// Merges child entries into the internal node.
    pub fn merge_child_entries(
        self,
        writer: &'w Writer<'s>,
        entries: &[ChildEntry],
    ) -> InternalEffect<'w, 's> {
        let delta_keys = entries
            .iter()
            .map(|ce| match ce {
                ChildEntry::Insert { .. } => 1,
                ChildEntry::Delete { .. } => -1,
                _ => 0,
            })
            .sum::<isize>();
        let delta_size = entries
            .iter()
            .map(|ce| match ce {
                ChildEntry::Insert { key, .. } => key.len() as isize + 10,
                ChildEntry::Update { i, key, .. } => {
                    key.len() as isize - self.get_key(*i).len() as isize
                }
                ChildEntry::Delete { i } => -10 - (self.get_key(*i).len() as isize),
            })
            .sum::<isize>();
        let itr_func = || self.merge_iter(entries);
        let num_keys = (self.get_num_keys() as isize + delta_keys) as usize;
        let overflow = (self.get_num_bytes() as isize + delta_size) as usize > consts::PAGE_SIZE;
        self.build_then_free(writer, itr_func, num_keys, overflow)
    }

    /// Resolves underflow of either `left` or `right` by either having one
    /// steal from the other, or merging the two.
    pub fn steal_or_merge(
        left: Internal<'w, WriterPage<'w, 's>>,
        right: Internal<'w, WriterPage<'w, 's>>,
        writer: &'w Writer<'s>,
    ) -> InternalEffect<'w, 's> {
        let itr_func = || left.iter().chain(right.iter());
        let num_keys = left.get_num_keys() + right.get_num_keys();
        let effect = if left.get_num_bytes() + right.get_num_bytes() - 4 > consts::PAGE_SIZE {
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

    /// Creates a child-entry iterator for the internal node merged with
    /// the specified child entries.
    fn merge_iter<'i>(
        &'i self,
        entries: &'w [ChildEntry],
    ) -> MergeIterator<
        'w,
        InternalIterator<'i, 'w, WriterPage<'w, 's>>,
        std::slice::Iter<'w, ChildEntry>,
    > {
        MergeIterator {
            node_iter: self.iter().enumerate().peekable(),
            entries_iter: entries.iter().peekable(),
        }
    }

    /// Builds an [`InternalEffect`], then frees self back to the store.
    fn build_then_free<'i, I, F>(
        &'i self,
        writer: &'w Writer<'s>,
        itr_func: F,
        num_keys: usize,
        overflow: bool,
    ) -> InternalEffect<'w, 's>
    where
        I: Iterator<Item = (&'i [u8], usize)>,
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

/// An enum representing the effect of an internal node operation.
pub enum InternalEffect<'w, 's> {
    /// A newly created internal node that remained  "intact",
    /// i.e. it did not split.
    Intact(Internal<'w, WriterPage<'w, 's>>),
    /// The left and right splits of an internal node that was created.
    Split {
        left: Internal<'w, WriterPage<'w, 's>>,
        right: Internal<'w, WriterPage<'w, 's>>,
    },
}

impl<'w, 's> InternalEffect<'w, 's> {
    #[allow(dead_code)]
    fn take_intact(self) -> Internal<'w, WriterPage<'w, 's>> {
        match self {
            InternalEffect::Intact(internal) => internal,
            _ => panic!("is not InternalEffect::Intact"),
        }
    }

    #[allow(dead_code)]
    fn take_split(
        self,
    ) -> (
        Internal<'w, WriterPage<'w, 's>>,
        Internal<'w, WriterPage<'w, 's>>,
    ) {
        match self {
            InternalEffect::Split { left, right } => (left, right),
            _ => panic!("is not InternalEffect::Split"),
        }
    }
}

/// A child entry to insert, update, or delete in an internal node.
#[derive(Debug)]
pub enum ChildEntry {
    Insert {
        key: Rc<[u8]>,
        page_num: usize,
    },
    Update {
        i: usize,
        key: Rc<[u8]>,
        page_num: usize,
    },
    Delete {
        i: usize,
    },
}

// A builder of a B+ tree internal node.
struct Builder<'w, 's> {
    i: usize,
    n: usize,
    page: WriterPage<'w, 's>,
}

impl<'w, 's> Builder<'w, 's> {
    /// Creates a new internal node builder.
    fn new(writer: &'w Writer<'s>, num_keys: usize) -> Self {
        let mut page = writer.new_page();
        header::set_node_type(&mut page, NodeType::Internal);
        header::set_num_keys(&mut page, num_keys);
        Self {
            i: 0,
            n: num_keys,
            page,
        }
    }

    /// Adds a child entry to the builder.
    fn add_child_entry(mut self, key: &[u8], page_num: usize) -> Self {
        debug_assert!(
            self.i < self.n,
            "add_child_entry() called {} times, cannot be called more times than num_keys = {}",
            self.i,
            self.n
        );
        debug_assert!(key.len() <= consts::MAX_KEY_SIZE);

        let offset = set_next_offset(&mut self.page, self.i, self.n, key);
        set_child_pointer(&mut self.page, self.i, page_num);
        let pos = 4 + self.n * 10 + offset;
        debug_assert!(
            pos + key.len() <= consts::PAGE_SIZE,
            "builder unexpectedly overflowed: i = {}, n = {}",
            self.i,
            self.n,
        );

        self.page[pos..pos + key.len()].copy_from_slice(key);

        self.i += 1;
        self
    }

    /// Builds an internal node.
    fn build(self) -> Internal<'w, WriterPage<'w, 's>> {
        debug_assert!(
            self.i == self.n,
            "build() called after calling add_child_entry() {} times < num_keys = {}",
            self.i,
            self.n
        );
        Internal {
            _phantom: PhantomData,
            page: self.page.read_only(),
            num_keys: self.n,
        }
    }
}

/// Finds the split point of an overflow internal node that is accessed via
/// an iterator of child entries.
fn find_split<'i, I>(itr: I, num_keys: usize) -> usize
where
    I: Iterator<Item = (&'i [u8], usize)>,
{
    debug_assert!(num_keys >= 4);

    // Try to split such that both splits are sufficient
    // (i.e. have at least 2 keys).
    itr.enumerate()
        .scan(4usize, |size, (i, (k, _))| {
            *size += 10 + k.len();
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

/// Builds a new internal node from the provided iterator of child entries.
fn build<'i, 'w, 's, I>(writer: &'w Writer<'s>, itr: I, num_keys: usize) -> InternalEffect<'w, 's>
where
    I: Iterator<Item = (&'i [u8], usize)>,
{
    let mut b = Builder::new(writer, num_keys);
    for (k, pn) in itr {
        b = b.add_child_entry(k, pn);
    }
    InternalEffect::Intact(b.build())
}

/// Builds two internal nodes by finding the split point from the provided iterator of
/// child entries.
fn build_split<'i, 'w, 's, I, F>(
    writer: &'w Writer<'s>,
    itr_func: &F,
    num_keys: usize,
) -> InternalEffect<'w, 's>
where
    I: Iterator<Item = (&'i [u8], usize)>,
    F: Fn() -> I,
{
    let split_at = find_split(itr_func(), num_keys);
    let itr = itr_func();
    let (mut lb, mut rb) = (
        Builder::new(writer, split_at),
        Builder::new(writer, num_keys - split_at),
    );
    for (i, (k, pn)) in itr.enumerate() {
        if i < split_at {
            lb = lb.add_child_entry(k, pn);
        } else {
            rb = rb.add_child_entry(k, pn);
        }
    }
    let (left, right) = (lb.build(), rb.build());
    InternalEffect::Split { left, right }
}

/// A child-entry iterator of an internal node.
pub struct InternalIterator<'i, 'g, P: ImmutablePage<'g>> {
    node: &'i Internal<'g, P>,
    i: usize,
    n: usize,
}

impl<'g, P: ImmutablePage<'g>> Iterator for InternalIterator<'_, 'g, P> {
    type Item = (&'g [u8], usize);

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

/// A child-entry iterator of an internal node
/// merged with new child entries.
struct MergeIterator<'a, I, E>
where
    I: Iterator<Item = (&'a [u8], usize)>,
    E: Iterator<Item = &'a ChildEntry>,
{
    node_iter: Peekable<std::iter::Enumerate<I>>,
    entries_iter: Peekable<E>,
}

impl<'a, I, E> Iterator for MergeIterator<'a, I, E>
where
    I: Iterator<Item = (&'a [u8], usize)>,
    E: Iterator<Item = &'a ChildEntry>,
{
    type Item = (&'a [u8], usize);

    fn next(&mut self) -> Option<Self::Item> {
        match (self.node_iter.peek(), self.entries_iter.peek()) {
            (None, None) => None,
            (Some(&(_, (k, pn))), None) => {
                self.node_iter.next();
                Some((k, pn))
            }
            (None, Some(&ce)) => match ce {
                ChildEntry::Insert { key, page_num } => {
                    self.entries_iter.next();
                    Some((key.as_ref(), *page_num))
                }
                _ => panic!("ChildEntry {ce:?} cannot be applied without corresponding index"),
            },
            (Some(&(i, (k, pn))), Some(ce)) => match ce {
                ChildEntry::Update {
                    i: j,
                    key,
                    page_num,
                } if i == *j => {
                    self.node_iter.next();
                    self.entries_iter.next();
                    Some((key.as_ref(), *page_num))
                }
                ChildEntry::Delete { i: j } if i == *j => {
                    self.node_iter.next();
                    self.entries_iter.next();
                    self.next()
                }
                ChildEntry::Insert { key, page_num } if key.as_ref() < k => {
                    self.entries_iter.next();
                    Some((key.as_ref(), *page_num))
                }
                ce @ &ChildEntry::Insert { key, .. } if key.as_ref() == k => {
                    panic!("ChildEntry {ce:?} has a duplicate key");
                }
                _ => {
                    self.node_iter.next();
                    Some((k, pn))
                }
            },
        }
    }
}

/// Gets the `i`th key in an internal node's page buffer.
fn get_key<'g, P: ImmutablePage<'g>>(page: &P, i: usize, n: usize) -> &'g [u8] {
    let offset = get_offset(page, i, n);
    let key_len = get_offset(page, i + 1, n) - offset;
    let key = &page[4 + n * 10 + offset..4 + n * 10 + offset + key_len];
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

/// Gets the `i`th child pointer in an internal node's page buffer.
fn get_child_pointer<'g, P: ImmutablePage<'g>>(page: &P, i: usize) -> usize {
    u64::from_le_bytes([
        page[4 + i * 8],
        page[4 + i * 8 + 1],
        page[4 + i * 8 + 2],
        page[4 + i * 8 + 3],
        page[4 + i * 8 + 4],
        page[4 + i * 8 + 5],
        page[4 + i * 8 + 6],
        page[4 + i * 8 + 7],
    ]) as usize
}

/// Sets the `i`th child pointer in an internal node's page buffer.
#[inline]
fn set_child_pointer(page: &mut [u8], i: usize, page_num: usize) {
    page[4 + i * 8..4 + (i + 1) * 8].copy_from_slice(&(page_num as u64).to_le_bytes());
}

/// Gets the `i`th offset value.
#[inline]
fn get_offset(page: &[u8], i: usize, n: usize) -> usize {
    if i == 0 {
        return 0;
    }
    u16::from_le_bytes([page[4 + n * 8 + 2 * (i - 1)], page[4 + n * 8 + 2 * i - 1]]) as usize
}

/// Sets the next (i.e. `i+1`th) offset and returns the current offset.
fn set_next_offset(page: &mut [u8], i: usize, n: usize, key: &[u8]) -> usize {
    let curr_offset = get_offset(page, i, n);
    let next_offset = curr_offset + key.len();
    let next_i = i + 1;
    page[4 + n * 8 + 2 * (next_i - 1)..4 + n * 8 + 2 * next_i]
        .copy_from_slice(&(next_offset as u16).to_le_bytes());
    curr_offset
}

/// Gets the number of bytes consumed by a page.
fn get_num_bytes(page: &[u8], n: usize) -> usize {
    let offset = get_offset(page, n, n);
    4 + (n * 10) + offset
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use seize::Collector;
    use tempfile::NamedTempFile;

    use crate::core::mmap::{DEFAULT_MIN_FILE_GROWTH_SIZE, Mmap, Store};

    use super::*;

    fn new_test_store() -> (Arc<Store>, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        println!("Created temporary file {path:?}");
        let mmap = Mmap::open_or_create(path, DEFAULT_MIN_FILE_GROWTH_SIZE).unwrap();

        // Use batch size of 1 to trigger garbage collection ASAP.
        let collector = Collector::new().batch_size(1);

        let store = Arc::new(Store::new(mmap, collector));
        (store, temp_file)
    }

    #[test]
    fn test_parent_of_split() {
        let (store, _temp_file) = new_test_store();
        let writer = store.writer();
        let parent =
            Internal::parent_of_split(&writer, ["key1".as_bytes(), "key2".as_bytes()], [111, 222]);
        assert_eq!(
            parent.iter().collect::<Vec<_>>(),
            vec![("key1".as_bytes(), 111), ("key2".as_bytes(), 222)]
        );
    }

    #[test]
    fn test_merge_child_entries_intact() {
        let (store, _temp_file) = new_test_store();
        let writer = store.writer();
        let node =
            Internal::parent_of_split(&writer, ["key1".as_bytes(), "key2".as_bytes()], [111, 222]);
        let node = node
            .merge_child_entries(
                &writer,
                &[
                    ChildEntry::Insert {
                        key: "key0".as_bytes().into(),
                        page_num: 0,
                    },
                    ChildEntry::Update {
                        i: 0,
                        key: "key1_new".as_bytes().into(),
                        page_num: 1111,
                    },
                    ChildEntry::Delete { i: 1 },
                    ChildEntry::Insert {
                        key: "key3".as_bytes().into(),
                        page_num: 333,
                    },
                ],
            )
            .take_intact();
        assert_eq!(
            node.iter().collect::<Vec<_>>(),
            vec![
                ("key0".as_bytes(), 0),
                ("key1_new".as_bytes(), 1111),
                ("key3".as_bytes(), 333)
            ]
        );
    }

    #[test]
    fn test_merge_child_entries_insert_split() {
        let (store, _temp_file) = new_test_store();
        let writer = store.writer();
        let node = Builder::new(&writer, 4)
            .add_child_entry(&[0; consts::MAX_KEY_SIZE], 0)
            .add_child_entry(&[1; consts::MAX_KEY_SIZE], 1)
            .add_child_entry(&[2; consts::MAX_KEY_SIZE], 2)
            .add_child_entry(&[4; consts::MAX_KEY_SIZE], 4)
            .build();
        let (left, right) = node
            .merge_child_entries(
                &writer,
                &[ChildEntry::Insert {
                    key: [3; consts::MAX_KEY_SIZE].into(),
                    page_num: 3,
                }],
            )
            .take_split();
        assert!(left.get_num_keys() >= 2);
        assert!(right.get_num_keys() >= 2);
        let chained = left.iter().chain(right.iter()).collect::<Vec<_>>();
        assert_eq!(
            chained,
            vec![
                (&[0; consts::MAX_KEY_SIZE][..], 0),
                (&[1; consts::MAX_KEY_SIZE], 1),
                (&[2; consts::MAX_KEY_SIZE], 2),
                (&[3; consts::MAX_KEY_SIZE], 3),
                (&[4; consts::MAX_KEY_SIZE], 4),
            ]
        );
    }

    #[test]
    fn test_merge_child_entries_update_split() {
        let (store, _temp_file) = new_test_store();
        let writer = store.writer();
        let node = Builder::new(&writer, 5)
            .add_child_entry(&[0; consts::MAX_KEY_SIZE], 0)
            .add_child_entry(&[1; consts::MAX_KEY_SIZE], 1)
            .add_child_entry(&[2; consts::MAX_KEY_SIZE], 2)
            .add_child_entry(&[3; 1], 3)
            .add_child_entry(&[4; consts::MAX_KEY_SIZE], 4)
            .build();
        let (left, right) = node
            .merge_child_entries(
                &writer,
                &[ChildEntry::Update {
                    i: 3,
                    key: [3; consts::MAX_KEY_SIZE].into(),
                    page_num: 3,
                }],
            )
            .take_split();
        assert!(left.get_num_keys() >= 2);
        assert!(right.get_num_keys() >= 2);
        let chained = left.iter().chain(right.iter()).collect::<Vec<_>>();
        assert_eq!(
            chained,
            vec![
                (&[0; consts::MAX_KEY_SIZE][..], 0),
                (&[1; consts::MAX_KEY_SIZE], 1),
                (&[2; consts::MAX_KEY_SIZE], 2),
                (&[3; consts::MAX_KEY_SIZE], 3),
                (&[4; consts::MAX_KEY_SIZE], 4),
            ]
        );
    }

    #[test]
    fn test_steal_or_merge_steal() {
        let (store, _temp_file) = new_test_store();
        let writer = store.writer();
        let left = Builder::new(&writer, 1)
            .add_child_entry(&[1; consts::MAX_KEY_SIZE], 1)
            .build();
        let right = Builder::new(&writer, 4)
            .add_child_entry(&[2; consts::MAX_KEY_SIZE], 2)
            .add_child_entry(&[3; consts::MAX_KEY_SIZE], 3)
            .add_child_entry(&[4; consts::MAX_KEY_SIZE], 4)
            .add_child_entry(&[5; consts::MAX_KEY_SIZE], 5)
            .build();

        let (left, right) = Internal::steal_or_merge(left, right, &writer).take_split();
        assert!(left.get_num_keys() >= 2);
        assert!(right.get_num_keys() >= 2);
        assert!(right.get_num_keys() < 4);
        let chained = left.iter().chain(right.iter()).collect::<Vec<_>>();
        assert_eq!(
            chained,
            vec![
                (&[1; consts::MAX_KEY_SIZE][..], 1),
                (&[2; consts::MAX_KEY_SIZE], 2),
                (&[3; consts::MAX_KEY_SIZE], 3),
                (&[4; consts::MAX_KEY_SIZE], 4),
                (&[5; consts::MAX_KEY_SIZE], 5),
            ]
        );
    }

    #[test]
    fn test_steal_or_merge_merge() {
        let (store, _temp_file) = new_test_store();
        let writer = store.writer();
        let left = Builder::new(&writer, 1).add_child_entry(&[1], 1).build();
        let right = Builder::new(&writer, 2)
            .add_child_entry(&[2], 2)
            .add_child_entry(&[3], 3)
            .build();

        let merged = Internal::steal_or_merge(left, right, &writer).take_intact();
        assert_eq!(
            merged.iter().collect::<Vec<_>>(),
            vec![(&[1][..], 1), (&[2], 2), (&[3], 3),]
        );
    }

    #[test]
    fn test_find() {
        let (store, _temp_file) = new_test_store();
        let writer = store.writer();
        let node = Builder::new(&writer, 2)
            .add_child_entry(&[1], 1)
            .add_child_entry(&[2], 2)
            .build();
        assert_eq!(node.find(&[1]), 0);
        assert_eq!(node.find(&[3]), 1);

        // This works b/c we'd want to be able to insert &[0] into the 0-th child.
        assert_eq!(node.find(&[0]), 0);
    }
}
