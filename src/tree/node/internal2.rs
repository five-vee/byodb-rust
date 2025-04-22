use std::{iter::Peekable, ops::Deref as _, rc::Rc};

use crate::{
    consts,
    error::NodeError,
    mmap::{Page, ReadOnlyPage, Writer},
};

use super::header::{self, NodeType};

type Result<T> = std::result::Result<T, NodeError>;

/// A B+ tree internal node.
pub struct Internal<'a> {
    page: ReadOnlyPage<'a>,
}

impl<'a> Internal<'a> {
    /// Creates an internal node that is the parent of two splits.
    pub fn parent_of_split(
        writer: &'a Writer,
        keys: [&[u8]; 2],
        child_pointers: [usize; 2],
    ) -> Self {
        Builder::new(writer, 2)
            .add_child_entry(keys[0], child_pointers[0])
            .add_child_entry(keys[1], child_pointers[1])
            .build()
    }

    /// Merges child entries into the internal node.
    pub fn merge_child_entries<'w>(
        &self,
        writer: &'w Writer,
        entries: &[ChildEntry],
    ) -> InternalEffect<'w> {
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
                ChildEntry::Insert { key, .. } => key.len() as isize + 8,
                ChildEntry::Update { i, key, .. } => {
                    key.len() as isize - self.get_key(*i).len() as isize
                }
                ChildEntry::Delete { i } => -8 - (self.get_key(*i).len() as isize),
            })
            .sum::<isize>();
        let itr_func = || self.merge_iter(entries);
        let num_keys = (self.get_num_keys() as isize + delta_keys) as usize;
        let overflow = (self.get_num_bytes() as isize + delta_size) as usize > consts::PAGE_SIZE;
        if overflow {
            return build_split(writer, &itr_func, num_keys);
        }
        build(writer, itr_func(), num_keys)
    }

    /// Resolves underflow of either `left` or `right` by either having one
    /// steal from the other, or merging the two.
    pub fn steal_or_merge<'i, 'w>(
        writer: &'w Writer,
        left: &'i Internal<'a>,
        right: &'i Internal<'a>,
    ) -> InternalEffect<'w> {
        let itr_func = || left.iter().chain(right.iter());
        let num_keys = left.get_num_keys() + right.get_num_keys();
        if left.get_num_bytes() + right.get_num_bytes() - 4 > consts::PAGE_SIZE {
            // Steal
            return build_split(writer, &itr_func, num_keys);
        }
        // Merge
        build(writer, itr_func(), num_keys)
    }

    /// Finds the index of the child that contains the key.
    pub fn find(&self, key: &[u8]) -> usize {
        let n = self.get_num_keys();
        assert_ne!(n, 0);
        (1..n).rev().find(|i| self.get_key(*i) <= key).unwrap_or(0)
    }

    /// Gets the child pointer at an index.
    pub fn get_child_pointer(&self, i: usize) -> usize {
        get_child_pointer(&self.page, i)
    }

    /// Gets the number of keys.
    pub fn get_num_keys(&self) -> usize {
        header::get_num_keys(&self.page)
    }

    /// Gets the `i`th key in the internal buffer.
    pub fn get_key(&self, i: usize) -> &[u8] {
        get_key(&self.page, i)
    }

    pub fn get_num_bytes(&self) -> usize {
        get_num_bytes(&self.page)
    }

    /// Gets the page number associated to the internal node.
    pub fn page_num(&self) -> usize {
        self.page.page_num
    }

    /// Creates a child-entry iterator for the internal node.
    pub fn iter(&self) -> InternalIterator<'_, 'a> {
        InternalIterator {
            node: self,
            i: 0,
            n: self.get_num_keys(),
        }
    }

    /// Creates a child-entry iterator for the internal node merged with
    /// the specified child entries.
    fn merge_iter<'i>(
        &'i self,
        entries: &'i [ChildEntry],
    ) -> MergeIterator<'i, InternalIterator<'i, 'a>, std::slice::Iter<'i, ChildEntry>> {
        MergeIterator {
            node_iter: self.iter().enumerate().peekable(),
            entries_iter: entries.iter().peekable(),
        }
    }
}

impl<'a> TryFrom<ReadOnlyPage<'a>> for Internal<'a> {
    type Error = NodeError;
    fn try_from(page: ReadOnlyPage<'a>) -> Result<Self> {
        let node_type = header::get_node_type(page.deref())?;
        if node_type != NodeType::Internal {
            return Err(NodeError::UnexpectedNodeType(node_type as u16));
        }
        Ok(Internal { page })
    }
}

/// An enum representing the effect of an internal node operation.
pub enum InternalEffect<'a> {
    /// A newly created internal node that remained  "intact",
    /// i.e. it did not split.
    Intact(Internal<'a>),
    /// The left and right splits of an internal node that was created.
    Split {
        left: Internal<'a>,
        right: Internal<'a>,
    },
}

impl<'a> InternalEffect<'a> {
    #[allow(dead_code)]
    fn take_intact(self) -> Internal<'a> {
        match self {
            InternalEffect::Intact(internal) => internal,
            _ => panic!("is not InternalEffect::Intact"),
        }
    }

    #[allow(dead_code)]
    fn take_split(self) -> (Internal<'a>, Internal<'a>) {
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
struct Builder<'s, 'w> {
    i: usize,
    writer: &'w Writer<'s>,
    page: Page<'s, 'w>,
}

impl<'s, 'w> Builder<'s, 'w> {
    /// Creates a new internal node builder.
    fn new(writer: &'w Writer<'s>, num_keys: usize) -> Self {
        let mut page = writer.new_page();
        header::set_node_type(&mut page, NodeType::Internal);
        header::set_num_keys(&mut page, num_keys);
        Self { i: 0, writer, page }
    }

    /// Adds a child entry to the builder.
    fn add_child_entry(mut self, key: &[u8], page_num: usize) -> Self {
        let n = header::get_num_keys(&self.page);
        assert!(
            self.i < n,
            "add_child_entry() called {} times, cannot be called more times than num_keys = {}",
            self.i,
            n
        );
        assert!(key.len() <= consts::MAX_KEY_SIZE);

        let offset = set_next_offset(&mut self.page, self.i, n, key);
        set_child_pointer(&mut self.page, self.i, page_num);
        let pos = 4 + n * 10 + offset;
        assert!(
            pos + key.len() <= consts::PAGE_SIZE,
            "builder unexpectedly overflowed: i = {}, n = {}",
            self.i,
            n,
        );

        self.page[pos..pos + key.len()].copy_from_slice(key);

        self.i += 1;
        self
    }

    /// Builds a leaf.
    fn build(self) -> Internal<'w> {
        let n = header::get_num_keys(&self.page);
        assert!(
            self.i == n,
            "build() called after calling add_child_entry() {} times < num_keys = {}",
            self.i,
            n
        );
        let page = self.writer.write_page(self.page);
        page.try_into().unwrap()
    }
}

/// Finds the split point of an overflow internal node that is accessed via
/// an iterator of child entries.
fn find_split<'i, I>(itr: I, num_keys: usize) -> usize
where
    I: Iterator<Item = (&'i [u8], usize)>,
{
    assert!(num_keys >= 4);

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
fn build<'i, 's, 'w, I>(writer: &'w Writer<'s>, itr: I, num_keys: usize) -> InternalEffect<'w>
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
fn build_split<'i, 's, 'w, I, F>(
    writer: &'w Writer<'s>,
    itr_func: &'i F,
    num_keys: usize,
) -> InternalEffect<'w>
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
pub struct InternalIterator<'i, 'a> {
    node: &'i Internal<'a>,
    i: usize,
    n: usize,
}

impl<'i> Iterator for InternalIterator<'i, '_> {
    type Item = (&'i [u8], usize);

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
struct MergeIterator<'i, I, E>
where
    I: Iterator<Item = (&'i [u8], usize)>,
    E: Iterator<Item = &'i ChildEntry>,
{
    node_iter: Peekable<std::iter::Enumerate<I>>,
    entries_iter: Peekable<E>,
}

impl<'i, I, E> Iterator for MergeIterator<'i, I, E>
where
    I: Iterator<Item = (&'i [u8], usize)>,
    E: Iterator<Item = &'i ChildEntry>,
{
    type Item = (&'i [u8], usize);

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
fn get_key(page: &[u8], i: usize) -> &[u8] {
    let n = header::get_num_keys(page);
    let offset = get_offset(page, i);
    let key_len = get_offset(page, i + 1) - offset;
    &page[4 + n * 10 + offset..4 + n * 10 + offset + key_len]
}

/// Gets the `i`th child pointer in an internal node's page buffer.
fn get_child_pointer(page: &[u8], i: usize) -> usize {
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
fn set_child_pointer(page: &mut [u8], i: usize, page_num: usize) {
    page[4 + i * 8..4 + (i + 1) * 8].copy_from_slice(&(page_num as u64).to_le_bytes());
}

/// Gets the `i`th offset value.
fn get_offset(page: &[u8], i: usize) -> usize {
    if i == 0 {
        return 0;
    }
    let n = header::get_num_keys(page);
    u16::from_le_bytes([page[4 + n * 8 + 2 * (i - 1)], page[4 + n * 8 + 2 * i - 1]]) as usize
}

/// Sets the next (i.e. `i+1`th) offset and returns the current offset.
fn set_next_offset(page: &mut [u8], i: usize, n: usize, key: &[u8]) -> usize {
    let curr_offset = get_offset(page, i);
    let next_offset = curr_offset + key.len();
    let next_i = i + 1;
    page[4 + n * 8 + 2 * (next_i - 1)..4 + n * 8 + 2 * next_i]
        .copy_from_slice(&(next_offset as u16).to_le_bytes());
    curr_offset
}

/// Gets the number of bytes consumed by a page.
fn get_num_bytes(page: &[u8]) -> usize {
    let n = header::get_num_keys(page);
    let offset = get_offset(page, n);
    4 + (n * 10) + offset
}

#[cfg(test)]
mod tests {
    use crate::mmap::{Mmap, Store};

    use super::*;
    use std::rc::Rc;

    fn new_test_store() -> Store {
        Store::new(Mmap::new_anonymous(0)).unwrap()
    }

    #[test]
    fn test_parent_of_split() {
        let store = new_test_store();
        let writer = store.writer();
        let parent = Internal::parent_of_split(
            &writer,
            ["key1".as_bytes(), "key2".as_bytes()],
            [111, 222],
        );
        assert_eq!(
            parent.iter().collect::<Vec<_>>(),
            vec![("key1".as_bytes(), 111), ("key2".as_bytes(), 222)]
        );
    }

    #[test]
    fn test_merge_child_entries_intact() {
        let store = new_test_store();
        let writer = store.writer();
        let node = Internal::parent_of_split(
            &writer,
            ["key1".as_bytes(), "key2".as_bytes()],
            [111, 222],
        );
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
        let store = new_test_store();
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
        let store = new_test_store();
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
        let store = new_test_store();
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

        let (left, right) = Internal::steal_or_merge(&writer, &left, &right).take_split();
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
        let store = new_test_store();
        let writer = store.writer();
        let left = Builder::new(&writer, 1)
            .add_child_entry(&[1], 1)
            .build();
        let right = Builder::new(&writer, 2)
            .add_child_entry(&[2], 2)
            .add_child_entry(&[3], 3)
            .build();

        let merged = Internal::steal_or_merge(&writer, &left, &right).take_intact();
        assert_eq!(
            merged.iter().collect::<Vec<_>>(),
            vec![(&[1][..], 1), (&[2], 2), (&[3], 3),]
        );
    }

    #[test]
    fn test_find() {
        let store = new_test_store();
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
