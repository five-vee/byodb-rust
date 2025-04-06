use std::rc::Rc;

use crate::tree::consts;
use crate::tree::node::{self, NodeType, Result};
use crate::tree::page_store::{PageStore, ReadOnlyPage};

/// An enum representing the effect of an internal node operation.
pub enum InternalEffect<P: PageStore> {
    /// A newly created internal node that remained  "intact",
    /// i.e. it did not split.
    Intact(Internal<P>),
    /// The left and right splits of an internal node that was created.
    Split {
        left: Internal<P>,
        right: Internal<P>,
    },
}

impl<P: PageStore> InternalEffect<P> {
    #[allow(dead_code)]
    fn take_intact(self) -> Internal<P> {
        match self {
            InternalEffect::Intact(internal) => internal,
            _ => panic!("is not InternalEffect::Intact"),
        }
    }

    #[allow(dead_code)]
    fn take_split(self) -> (Internal<P>, Internal<P>) {
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

/// Gets the `i`th key in an internal node's page buffer.
fn get_key(page: &[u8], i: usize) -> &[u8] {
    let n = node::get_num_keys(page);
    let offset = get_offset(page, i);
    let key_len = get_offset(page, i + 1) - offset;
    &page[4 + n * 10 + offset..4 + n * 10 + offset + key_len]
}

/// Gets the `i`th child pointer in an internal node's page buffer.
fn get_child_pointer(page: &[u8], i: usize) -> usize {
    u64::from_be_bytes([
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
    page[4 + i * 8..4 + (i + 1) * 8].copy_from_slice(&(page_num as u64).to_be_bytes());
}

/// Gets the `i`th offset value.
fn get_offset(page: &[u8], i: usize) -> usize {
    if i == 0 {
        return 0;
    }
    let n = node::get_num_keys(page);
    u16::from_be_bytes([page[4 + n * 8 + 2 * (i - 1)], page[4 + n * 8 + 2 * i - 1]]) as usize
}

/// Sets the next (i.e. `i+1`th) offset and returns the current offset.
fn set_next_offset(page: &mut [u8], i: usize, n: usize, key: &[u8]) -> usize {
    let curr_offset = get_offset(page, i);
    let next_offset = curr_offset + key.len();
    let next_i = i + 1;
    page[4 + n * 8 + 2 * (next_i - 1)..4 + n * 8 + 2 * next_i]
        .copy_from_slice(&(next_offset as u16).to_be_bytes());
    curr_offset
}

/// Gets the number of bytes consumed by a page.
fn get_num_bytes(page: &[u8]) -> usize {
    let n = node::get_num_keys(page);
    let offset = get_offset(page, n);
    4 + (n * 10) + offset
}

trait ChildEntryAdder<P: PageStore> {
    fn add_child_entry(self, key: &[u8], page_num: usize) -> Self;
    fn build(self) -> Result<InternalEffect<P>>;
}

/// A builder of a B+ tree internal node.
struct Builder<P: PageStore> {
    i: usize,
    store: P,
    page: P::Page,
}

impl<P: PageStore> ChildEntryAdder<P> for Builder<P> {
    fn add_child_entry(self, key: &[u8], page_num: usize) -> Self {
        self.add_child_entry(key, page_num)
    }

    fn build(self) -> Result<InternalEffect<P>> {
        Ok(InternalEffect::Intact(self.build()))
    }
}

impl<P: PageStore> Builder<P> {
    /// Creates a new internal builder.
    fn new(num_keys: usize, store: P) -> Result<Self> {
        let mut page = store.new_page()?;
        node::set_node_type(&mut page, NodeType::Internal);
        node::set_num_keys(&mut page, num_keys);
        Ok(Self { i: 0, store, page })
    }

    /// Adds a child entry to the builder.
    fn add_child_entry(mut self, key: &[u8], page_num: usize) -> Self {
        let n = node::get_num_keys(&self.page);
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

    /// Builds an internal node and optionally splits it if overflowed.
    fn build(self) -> Internal<P> {
        let n = node::get_num_keys(&self.page);
        assert!(
            self.i == n,
            "build() called after calling add_child_entry() {} times < num_keys = {}",
            self.i,
            n
        );
        Internal {
            page: self.store.write_page(self.page),
            store: self.store.clone(),
        }
    }
}

struct SplitBuilder<P: PageStore> {
    i: usize,
    store: P,
    page: P::OverflowPage,
}

impl<P: PageStore> ChildEntryAdder<P> for SplitBuilder<P> {
    fn add_child_entry(self, key: &[u8], page_num: usize) -> Self {
        self.add_child_entry(key, page_num)
    }

    fn build(self) -> Result<InternalEffect<P>> {
        let (left, right) = self.build()?;
        Ok(InternalEffect::Split { left, right })
    }
}

impl<P: PageStore> SplitBuilder<P> {
    fn new(num_keys: usize, store: P) -> Result<Self> {
        let mut page = store.new_overflow_page()?;
        node::set_node_type(&mut page, NodeType::Internal);
        node::set_num_keys(&mut page, num_keys);
        Ok(Self { i: 0, store, page })
    }

    fn add_child_entry(mut self, key: &[u8], page_num: usize) -> Self {
        let n = node::get_num_keys(&self.page);
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
            pos + key.len() <= 2 * consts::PAGE_SIZE - 4,
            "builder unexpectedly overflowed: i = {}, n = {}",
            self.i,
            n,
        );

        self.page[pos..pos + key.len()].copy_from_slice(key);

        self.i += 1;
        self
    }

    fn build(mut self) -> Result<(Internal<P>, Internal<P>)> {
        let n = node::get_num_keys(&self.page);
        assert_eq!(self.i, n);
        // There should be at least 2 keys to be sufficient
        // in each of left and right.
        let left_n = (2..=n - 2)
            .rev()
            .find(|i| {
                let next_offset = get_offset(&self.page, *i);
                4 + *i * 10 + next_offset <= consts::PAGE_SIZE
            })
            .unwrap();

        // Build right split with a Builder.
        let right = {
            let mut b = Builder::new(n - left_n, self.store.clone())?;
            for i in left_n..n {
                b = b.add_child_entry(get_key(&self.page, i), get_child_pointer(&self.page, i));
            }
            b.build()
        };

        // Build left split via truncation.
        {
            let left_n_offset = get_offset(&self.page, left_n);
            // First truncate offsets array into pointers array.
            self.page
                .copy_within(4 + n * 8..4 + n * 8 + left_n * 2, 4 + left_n * 8);
            // Then truncate keys array into the remaining space.
            self.page
                .copy_within(4 + n * 10..4 + n * 10 + left_n_offset, 4 + left_n * 10);
        }
        node::set_num_keys(&mut self.page, left_n);
        let left = Internal {
            page: self.store.write_overflow_left_split(self.page)?,
            store: self.store.clone(),
        };

        Ok((left, right))
    }
}

/// A B+ tree internal node.
#[derive(Debug)]
pub struct Internal<P: PageStore> {
    page: P::ReadOnlyPage,
    store: P,
}

impl<P: PageStore> Internal<P> {
    /// Creates an internal node that is the parent of two splits.
    pub fn parent_of_split(keys: [&[u8]; 2], child_pointers: [usize; 2], store: P) -> Result<Self> {
        let parent = Builder::new(2, store.clone())?
            .add_child_entry(keys[0], child_pointers[0])
            .add_child_entry(keys[1], child_pointers[1])
            .build();
        Ok(parent)
    }

    pub fn from_page(store: P, page: P::ReadOnlyPage) -> Self {
        Internal { page, store }
    }

    pub fn page_num(&self) -> usize {
        self.page.page_num()
    }

    /// Merges child entries into the internal node.
    pub fn merge_child_entries(&self, entries: &[ChildEntry]) -> Result<InternalEffect<P>> {
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
        let overflow = (self.get_num_bytes() as isize + delta_size) as usize > consts::PAGE_SIZE;
        if overflow {
            return self.merge_child_entries_with_builder(
                SplitBuilder::new(
                    (self.get_num_keys() as isize + delta_keys) as usize,
                    self.store.clone(),
                )?,
                entries,
            );
        }
        self.merge_child_entries_with_builder(
            Builder::new(
                (self.get_num_keys() as isize + delta_keys) as usize,
                self.store.clone(),
            )?,
            entries,
        )
    }

    fn merge_child_entries_with_builder<B: ChildEntryAdder<P>>(
        &self,
        mut b: B,
        entries: &[ChildEntry],
    ) -> Result<InternalEffect<P>> {
        let mut self_iter = self.iter().enumerate().peekable();
        let mut entries_iter = entries.iter().peekable();
        loop {
            match (self_iter.peek(), entries_iter.peek()) {
                (None, None) => break,
                (Some((_, (k, pn))), None) => {
                    b = b.add_child_entry(k, *pn);
                    self_iter.next();
                }
                (None, Some(ce)) => match ce {
                    ChildEntry::Insert { key, page_num } => {
                        b = b.add_child_entry(key, *page_num);
                        entries_iter.next();
                    }
                    _ => panic!("ChildEntry {ce:?} cannot be applied without corresponding index"),
                },
                (Some((i, (k, pn))), Some(ce)) => match ce {
                    ChildEntry::Update {
                        i: j,
                        key,
                        page_num,
                    } if *i == *j => {
                        b = b.add_child_entry(key, *page_num);
                        self_iter.next();
                        entries_iter.next();
                    }
                    ChildEntry::Delete { i: j } if *i == *j => {
                        self_iter.next();
                        entries_iter.next();
                    }
                    ChildEntry::Insert { key, page_num } if (*key).as_ref() < *k => {
                        b = b.add_child_entry(key, *page_num);
                        entries_iter.next();
                    }
                    ce @ ChildEntry::Insert { key, .. } if (*key).as_ref() == *k => {
                        panic!("ChildEntry {ce:?} has a duplicate key");
                    }
                    _ => {
                        b = b.add_child_entry(k, *pn);
                        self_iter.next();
                    }
                },
            }
        }
        b.build()
    }

    /// Resolves underflow of either `left` or `right` by either having one
    /// steal from the other, or merging the two.
    pub fn steal_or_merge(left: &Internal<P>, right: &Internal<P>) -> Result<InternalEffect<P>> {
        if left.get_num_bytes() + right.get_num_bytes() - 4 > consts::PAGE_SIZE {
            // Steal
            return Self::steal_or_merge_with_builder(
                SplitBuilder::new(
                    left.get_num_keys() + right.get_num_keys(),
                    left.store.clone(),
                )?,
                left,
                right,
            );
        }
        // Merge
        Self::steal_or_merge_with_builder(
            Builder::new(
                left.get_num_keys() + right.get_num_keys(),
                left.store.clone(),
            )?,
            left,
            right,
        )
    }

    fn steal_or_merge_with_builder<B: ChildEntryAdder<P>>(
        mut b: B,
        left: &Internal<P>,
        right: &Internal<P>,
    ) -> Result<InternalEffect<P>> {
        for (key, page_num) in left.iter() {
            b = b.add_child_entry(key, page_num);
        }
        for (key, page_num) in right.iter() {
            b = b.add_child_entry(key, page_num);
        }
        b.build()
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
        node::get_num_keys(&self.page)
    }

    /// Gets the `i`th key in the internal buffer.
    pub fn get_key(&self, i: usize) -> &[u8] {
        get_key(&self.page, i)
    }

    /// Creates an key-value iterator for the internal node.
    pub fn iter(&self) -> InternalIterator<P> {
        InternalIterator {
            node: self,
            i: 0,
            n: self.get_num_keys(),
        }
    }

    pub fn get_num_bytes(&self) -> usize {
        get_num_bytes(&self.page)
    }
}

/// A key-value iterator of an internal node.
pub struct InternalIterator<'a, P: PageStore> {
    node: &'a Internal<P>,
    i: usize,
    n: usize,
}

impl<'a, P: PageStore> Iterator for InternalIterator<'a, P> {
    type Item = (&'a [u8], usize);

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

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use crate::tree::page_store::InMemory;

    use super::*;

    static TEST_STORE: OnceLock<InMemory> = OnceLock::new();

    fn test_store() -> InMemory {
        TEST_STORE.get_or_init(InMemory::new).clone()
    }

    #[test]
    fn parent_of_slit() {
        let parent = Internal::parent_of_split(
            ["key1".as_bytes(), "key2".as_bytes()],
            [111, 222],
            test_store(),
        )
        .unwrap();
        assert_eq!(
            parent.iter().collect::<Vec<_>>(),
            vec![("key1".as_bytes(), 111), ("key2".as_bytes(), 222)]
        );
    }

    #[test]
    fn merge_child_entries_intact() {
        let node = Internal::parent_of_split(
            ["key1".as_bytes(), "key2".as_bytes()],
            [111, 222],
            test_store(),
        )
        .unwrap();
        let node = node
            .merge_child_entries(&[
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
            ])
            .unwrap()
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
    fn merge_child_entries_insert_split() {
        let node = Builder::new(4, test_store())
            .unwrap()
            .add_child_entry(&[0; consts::MAX_KEY_SIZE], 0)
            .add_child_entry(&[1; consts::MAX_KEY_SIZE], 1)
            .add_child_entry(&[2; consts::MAX_KEY_SIZE], 2)
            .add_child_entry(&[4; consts::MAX_KEY_SIZE], 4)
            .build();
        let (left, right) = node
            .merge_child_entries(&[ChildEntry::Insert {
                key: [3; consts::MAX_KEY_SIZE].into(),
                page_num: 3,
            }])
            .unwrap()
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
    fn merge_child_entries_udate_split() {
        let node = Builder::new(5, test_store())
            .unwrap()
            .add_child_entry(&[0; consts::MAX_KEY_SIZE], 0)
            .add_child_entry(&[1; consts::MAX_KEY_SIZE], 1)
            .add_child_entry(&[2; consts::MAX_KEY_SIZE], 2)
            .add_child_entry(&[3; 1], 3)
            .add_child_entry(&[4; consts::MAX_KEY_SIZE], 4)
            .build();
        let (left, right) = node
            .merge_child_entries(&[ChildEntry::Update {
                i: 3,
                key: [3; consts::MAX_KEY_SIZE].into(),
                page_num: 3,
            }])
            .unwrap()
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
    fn steal_or_merge_steal() {
        let left = Builder::new(1, test_store())
            .unwrap()
            .add_child_entry(&[1; consts::MAX_KEY_SIZE], 1)
            .build();
        let right = Builder::new(4, test_store())
            .unwrap()
            .add_child_entry(&[2; consts::MAX_KEY_SIZE], 2)
            .add_child_entry(&[3; consts::MAX_KEY_SIZE], 3)
            .add_child_entry(&[4; consts::MAX_KEY_SIZE], 4)
            .add_child_entry(&[5; consts::MAX_KEY_SIZE], 5)
            .build();

        let (left, right) = Internal::steal_or_merge(&left, &right)
            .unwrap()
            .take_split();
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
    fn steal_or_merge_merge() {
        let left = Builder::new(1, test_store())
            .unwrap()
            .add_child_entry(&[1], 1)
            .build();
        let right = Builder::new(2, test_store())
            .unwrap()
            .add_child_entry(&[2], 2)
            .add_child_entry(&[3], 3)
            .build();

        let merged = Internal::steal_or_merge(&left, &right)
            .unwrap()
            .take_intact();
        assert_eq!(
            merged.iter().collect::<Vec<_>>(),
            vec![(&[1][..], 1), (&[2], 2), (&[3], 3),]
        );
    }

    #[test]
    fn find() {
        let node = Builder::new(2, test_store())
            .unwrap()
            .add_child_entry(&[1], 1)
            .add_child_entry(&[2], 2)
            .build();
        assert_eq!(node.find(&[1]), 0);
        assert_eq!(node.find(&[3]), 1);

        // This works b/c we'd want to be able to insert &[0] into the 0-th child.
        assert_eq!(node.find(&[0]), 0);
    }
}
