use crate::tree::buffer_store::BufferStore;
use crate::tree::node::{self, NodeType, Result};
use std::rc::Rc;

/// An enum representing the effect of an internal node operation.
#[derive(Debug)]
pub enum InternalEffect<B: BufferStore> {
    /// An internal node with 0 keys after a delete was performed on it.
    /// This is a special-case of `Underflow` done to avoid unnecessary
    /// page allocations, since empty non-root nodes aren't allowed.
    Empty,
    /// A newly created internal node that remained  "intact",
    /// i.e. it did not split.
    Intact(Internal<B>),
    /// The left and right splits of an internal node that was created.
    Split {
        left: Internal<B>,
        right: Internal<B>,
    },
}

impl<B: BufferStore> InternalEffect<B> {
    #[allow(dead_code)]
    fn take_intact(self) -> Internal<B> {
        match self {
            InternalEffect::Intact(internal) => internal,
            _ => panic!("{self:?} is not InternalEffect::Intact"),
        }
    }

    #[allow(dead_code)]
    fn take_split(self) -> (Internal<B>, Internal<B>) {
        match self {
            InternalEffect::Split { left, right } => (left, right),
            _ => panic!("{self:?} is not InternalEffect::Split"),
        }
    }
}

/// A child entry to insert, update, or delete in an internal node.
#[derive(Debug)]
pub enum ChildEntry {
    Insert {
        key: Rc<[u8]>,
        page_num: u64,
    },
    Update {
        i: usize,
        key: Rc<[u8]>,
        page_num: u64,
    },
    Delete {
        i: usize,
    },
}

/// Gets the `i`th key in an internal node's page buffer.
pub fn get_key(buf: &[u8], i: usize) -> &[u8] {
    let n = node::get_num_keys(buf);
    let offset = get_offset(buf, i);
    let key_len = get_offset(buf, i + 1) - offset;
    &buf[4 + n * 10 + offset..4 + n * 10 + offset + key_len]
}

/// Gets the `i`th child pointer in an internal node's page buffer.
pub fn get_child_pointer(buf: &[u8], i: usize) -> u64 {
    u64::from_be_bytes([
        buf[4 + i * 8],
        buf[4 + i * 8 + 1],
        buf[4 + i * 8 + 2],
        buf[4 + i * 8 + 3],
        buf[4 + i * 8 + 4],
        buf[4 + i * 8 + 5],
        buf[4 + i * 8 + 6],
        buf[4 + i * 8 + 7],
    ])
}

/// Sets the `i`th child pointer in an internal node's page buffer.
fn set_child_pointer(buf: &mut [u8], i: usize, page_num: u64) {
    buf[4 + i * 8..4 + (i + 1) * 8].copy_from_slice(&page_num.to_be_bytes());
}

/// Gets the `i`th offset value.
fn get_offset(buf: &[u8], i: usize) -> usize {
    if i == 0 {
        return 0;
    }
    let n = node::get_num_keys(buf);
    u16::from_be_bytes([buf[4 + n * 8 + 2 * (i - 1)], buf[4 + n * 8 + 2 * i - 1]]) as usize
}

/// Sets the next (i.e. `i+1`th) offset and returns the current offset.
fn set_next_offset(buf: &mut [u8], i: usize, n: usize, key: &[u8]) -> usize {
    let curr_offset = get_offset(buf, i);
    let next_offset = curr_offset + key.len();
    let next_i = i + 1;
    buf[4 + n * 8 + 2 * (next_i - 1)..4 + n * 8 + 2 * next_i]
        .copy_from_slice(&(next_offset as u16).to_be_bytes());
    curr_offset
}

/// Gets the number of bytes consumed by a page.
pub fn get_num_bytes(buf: &[u8]) -> usize {
    let n = node::get_num_keys(buf);
    let offset = get_offset(buf, n);
    4 + (n * 10) + offset
}

/// A builder of a B+ tree internal node.
pub struct InternalBuilder<'a, B: BufferStore> {
    i: usize,
    cap: usize,
    store: &'a B,
    buf: B::B,
}

impl<'a, B: BufferStore> InternalBuilder<'a, B> {
    /// Creates a new internal builder.
    pub fn new(num_keys: usize, store: &'a B, allow_overflow: bool) -> Self {
        let (mut buf, cap) = if allow_overflow {
            (store.get_buf(node::PAGE_SIZE * 2), 2 * node::PAGE_SIZE - 4)
        } else {
            (store.get_buf(node::PAGE_SIZE), node::PAGE_SIZE)
        };
        node::set_page_header(&mut buf, NodeType::Internal);
        node::set_num_keys(&mut buf, num_keys);
        Self {
            i: 0,
            cap,
            store,
            buf,
        }
    }

    /// Adds a child entry to the builder.
    pub fn add_child_entry(mut self, key: &[u8], page_num: u64) -> Result<Self> {
        let n = node::get_num_keys(&self.buf);
        assert!(
            self.i < n,
            "add_child_entry() called {} times, cannot be called more times than num_keys = {}",
            self.i,
            n
        );
        assert!(key.len() <= node::MAX_KEY_SIZE);

        let offset = set_next_offset(&mut self.buf, self.i, n, key);
        set_child_pointer(&mut self.buf, self.i, page_num);
        let pos = 4 + n * 10 + offset;
        assert!(
            pos + key.len() <= self.cap,
            "builder unexpectedly overflowed: i = {}, n = {}, pos = {}, key_len = {}, cap = {}",
            self.i,
            n,
            pos,
            key.len(),
            self.cap
        );

        self.buf[pos..pos + key.len()].copy_from_slice(key);

        self.i += 1;
        Ok(self)
    }

    /// Builds an internal node and optionally splits it if overflowed.
    pub fn build(self) -> Result<InternalEffect<B>> {
        let n = node::get_num_keys(&self.buf);
        assert!(
            self.i == n,
            "build() called after calling add_child_entry() {} times < num_keys = {}",
            self.i,
            n
        );
        if get_num_bytes(&self.buf) <= node::PAGE_SIZE {
            return Ok(InternalEffect::Intact(self.build_single()));
        }
        let (left, right) = self.build_split()?;
        Ok(InternalEffect::Split { left, right })
    }

    /// Builds an internal node.
    fn build_single(self) -> Internal<B> {
        let n = node::get_num_keys(&self.buf);
        assert!(
            self.i == n,
            "build_single() called after calling add_child_entry() {} times < num_keys = {}",
            self.i,
            n
        );
        assert!(get_num_bytes(&self.buf) <= node::PAGE_SIZE);
        Internal {
            buf: self.buf,
            store: self.store.clone(),
        }
    }

    /// Builds two splits of an internal node.
    fn build_split(self) -> Result<(Internal<B>, Internal<B>)> {
        let n = node::get_num_keys(&self.buf);
        // There should be at least 2 keys to be sufficient
        // in each of left and right.
        let left_end = (2..=n - 2)
            .rev()
            .find(|i| {
                let next_offset = get_offset(&self.buf, *i);
                4 + *i * 10 + next_offset <= node::PAGE_SIZE
            })
            .unwrap();

        let mut lb = Self::new(left_end, self.store, false);
        for i in 0..left_end {
            lb = lb.add_child_entry(get_key(&self.buf, i), get_child_pointer(&self.buf, i))?;
        }
        let mut rb = Self::new(n - left_end, self.store, false);
        for i in left_end..n {
            rb = rb.add_child_entry(get_key(&self.buf, i), get_child_pointer(&self.buf, i))?;
        }
        Ok((lb.build_single(), rb.build_single()))
    }
}

/// A B+ tree internal node.
#[derive(Debug)]
pub struct Internal<B: BufferStore> {
    buf: B::B,
    store: B,
}

impl<B: BufferStore> Internal<B> {
    /// Creates an internal node that is the parent of two splits.
    pub fn parent_of_split(keys: [&[u8]; 2], child_pointers: [u64; 2], store: &B) -> Result<Self> {
        let parent = InternalBuilder::new(2, store, false)
            .add_child_entry(keys[0], child_pointers[0])?
            .add_child_entry(keys[1], child_pointers[1])?
            .build()?
            .take_intact();
        Ok(parent)
    }

    /// Merges child entries into the internal node.
    pub fn merge_child_entries(&self, entries: &[ChildEntry]) -> Result<InternalEffect<B>> {
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
        let mut b = InternalBuilder::new(
            (self.get_num_keys() as isize + delta_keys) as usize,
            &self.store,
            (self.get_num_bytes() as isize + delta_size) as usize > node::PAGE_SIZE,
        );
        let mut self_iter = self.iter().enumerate().peekable();
        let mut entries_iter = entries.iter().peekable();
        loop {
            match (self_iter.peek(), entries_iter.peek()) {
                (None, None) => break,
                (Some((_, (k, pn))), None) => {
                    b = b.add_child_entry(k, *pn)?;
                    self_iter.next();
                }
                (None, Some(ce)) => match ce {
                    ChildEntry::Insert { key, page_num } => {
                        b = b.add_child_entry(key, *page_num)?;
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
                        b = b.add_child_entry(key, *page_num)?;
                        self_iter.next();
                        entries_iter.next();
                    }
                    ChildEntry::Delete { i: j } if *i == *j => {
                        self_iter.next();
                        entries_iter.next();
                    }
                    ChildEntry::Insert { key, page_num } if (*key).as_ref() < *k => {
                        b = b.add_child_entry(key, *page_num)?;
                        entries_iter.next();
                    }
                    ce @ ChildEntry::Insert { key, .. } if (*key).as_ref() == *k => {
                        panic!("ChildEntry {ce:?} has a duplicate key");
                    }
                    _ => {
                        b = b.add_child_entry(k, *pn)?;
                        self_iter.next();
                    }
                },
            }
        }
        b.build()
    }

    /// Resolves underflow of either `left` or `right` by either having one
    /// steal from the other, or merging the two.
    pub fn steal_or_merge(left: &Internal<B>, right: &Internal<B>) -> Result<InternalEffect<B>> {
        // TODO: Actually determine if overflow is needed.
        let allow_overflow = true;
        let mut b = InternalBuilder::new(
            left.get_num_keys() + right.get_num_keys(),
            &left.store,
            allow_overflow,
        );
        for (key, page_num) in left.iter() {
            b = b.add_child_entry(key, page_num)?;
        }
        for (key, page_num) in right.iter() {
            b = b.add_child_entry(key, page_num)?;
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
    pub fn get_child_pointer(&self, i: usize) -> u64 {
        get_child_pointer(&self.buf, i)
    }

    /// Gets the number of keys.
    pub fn get_num_keys(&self) -> usize {
        node::get_num_keys(&self.buf)
    }

    /// Gets the `i`th key in the internal buffer.
    pub fn get_key(&self, i: usize) -> &[u8] {
        get_key(&self.buf, i)
    }

    /// Creates an key-value iterator for the internal node.
    pub fn iter(&self) -> InternalIterator<B> {
        InternalIterator {
            node: self,
            i: 0,
            n: self.get_num_keys(),
        }
    }

    pub fn get_num_bytes(&self) -> usize {
        get_num_bytes(&self.buf)
    }
}

impl<B: BufferStore> Clone for Internal<B> {
    fn clone(&self) -> Self {
        let mut buf = self.store.get_buf(self.buf.len());
        buf.copy_from_slice(&self.buf);
        Self {
            buf,
            store: self.store.clone(),
        }
    }
}

/// A key-value iterator of an internal node.
pub struct InternalIterator<'a, B: BufferStore> {
    node: &'a Internal<B>,
    i: usize,
    n: usize,
}

impl<'a, B: BufferStore> Iterator for InternalIterator<'a, B> {
    type Item = (&'a [u8], u64);

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
    use crate::tree::{buffer_store::Heap, node};

    use super::{ChildEntry, Internal, InternalBuilder};

    static TEST_HEAP_STORE: Heap = Heap {};

    #[test]
    fn parent_of_slit() {
        let parent = Internal::parent_of_split(
            ["key1".as_bytes(), "key2".as_bytes()],
            [111, 222],
            &TEST_HEAP_STORE,
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
            &TEST_HEAP_STORE,
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
        let node = InternalBuilder::new(4, &TEST_HEAP_STORE, false)
            .add_child_entry(&[0; node::MAX_KEY_SIZE], 0)
            .unwrap()
            .add_child_entry(&[1; node::MAX_KEY_SIZE], 1)
            .unwrap()
            .add_child_entry(&[2; node::MAX_KEY_SIZE], 2)
            .unwrap()
            .add_child_entry(&[4; node::MAX_KEY_SIZE], 4)
            .unwrap()
            .build()
            .unwrap()
            .take_intact();
        let (left, right) = node
            .merge_child_entries(&[ChildEntry::Insert {
                key: [3; node::MAX_KEY_SIZE].into(),
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
                (&[0; node::MAX_KEY_SIZE][..], 0),
                (&[1; node::MAX_KEY_SIZE], 1),
                (&[2; node::MAX_KEY_SIZE], 2),
                (&[3; node::MAX_KEY_SIZE], 3),
                (&[4; node::MAX_KEY_SIZE], 4),
            ]
        );
    }

    #[test]
    fn merge_child_entries_udate_split() {
        let node = InternalBuilder::new(5, &TEST_HEAP_STORE, false)
            .add_child_entry(&[0; node::MAX_KEY_SIZE], 0)
            .unwrap()
            .add_child_entry(&[1; node::MAX_KEY_SIZE], 1)
            .unwrap()
            .add_child_entry(&[2; node::MAX_KEY_SIZE], 2)
            .unwrap()
            .add_child_entry(&[3; 1], 3)
            .unwrap()
            .add_child_entry(&[4; node::MAX_KEY_SIZE], 4)
            .unwrap()
            .build()
            .unwrap()
            .take_intact();
        let (left, right) = node
            .merge_child_entries(&[ChildEntry::Update {
                i: 3,
                key: [3; node::MAX_KEY_SIZE].into(),
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
                (&[0; node::MAX_KEY_SIZE][..], 0),
                (&[1; node::MAX_KEY_SIZE], 1),
                (&[2; node::MAX_KEY_SIZE], 2),
                (&[3; node::MAX_KEY_SIZE], 3),
                (&[4; node::MAX_KEY_SIZE], 4),
            ]
        );
    }

    #[test]
    fn steal_or_merge_steal() {
        let left = InternalBuilder::new(1, &TEST_HEAP_STORE, false)
            .add_child_entry(&[1; node::MAX_KEY_SIZE], 1)
            .unwrap()
            .build()
            .unwrap()
            .take_intact();
        let right = InternalBuilder::new(4, &TEST_HEAP_STORE, false)
            .add_child_entry(&[2; node::MAX_KEY_SIZE], 2)
            .unwrap()
            .add_child_entry(&[3; node::MAX_KEY_SIZE], 3)
            .unwrap()
            .add_child_entry(&[4; node::MAX_KEY_SIZE], 4)
            .unwrap()
            .add_child_entry(&[5; node::MAX_KEY_SIZE], 5)
            .unwrap()
            .build()
            .unwrap()
            .take_intact();

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
                (&[1; node::MAX_KEY_SIZE][..], 1),
                (&[2; node::MAX_KEY_SIZE], 2),
                (&[3; node::MAX_KEY_SIZE], 3),
                (&[4; node::MAX_KEY_SIZE], 4),
                (&[5; node::MAX_KEY_SIZE], 5),
            ]
        );
    }

    #[test]
    fn steal_or_merge_merge() {
        let left = InternalBuilder::new(1, &TEST_HEAP_STORE, false)
            .add_child_entry(&[1], 1)
            .unwrap()
            .build()
            .unwrap()
            .take_intact();
        let right = InternalBuilder::new(2, &TEST_HEAP_STORE, false)
            .add_child_entry(&[2], 2)
            .unwrap()
            .add_child_entry(&[3], 3)
            .unwrap()
            .build()
            .unwrap()
            .take_intact();

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
        let node = InternalBuilder::new(2, &TEST_HEAP_STORE, false)
            .add_child_entry(&[1], 1)
            .unwrap()
            .add_child_entry(&[2], 2)
            .unwrap()
            .build()
            .unwrap()
            .take_intact();
        assert_eq!(node.find(&[1]), 0);
        assert_eq!(node.find(&[3]), 1);

        // This works b/c we'd want to be able to insert &[0] into the 0-th child.
        assert_eq!(node.find(&[0]), 0);
    }
}
