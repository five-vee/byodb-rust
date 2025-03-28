use std::rc::Rc;

use crate::tree::buffer_store::{BufferStore, PooledBuf};
use crate::tree::node::{self, NodeType, Result};

/// An enum representing the effect of an internal node operation.
#[derive(Debug)]
pub enum InternalEffect<S: BufferStore> {
    /// An internal node with 0 keys after a delete was performed on it.
    /// This is a special-case of `Underflow` done to avoid unnecessary
    /// page allocations, since empty non-root nodes aren't allowed.
    Empty,
    /// A newly created internal node that remained  "intact",
    /// i.e. it did not split.
    Intact(Internal<S>),
    /// The left and right splits of an internal node that was created.
    Split { left: Internal<S>, right: Internal<S> },
}

impl<S: BufferStore> InternalEffect<S> {
    fn take_intact(self) -> Internal<S> {
        match self {
            InternalEffect::Intact(internal) => internal,
            _ => panic!("{self:?} is not InternalEffect::Intact"),
        }
    }

    fn take_split(self) -> (Internal<S>, Internal<S>) {
        match self {
            InternalEffect::Split { left, right } => (left, right),
            _ => panic!("{self:?} is not InternalEffect::Split"),
        }
    }
}

/// A child entry to insert, update, or delete in an internal node.
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
    let offset = get_offset(buf, i) as usize;
    let num_keys = node::get_num_keys(buf) as usize;
    let key_len = u16::from_be_bytes([
        buf[4 + num_keys * 10 + offset],
        buf[4 + num_keys * 10 + offset + 1],
    ]) as usize;
    &buf[4 + num_keys * 10 + offset..4 + num_keys * 10 + offset + key_len]
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
    let next_i = i as usize + 1;
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
pub struct InternalBuilder<'a, S: BufferStore> {
    num_keys: usize,
    i: usize,
    cap: usize,
    store: &'a S,
    buf: PooledBuf<S>,
}

impl<'a, S: BufferStore> InternalBuilder<'a, S> {
    /// Creates a new internal builder.
    pub fn new(num_keys: usize, store: &'a S, allow_overflow: bool) -> Self {
        assert!(num_keys >= 2, "An internal node must have at least 2 keys.");
        let (mut buf, cap) = if allow_overflow {
            (store.get_buf(node::PAGE_SIZE * 2), 2 * node::PAGE_SIZE - 4)
        } else {
            (store.get_buf(node::PAGE_SIZE), node::PAGE_SIZE)
        };
        node::set_page_header(&mut buf, NodeType::Internal);
        Self {
            num_keys,
            i: 0,
            cap,
            store,
            buf,
        }
    }

    /// Adds a child entry to the builder.
    pub fn add_child_entry(mut self, key: &[u8], page_num: u64) -> Result<Self> {
        assert!(
            self.i < self.num_keys,
            "add_child_entry() called {} times, cannot be called more times than num_keys = {}",
            self.i,
            self.num_keys
        );
        assert!(key.len() <= node::MAX_KEY_SIZE);

        let n = self.num_keys;
        let offset = set_next_offset(&mut self.buf, self.i, n, key);
        set_child_pointer(&mut self.buf, self.i, page_num);
        let simulated_bytes = 4 + self.i * 10 + offset;
        assert!(
                simulated_bytes + key.len() <= self.cap,
                "builder unexpectedly overflowed; please call allow_overflow(), or don't add too many key-value pairs.");

        let pos = 4 + n * 10 + offset;
        self.buf[pos..pos + key.len()].copy_from_slice(key);

        self.i += 1;
        node::set_num_keys(&mut self.buf, self.i);
        Ok(self)
    }

    /// Builds an internal node and optionally splits it if overflowed.
    pub fn build(self) -> Result<InternalEffect<S>> {
        assert!(
            self.i == self.num_keys,
            "build() called after calling add_child_entry() {} times < num_keys = {}",
            self.i,
            self.num_keys
        );
        if get_num_bytes(&self.buf) <= node::PAGE_SIZE {
            return Ok(InternalEffect::Intact(self.build_single()));
        }
        let (left, right) = self.build_split()?;
        Ok(InternalEffect::Split { left, right })
    }

    /// Builds an internal node.
    fn build_single(self) -> Internal<S> {
        assert!(
            self.i == self.num_keys,
            "build_single() called after calling add_child_entry() {} times < num_keys = {}",
            self.i,
            self.num_keys
        );
        assert!(get_num_bytes(&self.buf) <= node::PAGE_SIZE);
        Internal { buf: self.buf }
    }

    /// Builds two splits of an internal node.
    fn build_split(self) -> Result<(Internal<S>, Internal<S>)> {
        let num_keys = self.num_keys as usize;
        let mut left_end: usize = 0;
        for i in 0..num_keys {
            // include i?
            let offset = get_offset(&self.buf, i + 1);
            if 4 + (i + 1) * 2 + offset <= node::PAGE_SIZE {
                left_end = i + 1;
            }
        }

        let mut lb = Self::new(left_end, self.store, false);
        for i in 0..left_end {
            lb = lb.add_child_entry(get_key(&self.buf, i), get_child_pointer(&self.buf, i))?;
        }
        let mut rb = Self::new(num_keys - left_end, self.store, false);
        for i in left_end..num_keys {
            rb = rb.add_child_entry(get_key(&self.buf, i), get_child_pointer(&self.buf, i))?;
        }
        Ok((lb.build_single(), rb.build_single()))
    }
}

/// A B+ tree internal node.
#[derive(Debug)]
pub struct Internal<S: BufferStore> {
    buf: PooledBuf<S>,
}

impl<S: BufferStore> Internal<S> {
    /// Creates an internal node that is the parent of two splits.
    pub fn parent_of_split(
        keys: [&[u8]; 2],
        child_pointers: [u64; 2],
        store: &S,
    ) -> Result<Self> {
        let parent = InternalBuilder::new(2, store, false)
            .add_child_entry(keys[0], child_pointers[0])?
            .add_child_entry(keys[1], child_pointers[1])?
            .build()?
            .take_intact();
        Ok(parent)
    }

    /// Merges child entries into the internal node.
    pub fn merge_child_entries(&self, entries: &[ChildEntry]) -> Result<InternalEffect<S>> {
        let extra = entries
            .iter()
            .filter(|ce| {
                if let ChildEntry::Insert { .. } = *ce {
                    true
                } else {
                    false
                }
            })
            .count();
        let mut b = InternalBuilder::new(
            self.get_num_keys() + extra,
            &self.buf.store,
            extra > 0,
        );
        let mut entries_iter = entries.iter();
        let mut next = entries_iter.next();
        for i in 0..self.get_num_keys() {
            if next.is_none() {
                b = b.add_child_entry(self.get_key(i), self.get_child_pointer(i))?;
                continue;
            }
            match next.unwrap() {
                ChildEntry::Insert { key, page_num } => {
                    b = b.add_child_entry(key, *page_num)?;
                    b = b.add_child_entry(self.get_key(i), self.get_child_pointer(i))?;
                    next = entries_iter.next();
                }
                ChildEntry::Update {
                    i: j,
                    key,
                    page_num,
                } if i == *j => {
                    b = b.add_child_entry(key, *page_num)?;
                    next = entries_iter.next();
                }
                ChildEntry::Delete { i: j } if i == *j => {
                    next = entries_iter.next();
                }
                _ => {
                    b = b.add_child_entry(self.get_key(i), self.get_child_pointer(i))?;
                }
            }
        }
        if let Some(ChildEntry::Insert { key, page_num }) = next {
            b = b.add_child_entry(key, *page_num)?;
        }
        b.build()
    }

    pub fn steal_or_merge(left: &Internal<S>, right: &Internal<S>) -> Result<InternalEffect<S>> {
        // TODO: Actually determine if overflow is needed.
        let allow_overflow = true;
        let mut b = InternalBuilder::new(
            left.get_num_keys() + right.get_num_keys(),
            &left.buf.store,
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
    pub fn find(&self, key: &[u8]) -> Option<usize> {
        (self.get_num_keys() - 1..=0).find(|i| self.get_key(*i) <= key)
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
    pub fn iter<'a>(&'a self) -> InternalIterator<'a, S> {
        InternalIterator {
            node: self,
            i: 0,
            n: self.get_num_keys(),
        }
    }
}

impl<S: BufferStore> Clone for Internal<S> {
    fn clone(&self) -> Self {
        let buf = self.buf.store.get_buf(self.buf.len());
        Self { buf }
    }
}

/// A key-value iterator of an internal node.
pub struct InternalIterator<'a, S: BufferStore> {
    node: &'a Internal<S>,
    i: usize,
    n: usize,
}

impl<'a, S: BufferStore> Iterator for InternalIterator<'a, S> {
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
