use super::*;

pub use util::{ChildEntry, InternalBuilder};

/// Internal node utilities.
mod util {
    use std::rc::Rc;

    use super::{Deletion, Internal, Node, NodeError, NodeType, Result, Upsert};

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
        let num_keys = super::get_num_keys(buf) as usize;
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
        let n = super::get_num_keys(buf);
        u16::from_be_bytes([buf[4 + n * 8 + 2 * (i - 1)], buf[4 + n * 8 + 2 * i]]) as usize
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
        let n = super::get_num_keys(buf);
        let offset = get_offset(buf, n);
        4 + (n * 10) + offset
    }

    /// A builder of a B+ tree internal node.
    pub struct InternalBuilder {
        num_keys: usize,
        i: usize,
        buf: Option<Box<[u8]>>,
    }

    impl InternalBuilder {
        /// Creates a new internal builder.
        pub fn new(num_keys: usize) -> Self {
            assert!(num_keys >= 2, "An internal node must have at least 2 keys.");
            Self {
                num_keys,
                i: 0,
                buf: None,
            }
        }

        /// Allows the builder to overflow to two pages.
        pub fn allow_overflow(mut self) -> Self {
            assert!(
                self.buf.is_none(),
                "allow_overflow() must be called only once and before add_child_entry()"
            );
            self.buf = Some([0; 2 * super::PAGE_SIZE - 4].into());
            self
        }

        /// Adds a child entry to the builder.
        pub fn add_child_entry(mut self, key: &[u8], page_num: u64) -> Result<Self> {
            assert!(
                self.i < self.num_keys,
                "add_child_entry() called {} times, cannot be called more times than num_keys = {}",
                self.i,
                self.num_keys
            );
            if key.len() > super::MAX_KEY_SIZE {
                return Err(NodeError::MaxKeySize(key.len()));
            }

            // Make sure buffer is initialized.
            if self.buf.is_none() {
                self.buf = Some(Self::new_buffer());
            }
            let mut buf = self.buf.take().unwrap();

            let n = self.num_keys;
            let offset = set_next_offset(&mut buf, self.i, n, key);
            set_child_pointer(&mut buf, self.i, page_num);
            let simulated_bytes = 4 + self.i * 10 + offset;
            assert!(
                simulated_bytes + key.len() <= buf.len(),
                "builder unexpectedly overflowed; please call allow_overflow(), or don't add too many key-value pairs.");

            let pos = 4 + n * 10 + offset;
            buf[pos..pos + key.len()].copy_from_slice(key);

            self.i += 1;
            super::set_num_keys(&mut buf, self.i);
            self.buf = Some(buf);
            Ok(self)
        }

        /// Builds an Upsert.
        pub fn build_upsert(self) -> Result<Upsert> {
            assert!(
                self.i == self.num_keys,
                "build_upsert() must be called after calling add_child_entry() num_keys = {} times",
                self.num_keys
            );
            Ok(Self::new_upsert(self.build_single_or_split()?))
        }

        /// Builds a Deletion.
        pub fn build_deletion(self) -> Result<Deletion> {
            assert!(
                self.i == self.num_keys,
                "build_deletion() must be called after calling add_child_entry() num_keys = {} times",
                self.num_keys
            );
            Ok(Self::new_deletion(self.build_single_or_split()?))
        }

        /// Builds an internal node.
        pub fn build_single(mut self) -> Internal {
            assert!(
                self.i == self.num_keys,
                "build_single() must be called after calling add_child_entry() num_keys = {} times",
                self.num_keys
            );
            let buf = self.buf.take().unwrap();
            assert!(get_num_bytes(&buf) <= super::PAGE_SIZE);
            Internal {
                buf: buf[0..super::PAGE_SIZE].into(),
            }
        }

        /// Builds one internal node, or two due to splitting.
        pub fn build_single_or_split(mut self) -> Result<(Internal, Option<Internal>)> {
            assert!(
                self.i == self.num_keys,
                "build_single_or_split() must be called after calling add_child_entry() num_keys = {} times",
                self.num_keys
            );
            let buf = self.buf.take().unwrap();
            if get_num_bytes(&buf) <= super::PAGE_SIZE {
                return Ok((self.build_single(), None));
            }
            let (left, right) = self.build_split()?;
            Ok((left, Some(right)))
        }

        fn new_buffer() -> Box<[u8]> {
            let mut buf = [0; super::PAGE_SIZE];
            super::set_page_header(&mut buf, NodeType::Internal);
            buf.into()
        }

        /// Creates a new Upsert from at least 1 internal node.
        fn new_upsert(build_result: (Internal, Option<Internal>)) -> Upsert {
            match build_result {
                (left, Some(right)) => Upsert::Split {
                    left: Node::Internal(left),
                    right: Node::Internal(right),
                },
                (internal, None) => Upsert::Intact(Node::Internal(internal)),
            }
        }

        /// Creates a new Deletion from at least 1 leaf.
        fn new_deletion(build_result: (Internal, Option<Internal>)) -> Deletion {
            match build_result {
                (left, Some(right)) => Deletion::Split {
                    left: Node::Internal(left),
                    right: Node::Internal(right),
                },
                (internal, None) => {
                    let n = super::get_num_keys(&internal.buf);
                    if n == 0 {
                        Deletion::Empty
                    } else if n == 1 {
                        Deletion::Underflow(Node::Internal(internal))
                    } else {
                        Deletion::Sufficient(Node::Internal(internal))
                    }
                }
            }
        }

        /// Builds two splits of an internal node.
        fn build_split(mut self) -> Result<(Internal, Internal)> {
            let buf = self.buf.take().unwrap();
            let num_keys = self.num_keys as usize;
            let mut left_end: usize = 0;
            for i in 0..num_keys {
                // include i?
                let offset = get_offset(&buf, i + 1);
                if 4 + (i + 1) * 2 + offset <= super::PAGE_SIZE {
                    left_end = i + 1;
                }
            }

            let mut lb = Self::new(left_end);
            for i in 0..left_end {
                lb = lb.add_child_entry(get_key(&buf, i), get_child_pointer(&buf, i))?;
            }
            let mut rb = Self::new(num_keys - left_end);
            for i in left_end..num_keys {
                rb = rb.add_child_entry(get_key(&buf, i), get_child_pointer(&buf, i))?;
            }
            Ok((lb.build_single(), rb.build_single()))
        }
    }
}

/// A B+ tree internal node.
#[derive(Debug, Clone)]
pub struct Internal {
    buf: Box<[u8]>,
}

impl Internal {
    /// Creates an internal node that is the parent of two splits.
    pub fn parent_of_split(keys: [&[u8]; 2], child_pointers: [u64; 2]) -> Result<Self> {
        let parent = InternalBuilder::new(2)
            .add_child_entry(keys[0], child_pointers[0])?
            .add_child_entry(keys[1], child_pointers[1])?
            .build_single();
        Ok(parent)
    }

    /// Inserts or updates child entries.
    pub fn merge_as_upsert(&self, entries: &[ChildEntry]) -> Result<Upsert> {
        let b = self.merge_child_entries(entries)?;
        b.build_upsert()
    }

    /// Updates or deletes child entries.
    pub fn merge_as_deletion(&self, entries: &[ChildEntry]) -> Result<Deletion> {
        let b = self.merge_child_entries(entries)?;
        b.build_deletion()
    }

    /// Finds the index of the child that contains the key.
    pub fn find(&self, key: &[u8]) -> Option<usize> {
        (self.get_num_keys() - 1..=0).find(|i| self.get_key(*i) <= key)
    }

    /// Gets the child pointer at an index.
    pub fn get_child_pointer(&self, i: usize) -> u64 {
        util::get_child_pointer(&self.buf, i)
    }

    /// Gets the number of keys.
    pub fn get_num_keys(&self) -> usize {
        get_num_keys(&self.buf)
    }

    /// Gets the `i`th key in the internal buffer.
    pub fn get_key(&self, i: usize) -> &[u8] {
        util::get_key(&self.buf, i)
    }

    pub fn steal_or_merge(left: &Internal, right: &Internal) -> Result<Deletion> {
        let mut b =
            InternalBuilder::new(left.get_num_keys() + right.get_num_keys()).allow_overflow();
        for (key, page_num) in left.iter() {
            b = b.add_child_entry(key, page_num)?;
        }
        for (key, page_num) in right.iter() {
            b = b.add_child_entry(key, page_num)?;
        }
        b.build_deletion()
    }

    /// Creates an key-value iterator for the internal node.
    pub fn iter<'a>(&'a self) -> InternalIterator<'a> {
        InternalIterator {
            node: self,
            i: 0,
            n: self.get_num_keys(),
        }
    }

    /// Merges child entries into the internal node.
    /// Returns a builder so the user can decide on building
    /// an `Upsert` or a `Deletion`.
    fn merge_child_entries(&self, entries: &[ChildEntry]) -> Result<InternalBuilder> {
        let extra = entries
            .iter()
            .filter(|ce| {
                if let ChildEntry::Insert {
                    key: _,
                    page_num: _,
                } = *ce
                {
                    true
                } else {
                    false
                }
            })
            .count();
        let mut b = InternalBuilder::new(self.get_num_keys() + extra);
        if extra > 0 {
            b = b.allow_overflow();
        }
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
        Ok(b)
    }
}

/// A key-value iterator of an internal node.
pub struct InternalIterator<'a> {
    node: &'a Internal,
    i: usize,
    n: usize,
}

impl<'a> Iterator for InternalIterator<'a> {
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
