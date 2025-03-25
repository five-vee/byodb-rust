use super::*;

use util::LeafBuilder;

/// Leaf node utilities.
mod util {
    use super::{Deletion, Leaf, Node, NodeError, NodeType, Result, Upsert};

    /// Gets the `i`th key in a leaf page buffer.
    pub fn get_key(buf: &[u8], i: usize) -> &[u8] {
        let offset = get_offset(buf, i) as usize;
        let num_keys = super::get_num_keys(buf) as usize;
        let key_len = u16::from_be_bytes([
            buf[4 + num_keys * 2 + offset],
            buf[4 + num_keys * 2 + offset + 1],
        ]) as usize;
        &buf[4 + num_keys * 2 + offset + 4..4 + num_keys * 2 + offset + 4 + key_len]
    }

    /// Gets the `i`th value in a leaf page buffer.
    pub fn get_value(buf: &[u8], i: usize) -> &[u8] {
        let offset = get_offset(buf, i) as usize;
        let num_keys = super::get_num_keys(buf) as usize;
        let key_len = u16::from_be_bytes([
            buf[4 + num_keys * 2 + offset],
            buf[4 + num_keys * 2 + offset + 1],
        ]) as usize;
        let val_len = u16::from_be_bytes([
            buf[4 + num_keys * 2 + offset + 2],
            buf[4 + num_keys * 2 + offset + 3],
        ]) as usize;
        &buf[4 + num_keys * 2 + offset + 4 + key_len
            ..4 + num_keys * 2 + offset + 4 + key_len + val_len]
    }

    /// Gets the `i`th offset value.
    fn get_offset(buf: &[u8], i: usize) -> usize {
        if i == 0 {
            return 0;
        }
        u16::from_be_bytes([buf[4 + 2 * (i - 1)], buf[4 + 2 * i]]) as usize
    }

    /// Gets the number of bytes consumed by a page.
    pub fn get_num_bytes(buf: &[u8]) -> usize {
        let n = super::get_num_keys(buf);
        let offset = get_offset(buf, n);
        4 + (n * 2) + offset
    }

    /// Sets the next (i.e. `i+1`th) offset and returns the current offset.
    fn set_next_offset(buf: &mut [u8], i: usize, key: &[u8], val: &[u8]) -> usize {
        let curr_offset = get_offset(buf, i);
        let next_offset = curr_offset + 4 + key.len() + val.len();
        let next_i = i as usize + 1;
        buf[4 + 2 * (next_i - 1)..4 + 2 * next_i]
            .copy_from_slice(&(next_offset as u16).to_be_bytes());
        curr_offset
    }

    // A builder of a B+ tree leaf node.
    pub struct LeafBuilder {
        num_keys: usize,
        i: usize,
        buf: Option<Box<[u8]>>,
    }

    impl LeafBuilder {
        /// Creates a new leaf builder.
        pub fn new(num_keys: usize) -> Self {
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
                "allow_overflow() must be called only once and before add_key_value()"
            );
            self.buf = Some([0; 2 * super::PAGE_SIZE - 4].into());
            self
        }

        /// Adds a key-value pair to the builder.
        pub fn add_key_value(mut self, key: &[u8], val: &[u8]) -> Result<Self> {
            assert!(
                self.i < self.num_keys,
                "add_key_value() called {} times, cannot be called more times than num_keys = {}",
                self.i,
                self.num_keys
            );
            if key.len() > super::MAX_KEY_SIZE {
                return Err(NodeError::MaxKeySize(key.len()));
            }
            if val.len() > super::MAX_VALUE_SIZE {
                return Err(NodeError::MaxValueSize(val.len()));
            }

            // Make sure buffer is initialized.
            if self.buf.is_none() {
                self.buf = Some(Self::new_buffer());
            }
            let mut buf = self.buf.take().unwrap();

            let n = self.num_keys;
            let offset = set_next_offset(&mut buf, self.i, key, val);
            let simulated_bytes = 4 + self.i * 2 + offset;
            assert!(simulated_bytes + 4 + key.len() + val.len() <= buf.len(),
            "builder unexpectedly overflowed; please call allow_overflow(), or don't add too many key-value pairs.");

            let pos = 4 + n * 2 + offset;
            buf[pos..pos + 2].copy_from_slice(&(key.len() as u16).to_be_bytes());
            buf[pos + 2..pos + 4].copy_from_slice(&(val.len() as u16).to_be_bytes());
            buf[pos + 4..pos + 4 + key.len()].copy_from_slice(key);
            buf[pos + 4 + key.len()..pos + 4 + key.len() + val.len()].copy_from_slice(val);

            self.i += 1;
            super::set_num_keys(&mut buf, self.i);
            self.buf = Some(buf);
            Ok(self)
        }

        /// Builds an Upsert.
        pub fn build_upsert(self) -> Result<Upsert> {
            assert!(
                self.i == self.num_keys,
                "build_upsert() called after calling add_key_value() {} times < num_keys = {}",
                self.i,
                self.num_keys
            );
            Ok(Self::new_upsert(self.build_single_or_split()?))
        }

        /// Builds a Deletion.
        pub fn build_deletion(self) -> Result<Deletion> {
            assert!(
                self.i == self.num_keys,
                "build_deletion() called after calling add_key_value() {} times < num_keys = {}",
                self.i,
                self.num_keys
            );
            Ok(Self::new_deletion(self.build_single_or_split()?))
        }

        /// Creates a new page-sized in-memory buffer.
        fn new_buffer() -> Box<[u8]> {
            let mut buf = [0; super::PAGE_SIZE];
            super::set_page_header(&mut buf, NodeType::Leaf);
            buf.into()
        }

        /// Creates a new Upsert from at least 1 leaf.
        fn new_upsert(build_result: (Leaf, Option<Leaf>)) -> Upsert {
            match build_result {
                (left, Some(right)) => Upsert::Split {
                    left: Node::Leaf(left),
                    right: Node::Leaf(right),
                },
                (leaf, None) => Upsert::Intact(Node::Leaf(leaf)),
            }
        }

        /// Creates a new Deletion from at least 1 leaf.
        fn new_deletion(build_result: (Leaf, Option<Leaf>)) -> Deletion {
            match build_result {
                (left, Some(right)) => Deletion::Split {
                    left: Node::Leaf(left),
                    right: Node::Leaf(right),
                },
                (leaf, None) => {
                    let n = super::get_num_keys(&leaf.buf);
                    if n == 0 {
                        Deletion::Empty
                    } else if n == 1 {
                        Deletion::Underflow(Node::Leaf(leaf))
                    } else {
                        Deletion::Sufficient(Node::Leaf(leaf))
                    }
                }
            }
        }

        /// Builds one leaf, or two due to splitting.
        fn build_single_or_split(mut self) -> Result<(Leaf, Option<Leaf>)> {
            if self.buf.is_none() {
                // Technically, an empty leaf is allowed.
                return Ok((Leaf::default(), None));
            }
            let buf = self.buf.take().unwrap();
            if get_num_bytes(&buf) <= super::PAGE_SIZE {
                self.buf = Some(buf);
                return Ok((self.build_single(), None));
            }
            self.buf = Some(buf);
            let (left, right) = self.build_split()?;
            Ok((left, Some(right)))
        }

        /// Builds two splits of a leaf.
        fn build_split(mut self) -> Result<(Leaf, Leaf)> {
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
                lb = lb.add_key_value(get_key(&buf, i), get_value(&buf, i))?;
            }
            let mut rb = Self::new(num_keys - left_end);
            for i in left_end..num_keys {
                rb = rb.add_key_value(get_key(&buf, i), get_value(&buf, i))?;
            }
            Ok((lb.build_single(), rb.build_single()))
        }

        /// Builds a leaf.
        fn build_single(mut self) -> Leaf {
            let buf = self.buf.take().unwrap();
            assert!(get_num_bytes(&buf) <= super::PAGE_SIZE);
            Leaf {
                buf: buf[0..super::PAGE_SIZE].into(),
            }
        }
    }
}

/// A B+ tree leaf node.
#[derive(Debug, Clone)]
pub struct Leaf {
    buf: Box<[u8]>,
}

impl Default for Leaf {
    fn default() -> Self {
        let mut buf = Box::new([0; PAGE_SIZE]);
        set_page_header(buf.as_mut(), NodeType::Leaf);
        return Self { buf };
    }
}

impl Leaf {
    /// Inserts a key-value pair.
    pub fn insert(&self, key: &[u8], val: &[u8]) -> Result<Upsert> {
        if self.find(key).is_some() {
            return Err(NodeError::AlreadyExists);
        }
        let mut b = LeafBuilder::new(self.get_num_keys() + 1).allow_overflow();
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
        b.build_upsert()
    }

    /// Updates the value corresponding to a key.
    pub fn update(&self, key: &[u8], val: &[u8]) -> Result<Upsert> {
        if self.find(key).is_none() {
            return Err(NodeError::KeyNotFound);
        }
        let mut b = LeafBuilder::new(self.get_num_keys()).allow_overflow();
        let mut added = false;
        for (k, v) in self.iter() {
            if !added && key == k {
                added = true;
                b = b.add_key_value(key, val)?;
                continue;
            }
            b = b.add_key_value(k, v)?;
        }
        b.build_upsert()
    }

    /// Deletes a key and its corresponding value.
    pub fn delete(&self, key: &[u8]) -> Result<Deletion> {
        if self.find(key).is_none() {
            return Err(NodeError::KeyNotFound);
        }
        // Optimization: avoid memory allocation and
        // just return Deletion::Empty if only 1 key.
        let n = self.get_num_keys();
        if n == 1 {
            return Ok(Deletion::Empty);
        }
        let mut b = LeafBuilder::new(n - 1).allow_overflow();
        let mut added = false;
        for (k, v) in self.iter() {
            if !added && key == k {
                added = true;
                continue;
            }
            b = b.add_key_value(k, v)?;
        }
        b.build_deletion()
    }

    /// Finds the value corresponding to the queried key.
    pub fn find(&self, key: &[u8]) -> Option<&[u8]> {
        self.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    pub fn steal_or_merge(left: &Leaf, right: &Leaf) -> Result<Deletion> {
        let mut b = LeafBuilder::new(left.get_num_keys() + right.get_num_keys()).allow_overflow();
        for (key, val) in left.iter() {
            b = b.add_key_value(key, val)?;
        }
        for (key, val) in right.iter() {
            b = b.add_key_value(key, val)?;
        }
        b.build_deletion()
    }

    pub fn get_key(&self, i: usize) -> &[u8] {
        util::get_key(&self.buf, i)
    }

    pub fn get_num_keys(&self) -> usize {
        get_num_keys(&self.buf)
    }

    pub fn iter<'a>(&'a self) -> LeafIterator<'a> {
        LeafIterator {
            node: self,
            i: 0,
            n: self.get_num_keys(),
        }
    }

    fn get_value(&self, i: usize) -> &[u8] {
        util::get_value(&self.buf, i)
    }
}

/// An key-value iterator for a leaf node.
pub struct LeafIterator<'a> {
    node: &'a Leaf,
    i: usize,
    n: usize,
}

impl<'a> Iterator for LeafIterator<'a> {
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

    fn test_upsert_intact(u: Upsert, f: impl FnOnce(Leaf)) {
        match u {
            Upsert::Intact(Node::Leaf(leaf)) => { f(leaf) },
            _ => { panic!("{u:?} is not an Upsert::Intact(Node::Leaf(_))") }
        }
    }

    #[test]
    fn leaf_insert_intact() {
        let u = Leaf::default()
            .insert("hello".as_bytes(), "world".as_bytes())
            .unwrap();
        test_upsert_intact(u, |leaf| {
            assert_eq!(leaf.find("hello".as_bytes()).unwrap(), "world".as_bytes());
        })
    }

    #[test]
    fn leaf_insert_max_key_size() {
        let key = &[0u8; super::PAGE_SIZE + 1];
        let result = Leaf::default()
            .insert(key, "val".as_bytes());
        assert!(matches!(result, Err(NodeError::MaxKeySize(x)) if x == 4097));
    }

    #[test]
    fn leaf_insert_max_value_size() {
        let val = &[0u8; super::PAGE_SIZE + 1];
        let result = Leaf::default()
            .insert("key".as_bytes(), val);
        assert!(matches!(result, Err(NodeError::MaxValueSize(x)) if x == 4097));
    }
}
