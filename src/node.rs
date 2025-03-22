use std::rc::Rc;

pub type Result<T> = std::result::Result<T, ()>;

#[derive(Debug)]
pub enum Node {
    Leaf(Leaf),
    Internal(Internal),
}

impl Node {
    pub fn get_key(&self, i: u16) -> &[u8] {
        match self {
            Node::Leaf(leaf) => leaf.get_key(i),
            Node::Internal(internal) => internal.get_key(i),
        }
    }
}

pub enum Upsert {
    Intact(Node),
    Split { left: Node, right: Node },
}

#[derive(Debug)]
pub enum Deletion {
    Empty,
    Sufficient(Node),
    Split { left: Node, right: Node },
    Underflow(Node),
}

pub fn sufficient_steal(from: &Node, into: &Node) -> bool {
    unimplemented!()
}

pub fn sufficient_merge(from: &Node, into: &Node) -> bool {
    unimplemented!()
}

pub fn steal(from: Node, into: Node) -> Result<(Node, Node)> {
    unimplemented!();
}

pub fn merge(from: Node, into: Node) -> Result<Node> {
    unimplemented!();
}

pub type DeletionDelta = Rc<[(u16, Option<u64>)]>;

pub struct ChildEntry {
    pub maybe_i: Option<u16>,
    pub key: Rc<[u8]>,
    pub page_num: u64,
}

#[derive(Debug)]
pub struct Leaf {
    buf: Box<[u8]>,
}

struct LeafBuilder {
    buf: Box<[u8]>,
}

impl Default for LeafBuilder {
    fn default() -> Self {
        unimplemented!();
    }
}

impl LeafBuilder {
    fn new(len: usize) -> Self {
        unimplemented!();
    }

    fn add_key_value(mut self, key: &[u8], val: &[u8]) -> Self {
        unimplemented!();
    }

    fn build(self) -> Result<Leaf> {
        unimplemented!();
    }
}

impl Leaf {
    pub fn new(keys: &[&[u8]], vals: &[&[u8]]) -> Result<Self> {
        // Error if keys.len() != vals.len().
        // Error if resulting leaf is too large.
        // Build the leaf from keys + vals.
        unimplemented!();
    }

    pub fn insert(&self, key: &[u8], val: &[u8]) -> Result<Upsert> {
        // If the insertion can cause a split, allocate via LeafBuilder::new.
        // Otherwise, allocate via LeafBuilder::default.
        // Build the new leaf from self, key, and val.
        // If overflowed, split into two leaves.
        unimplemented!();
    }

    pub fn update(&self, key: &[u8], val: &[u8]) -> Result<Upsert> {
        // If the update can cause a split, allocate via LeafBuilder::new.
        // Otherwise, allocate via LeafBuilder::default.
        // Build the new leaf from self + key + val.
        // If overflowed, split into two leaves.
        unimplemented!();
    }

    pub fn delete(&self, key: &[u8]) -> Result<Deletion> {
        unimplemented!();
    }

    fn get_key(&self, i: u16) -> &[u8] {
        unimplemented!();
    }
}

#[derive(Debug)]
pub struct Internal {
    buf: Box<[u8]>,
}

struct InternalBuilder {
    buf: Box<[u8]>,
}

impl InternalBuilder {
    fn new(len: usize) -> Self {
        unimplemented!();
    }

    fn add_child(mut self, key: &[u8], pointer: u64) -> Self {
        unimplemented!();
    }

    fn build(self) -> Result<Internal> {
        unimplemented!();
    }
}

impl Internal {
    pub fn new(keys: &[&[u8]], child_pointers: &[u64]) -> Self {
        // Error if keys.len() != vals.len().
        // Error if resulting leaf is too large.
        // Build the internal from keys + vals.
        unimplemented!();
    }

    pub fn upsert_child_entries(&self, entries: &[ChildEntry]) -> Result<Upsert> {
        // If the connection can cause a split, allocate via InternalBuilder::new.
        // Otherwise, allocate via LeafBuilder::default.
        // Build the internal from self + child_entries.
        // If overflowed, split into two internals.
        unimplemented!();
    }

    pub fn find_child_pointer(&self, key: &[u8]) -> Result<(u16, u64)> {
        unimplemented!();
    }

    pub fn get_child_pointer(&self, i: u16) -> Result<u64> {
        unimplemented!();
    }

    pub fn delete_child_entry(&self, i: u16) -> Result<Deletion> {
        // delete child entry @ i
        unimplemented!();
    }

    pub fn update_child_entry(&self, i: u16, key: &[u8], page_num: u64) -> Result<Deletion> {
        // INVARIANT: internal is Sufficient.
        // update child page num @ i
        // If the update can cause a split, allocate via InternalBuilder::new.
        // Otherwise, allocate via InternalBuilder::default.
        // Build the new internal from self + key + val.
        // If overflowed, split into two internal.
        unimplemented!();
    }

    pub fn merge_delta(&self, delta: DeletionDelta) -> Result<Deletion> {
        unimplemented!();
    }

    pub fn get_num_keys(&self) -> u16 {
        unimplemented!();
    }

    fn get_key(&self, i: u16) -> &[u8] {
        unimplemented!();
    }
}
