use std::rc::Rc;

pub type Result<T> = std::result::Result<T, ()>;

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

pub enum NodeResult {
    NonSplit(Node),
    Split(Node, Node),
}

pub struct ChildEntry {
    pub parent_i: Option<u16>,
    pub min_key: Rc<[u8]>,
    pub page_num: u64,
}

pub struct Leaf { buf: Box<[u8]> }

struct LeafBuilder { buf: Box<[u8]> }

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
    
    pub fn insert(&self, key: &[u8], val: &[u8]) -> Result<NodeResult> {
        // If the insertion can cause a split, allocate via LeafBuilder::new.
        // Otherwise, allocate via LeafBuilder::default.
        // Build the new leaf from self, key, and val.
        // If overflowed, split into two leaves.
        unimplemented!();
    }
    
    pub fn update(&self, key: &[u8], val: &[u8]) -> Result<NodeResult> {
        // If the update can cause a split, allocate via LeafBuilder::new.
        // Otherwise, allocate via LeafBuilder::default.
        // Build the new leaf from self + key + val.
        // If overflowed, split into two leaves.
        unimplemented!();
    }
    
    fn get_key(&self, i: u16) -> &[u8] {
        unimplemented!();
    }
}

pub struct Internal { buf: Box<[u8]> }

struct InternalBuilder { buf: Box<[u8]> }

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
    
    pub fn connect_children(&self, new_children: &[ChildEntry]) -> Result<NodeResult> {
        // If the connection can cause a split, allocate via InternalBuilder::new.
        // Otherwise, allocate via LeafBuilder::default.
        // Build the internal from self + new_children.
        // If overflowed, split into two internals.
        unimplemented!();
    }
    
    fn get_key(&self, i: u16) -> &[u8] {
        unimplemented!();
    }
    
    pub(crate) fn find_child_pointer(&self, key: &[u8]) -> Result<(u16, u64)> {
        unimplemented!();
    }
}
