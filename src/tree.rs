//! A copy-on-write (COW) B+ tree is a variation of the standard B+ tree that
//! employs the copy-on-write technique for handling modifications. Instead of
//! directly updating the existing nodes in the tree when an insertion,
//! deletion, or update occurs, the COW approach creates a modified copy of the
//! node (or the path of nodes leading to the change). The original node
//! remains unchanged. This means that any other readers concurrently accessing
//! the tree will continue to see the consistent, older version of the data
//! until they reach a point where they would naturally access the newly
//! written parts.
//!
//! This method offers significant advantages, particularly in concurrent
//! environments. By not modifying the original structure in place,
//! COW B+ trees naturally support snapshot isolation and improve concurrency
//! control, as readers and writers do not contend for the same locks on nodes.
//! Furthermore, this approach aids in crash recovery, as the original,
//! consistent state of the tree is preserved until the new changes are fully
//! committed. It can also facilitate the implementation of features like
//! versioning and auditing, as previous states of the data structure are
//! implicitly retained.

mod node;

use std::rc::Rc;

pub use crate::error::TreeError;
use crate::mmap::{self, Guard, Writer};
use node::{ChildEntry, Internal, Node, NodeEffect, Sufficiency};

type Result<T> = std::result::Result<T, TreeError>;

/// A copy-on-write (COW) B+ Tree data structure that stores data in disk.
///
/// Only the tree's root node is stored in memory, as descendant nodes are
/// loaded into memory (and unloaded once dropped) dynamically during
/// read/write operations on the tree. The node contents are stored as pages
/// on disk every time the node is modified, as per the COW approach.
///
/// Instead of directly updating the existing nodes in the tree when an
/// insertion, deletion, or update occurs, the COW approach creates a modified
/// copy of the node (or the path of nodes leading to the change). The original
/// node remains unchanged. This means that any other readers concurrently
/// accessing the tree will continue to see the consistent, older version of
/// the data until they reach a point where they would naturally access the
/// newly written parts.
pub struct Tree<'g, G: Guard> {
    page_num: usize,
    guard: &'g G,
}

impl<'g, G: Guard> Tree<'g, G> {
    /// Loads the root of the tree found in the store.
    pub fn new(guard: &'g G) -> Self {
        let root_ptr = guard.read_meta_node().root_page;
        Tree {
            page_num: root_ptr,
            guard,
        }
    }

    /// Gets the value corresponding to the key.
    pub fn get(&self, key: &[u8]) -> Result<Option<&[u8]>> {
        match Node::read(self.guard, self.page_num) {
            Node::Internal(parent) => {
                let child_idx = parent.find(key);
                let child_num = parent.get_child_pointer(child_idx);
                let child = Tree {
                    page_num: child_num,
                    guard: self.guard,
                };
                child.get(key).map(|o| {
                    o.map(|val| {
                        // Safety: although val borrows from child,
                        // both self and child source the data bytes from the same
                        // reader, and therefore the same underlying mmap.
                        unsafe { std::slice::from_raw_parts(val.as_ptr(), val.len()) }
                    })
                })
            }
            Node::Leaf(root) => Ok(root.find(key)),
        }
    }
}

impl<'g> Tree<'g, Writer<'_>> {
    /// Inserts a key-value pair. The resulting tree won't be visible to
    /// readers until the writer is externally flushed.
    pub fn insert(self, key: &[u8], val: &[u8]) -> Result<Self> {
        let new_page_num = match &self.insert_helper(key, val)? {
            NodeEffect::Intact(new_root) => new_root.page_num(),
            NodeEffect::Split { left, right } => self.parent_of_split(left, right).page_num(),
            _ => unreachable!(),
        };
        Ok(Tree {
            page_num: new_page_num,
            guard: self.guard,
        })
    }

    /// Finds the node to insert into and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `insert`.
    fn insert_helper(&self, key: &'g [u8], val: &'g [u8]) -> Result<NodeEffect<'g>> {
        let node = Node::read(self.guard, self.page_num);
        match &node {
            // Base case
            Node::Leaf(leaf) => Ok(leaf.insert(self.guard, key, val)?.into()),
            // Recursive case
            Node::Internal(internal) => {
                // Find which child to recursively insert into.
                let child_idx = internal.find(key);
                let child_num = internal.get_child_pointer(child_idx);
                let child = Tree {
                    page_num: child_num,
                    guard: self.guard,
                };
                let child_entries = child.insert_helper(key, val)?.child_entries(child_idx);
                let effect = internal.merge_child_entries(self.guard, child_entries.as_ref());
                Ok(effect.into())
            }
        }
    }

    /// Updates the value corresponding to a key.
    pub fn update(self, key: &[u8], val: &[u8]) -> Result<Self> {
        let new_page_num = match &self.update_helper(key, val)? {
            NodeEffect::Intact(root) => root.page_num(),
            NodeEffect::Split { left, right } => self.parent_of_split(left, right).page_num(),
            _ => unreachable!(),
        };
        Ok(Tree {
            page_num: new_page_num,
            guard: self.guard,
        })
    }

    /// Finds the node to update and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `update`.
    fn update_helper(&self, key: &'g [u8], val: &'g [u8]) -> Result<NodeEffect<'g>> {
        let node = Node::read(self.guard, self.page_num);
        match &node {
            // Base case
            Node::Leaf(leaf) => Ok(leaf.update(self.guard, key, val)?.into()),
            // Recursive case
            Node::Internal(parent) => {
                // Find which child to recursively update at.
                let child_idx = parent.find(key);
                let child_num = parent.get_child_pointer(child_idx);
                let child = Tree {
                    page_num: child_num,
                    guard: self.guard,
                };
                let child_entries = child.update_helper(key, val)?.child_entries(child_idx);
                let effect = parent.merge_child_entries(self.guard, child_entries.as_ref());
                Ok(effect.into())
            }
        }
    }

    /// Deletes a key and its corresponding value.
    pub fn delete(self, key: &[u8]) -> Result<Self> {
        let new_page_num = match &self.delete_helper(key)? {
            NodeEffect::Empty => mmap::write_empty_leaf(self.guard),
            NodeEffect::Intact(root) => match node::sufficiency(&root) {
                Sufficiency::Empty => unreachable!(),
                Sufficiency::Underflow => match &root {
                    Node::Leaf(_) => root.page_num(),
                    Node::Internal(internal) => internal.get_child_pointer(0),
                },
                Sufficiency::Sufficient => root.page_num(),
            },
            NodeEffect::Split { left, right } => self.parent_of_split(left, right).page_num(),
        };
        Ok(Tree {
            page_num: new_page_num,
            guard: self.guard,
        })
    }

    /// Finds the node to delete the key from and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `delete`.
    fn delete_helper(&self, key: &[u8]) -> Result<NodeEffect<'g>> {
        let node = Node::read(self.guard, self.page_num);
        match &node {
            // Base case
            Node::Leaf(leaf) => Ok(leaf.delete(self.guard, key)?.into()),
            // Recursive case
            Node::Internal(parent) => {
                // Find which child to recursively delete from.
                let child_idx = parent.find(key);
                let child_num = parent.get_child_pointer(child_idx);
                let child = Tree {
                    page_num: child_num,
                    guard: self.guard,
                };
                match &child.delete_helper(key)? {
                    NodeEffect::Empty => {
                        let effect = parent.merge_child_entries(
                            self.guard,
                            &[ChildEntry::Delete { i: child_idx }],
                        );
                        Ok(effect.into())
                    }
                    NodeEffect::Intact(child) => match node::sufficiency(child) {
                        Sufficiency::Empty => unreachable!(),
                        Sufficiency::Underflow => {
                            Ok(self.try_fix_underflow(parent, child, child_idx))
                        }
                        Sufficiency::Sufficient => {
                            let child_entries = [ChildEntry::Update {
                                i: child_idx,
                                key: child.get_key(0).into(),
                                page_num: child.page_num(),
                            }];
                            let effect = parent.merge_child_entries(self.guard, &child_entries);
                            Ok(effect.into())
                        }
                    },
                    NodeEffect::Split { left, right } => {
                        let effect = parent.merge_child_entries(
                            self.guard,
                            &[
                                ChildEntry::Update {
                                    i: child_idx,
                                    key: left.get_key(0).into(),
                                    page_num: left.page_num(),
                                },
                                ChildEntry::Insert {
                                    key: right.get_key(0).into(),
                                    page_num: right.page_num(),
                                },
                            ],
                        );
                        Ok(effect.into())
                    }
                }
            }
        }
    }

    /// Fixes the underflow of `child` by stealing from or merging from
    /// one of its direct siblings
    /// (either at `child_idx - 1` or `child_idx + 1`) within the
    /// parent (internal) node.
    fn try_fix_underflow(
        &self,
        parent: &Internal<'g>,
        child: &Node<'g>,
        child_idx: usize,
    ) -> NodeEffect<'g> {
        // Try to steal from or merge with the left sibling.
        if child_idx > 0 {
            if let Some(effect) = self.try_steal_or_merge(parent, &child, child_idx, child_idx - 1)
            {
                return effect;
            }
        }
        // Try to do so with the right sibling.
        if child_idx < parent.get_num_keys() - 1 {
            if let Some(effect) = self.try_steal_or_merge(parent, &child, child_idx, child_idx + 1)
            {
                return effect;
            }
        }

        // Leave as underflow.
        parent
            .merge_child_entries(
                self.guard,
                &[ChildEntry::Update {
                    i: child_idx,
                    key: child.get_key(0).into(),
                    page_num: child.page_num(),
                }],
            )
            .into()
    }

    /// Fixes the underflow of `child` by stealing from or merging from
    /// one of its direct siblings.
    fn try_steal_or_merge(
        &self,
        parent: &Internal<'g>,
        child: &Node<'g>,
        child_idx: usize,
        sibling_idx: usize,
    ) -> Option<NodeEffect<'g>> {
        let sibling_num = parent.get_child_pointer(sibling_idx);
        let sibling = &Node::read(self.guard, sibling_num);

        let can_steal_or_merge = node::can_steal(sibling, child, sibling_idx < child_idx)
            || node::can_merge(sibling, child);
        if !can_steal_or_merge {
            return None;
        }

        let (mut left, mut right) = (sibling, child);
        let (mut left_idx, mut right_idx) = (sibling_idx, child_idx);
        if sibling_idx > child_idx {
            (right, left) = (sibling, child);
            (right_idx, left_idx) = (sibling_idx, child_idx);
        }
        match node::steal_or_merge(left, right, self.guard) {
            NodeEffect::Empty => unreachable!(),
            NodeEffect::Intact(child) => {
                // merged
                let effect = parent.merge_child_entries(
                    self.guard,
                    &[
                        ChildEntry::Update {
                            i: left_idx,
                            key: child.get_key(0).into(),
                            page_num: child.page_num(),
                        },
                        ChildEntry::Delete { i: right_idx },
                    ],
                );
                Some(effect.into())
            }
            NodeEffect::Split { left, right } => {
                // stolen
                let effect = parent.merge_child_entries(
                    self.guard,
                    &[
                        ChildEntry::Update {
                            i: left_idx,
                            key: left.get_key(0).into(),
                            page_num: left.page_num(),
                        },
                        ChildEntry::Update {
                            i: right_idx,
                            key: right.get_key(0).into(),
                            page_num: right.page_num(),
                        },
                    ],
                );
                Some(effect.into())
            }
        }
    }

    /// Creates a new internal root node whose children are split nodes
    /// newly-created due to an operation on the tree.
    fn parent_of_split(&self, left: &Node<'g>, right: &Node<'g>) -> Node<'g> {
        let keys = [left.get_key(0), right.get_key(0)];
        let child_pointers = [left.page_num(), right.page_num()];
        let root = Internal::parent_of_split(self.guard, keys, child_pointers);
        Node::Internal(root)
    }
}

#[cfg(test)]
impl<'g, G: Guard> Tree<'g, G> {
    /// Gets the height of the tree.
    /// This performs a scan of the entire tree, so it's not really efficient.
    fn height(&self) -> Result<u32> {
        let node = Node::read(self.guard, self.page_num);
        match &node {
            Node::Leaf(_) => Ok(1),
            Node::Internal(node) => {
                assert!(node.get_num_keys() >= 2);
                let mut height: Option<u32> = None;
                for (_, pn) in node.iter() {
                    let child = Tree {
                        page_num: pn,
                        guard: self.guard,
                    };
                    let child_height = child.height()?;
                    if height.is_some() {
                        assert_eq!(height.unwrap(), 1 + child_height);
                    }
                    height = Some(1 + child_height);
                }
                Ok(height.unwrap())
            }
        }
    }

    /// Iterates through the tree in-order.
    /// This is very slow, so be careful.
    fn inorder_iter(&self) -> InOrder<'g, G> {
        let copy = Self::new(self.guard);
        InOrder {
            stack: vec![(0, Rc::new(copy))],
        }
    }
}

#[cfg(test)]
struct InOrder<'g, G: Guard> {
    stack: Vec<(usize, Rc<Tree<'g, G>>)>,
}

#[cfg(test)]
impl<'g, G: Guard> Iterator for InOrder<'g, G> {
    type Item = (Rc<[u8]>, Rc<[u8]>);
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((i, tree)) = self.stack.pop() {
            let node = Node::read(tree.guard, tree.page_num);
            let n = node.get_num_keys();
            if i == n {
                continue;
            }
            match &node {
                Node::Leaf(leaf) => {
                    self.stack.push((i + 1, tree.clone()));
                    return Some((leaf.get_key(i).into(), leaf.get_value(i).into()));
                }
                Node::Internal(internal) => {
                    let pn = internal.get_child_pointer(i);
                    let child = Rc::new(Tree {
                        page_num: pn,
                        guard: tree.guard,
                    });
                    self.stack.push((i + 1, tree));
                    self.stack.push((0, child));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use crate::{
        consts,
        mmap::{Mmap, Store},
    };

    use super::*;

    fn new_file_mmap() -> (Store, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        println!("Created temporary file {path:?}");
        let mmap = Mmap::open_or_create(path).unwrap();
        let store = Store::new(mmap);
        (store, temp_file)
    }

    fn u64_to_key(i: u64) -> [u8; consts::MAX_KEY_SIZE] {
        let mut key = [0u8; consts::MAX_KEY_SIZE];
        key[0..8].copy_from_slice(&i.to_be_bytes());
        key
    }

    fn insert_until_height(writer: Writer, height: u32) {
        assert_ne!(height, 0);
        let mut tree = Tree::new(&writer);
        let mut i = 0u64;
        loop {
            if tree.height().unwrap() == height {
                break;
            }
            let x = u64_to_key(i);
            let result = tree.insert(&x, &x);
            assert!(
                result.is_ok(),
                "insert(i = {}) errored: {:?}",
                i,
                result.err().unwrap()
            );
            tree = result.unwrap();
            let found = tree.get(&x).unwrap();
            assert!(
                matches!(found, Some(v) if v == x),
                "did not find val for {i}"
            );
            i += 1;
        }
        writer.flush(tree.page_num);
    }

    fn insert_complete(writer: Writer, height: u32) {
        assert_ne!(height, 0);
        let mut tree = Tree::new(&writer);
        let mut i = 0u64;
        let mut root_ptr = tree.page_num;
        loop {
            let x = u64_to_key(i);
            let result = tree.insert(&x, &x);
            assert!(
                result.is_ok(),
                "insert(i = {}) errored: {:?}",
                i,
                result.err().unwrap()
            );
            let new_tree = result.unwrap();
            let found = new_tree.get(&x).unwrap();
            assert!(
                matches!(found, Some(v) if v == x),
                "did not find val for {i}"
            );
            i += 1;
            if new_tree.height().unwrap() > height {
                break;
            }
            tree = new_tree;
            root_ptr = tree.page_num;
        }
        writer.flush(root_ptr);
    }

    #[test]
    fn insert_into_empty_tree() {
        let (store, _temp_file) = new_file_mmap();
        {
            let writer = store.writer();
            let tree = Tree::new(&writer).insert(&[1], &[1]).unwrap();
            writer.flush(tree.page_num);
        }
        let reader = store.reader();
        let _ = Tree::new(&reader);
    }

    #[test]
    fn insert_until_split() {
        let (store, _temp_file) = new_file_mmap();
        insert_until_height(store.writer(), 3);
        let reader = store.reader();
        let tree = Tree::new(&reader);
        let root = Node::read(&reader, tree.page_num);
        assert!(matches!(root, Node::Internal(_)));
        assert!(root.get_num_keys() >= 2);
        let got = tree.inorder_iter().collect::<Vec<_>>();
        let want = (0..got.len() as u64)
            .map(|i| {
                let x = u64_to_key(i);
                let x: Rc<[u8]> = x.into();
                (x.clone(), x.clone())
            })
            .collect::<Vec<_>>();
        assert_eq!(got, want);
    }

    #[test]
    fn get() {
        let (store, _temp_file) = new_file_mmap();
        insert_until_height(store.writer(), 2);
        let reader = store.reader();
        let tree = Tree::new(&reader);
        let want = &u64_to_key(1u64);
        let got = tree.get(want).unwrap().unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn update_intact() {
        let (store, _temp_file) = new_file_mmap();
        let writer = store.writer();
        let tree = Tree::new(&writer)
            .insert(&[0], &[0])
            .unwrap()
            .update(&[0], &[1])
            .unwrap();
        writer.flush(tree.page_num);
        let reader = store.reader();
        let tree = Tree::new(&reader);
        let got = tree.get(&[0]).unwrap().unwrap();
        assert_eq!(got, &[1]);
        assert_eq!(tree.height().unwrap(), 1);
    }

    #[test]
    fn update_split() {
        let (store, _temp_file) = new_file_mmap();
        insert_complete(store.writer(), 2);
        let old_height = {
            let reader = store.reader();
            Tree::new(&reader).height().unwrap()
        };

        let writer = store.writer();
        let key = &u64_to_key(0);
        let new_value = &[0u8; consts::MAX_VALUE_SIZE];
        let tree = Tree::new(&writer).update(key, new_value).unwrap();
        let got = tree.get(key).unwrap().unwrap();
        assert_eq!(got, new_value);
        assert_eq!(tree.height().unwrap(), old_height + 1);
    }

    #[test]
    fn delete_until_empty() {
        let (store, _temp_file) = new_file_mmap();
        insert_until_height(store.writer(), 3);
        let max = {
            let reader = store.reader();
            let max_key = Tree::new(&reader)
                .inorder_iter()
                .last()
                .map(|(k, _)| k)
                .unwrap();
            u64::from_be_bytes([
                max_key[0], max_key[1], max_key[2], max_key[3], max_key[4], max_key[5], max_key[6],
                max_key[7],
            ])
        };

        let writer = store.writer();
        let mut tree = Tree::new(&writer);
        for i in 0..=max {
            let key = &u64_to_key(i);
            let result = tree.delete(key);
            assert!(
                result.is_ok(),
                "delete(i = {}) errored: {:?}",
                i,
                result.err().unwrap()
            );
            tree = result.unwrap();
            let result = tree.get(key);
            assert!(
                result.is_ok(),
                "get(deleted i = {}) errored: {:?}",
                i,
                result.err().unwrap()
            );
            let got = result.unwrap();
            assert!(
                got.is_none(),
                "get(deleted i = {}) was found = {:?}",
                i,
                got.unwrap()
            );
        }
        writer.flush(tree.page_num);

        let reader = store.reader();
        let tree = Tree::new(&reader);
        assert_eq!(tree.height().unwrap(), 1);
        let root = Node::read(&reader, tree.page_num);
        assert_eq!(root.get_num_keys(), 0);
    }

    // [ 1000          1                     1000          1000          1000      ]
    // [ 1000 3000 ] [ 1 1000, 1000 1000 ] [ 1000 3000 ] [ 1000 3000 ] [ 1000 3000 ]
    //
    // delete 1 1000
    //
    // [ 1000          1000          1000          1000          1000      ]
    // [ 1000 3000 ] [ 1000 1000 ] [ 1000 3000 ] [ 1000 3000 ] [ 1000 3000 ]
    //
    // split
    //
    // [ 1000                                      1000                    ]
    // [ 1000          1000          1000      ] [ 1000          1000      ]
    // [ 1000 3000 ] [ 1000 1000 ] [ 1000 3000 ] [ 1000 3000 ] [ 1000 3000 ]
    #[test]
    fn delete_triggers_higher_root() {
        // Setup
        let (store, _temp_file) = new_file_mmap();
        let writer = store.writer();
        let tree = Tree::new(&writer)
            .insert(&[0; consts::MAX_KEY_SIZE], &[0; consts::MAX_VALUE_SIZE])
            .unwrap()
            .insert(&[1], &[1; 1000])
            .unwrap()
            .insert(&[2; 1000], &[2; 1000])
            .unwrap()
            .insert(&[3; consts::MAX_KEY_SIZE], &[3; consts::MAX_VALUE_SIZE])
            .unwrap()
            .insert(&[4; consts::MAX_KEY_SIZE], &[4; consts::MAX_VALUE_SIZE])
            .unwrap()
            .insert(&[5; consts::MAX_KEY_SIZE], &[5; consts::MAX_VALUE_SIZE])
            .unwrap();
        assert_eq!(tree.height().unwrap(), 2);
        assert_eq!(Node::read(&writer, tree.page_num).get_num_keys(), 5);

        // Delete &[1]. This should trigger a split,
        // and the height should grow.
        let tree = tree.delete(&[1]).unwrap();
        assert!(tree.get(&[1]).unwrap().is_none());
        assert_eq!(tree.height().unwrap(), 3);
        assert_eq!(Node::read(&writer, tree.page_num).get_num_keys(), 2);

        // Delete the last leaf for more test coverage
        _ = tree.delete(&[5; consts::MAX_KEY_SIZE]).unwrap();
    }

    // [ 1000                                                                          1000                    ]
    // [ 1000          1000          1000          1                     1000      ] [ 1000          1000      ]
    // [ 1000 3000 ] [ 1000 3000 ] [ 1000 3000 ] [ 1 1000, 1000 1000 ] [ 1000 3000 ] [ 1000 3000 ] [ 1000 3000 ]
    //
    // delete 1 1000
    //
    // [ 1000                                                                  1000                    ]
    // [ 1000          1000          1000          1000          1000      ] [ 1000          1000      ]
    // [ 1000 3000 ] [ 1000 3000 ] [ 1000 3000 ] [ 1000 1000 ] [ 1000 3000 ] [ 1000 3000 ] [ 1000 3000 ]
    //
    // split
    //
    // [ 1000                                      1000                        1000                    ]
    // [ 1000          1000          1000      ] [ 1000          1000      ] [ 1000          1000      ]
    // [ 1000 3000 ] [ 1000 3000 ] [ 1000 3000 ] [ 1000 1000 ] [ 1000 3000 ] [ 1000 3000 ] [ 1000 3000 ]
    #[test]
    fn delete_triggers_larger_root() {
        // Setup
        let (store, _temp_file) = new_file_mmap();
        let writer = store.writer();
        let tree = Tree::new(&writer)
            .insert(&[0; consts::MAX_KEY_SIZE], &[0; consts::MAX_VALUE_SIZE])
            .unwrap()
            .insert(&[1; consts::MAX_KEY_SIZE], &[1; consts::MAX_VALUE_SIZE])
            .unwrap()
            .insert(&[2; consts::MAX_KEY_SIZE], &[2; consts::MAX_VALUE_SIZE])
            .unwrap()
            .insert(&[3], &[3; 1000])
            .unwrap()
            .insert(&[4; 1000], &[4; 1000])
            .unwrap()
            .insert(&[6; consts::MAX_KEY_SIZE], &[6; consts::MAX_VALUE_SIZE])
            .unwrap()
            .insert(&[7; consts::MAX_KEY_SIZE], &[7; consts::MAX_VALUE_SIZE])
            .unwrap()
            .insert(&[5; consts::MAX_KEY_SIZE], &[5; consts::MAX_VALUE_SIZE])
            .unwrap();
        assert_eq!(tree.height().unwrap(), 3);
        assert_eq!(Node::read(&writer, tree.page_num).get_num_keys(), 2);

        // Delete &[3]. This should trigger a split,
        // but the height shouldn't grow.
        let tree = tree.delete(&[3]).unwrap();
        assert!(tree.get(&[3]).unwrap().is_none());
        assert_eq!(tree.height().unwrap(), 3);
        assert_eq!(Node::read(&writer, tree.page_num).get_num_keys(), 3);
    }

    // [ 1000 x 2 ]
    // [ 1000 x 3 ], [ 1000 x 2 ]
    // [ 1000 x 4 ] x 4, [ 1000 x 4, 1, 1 ]
    // [ 1000 3000 ] x 20, [ 1 1000, 1000 1000 ], [ 1 2000, 1 1000, 1 1000 ]
    //
    // delete 1 1000
    //
    // [ 1000 x 2 ]
    // [ 1000 x 3 ], [ 1000 x 2 ]
    // [ 1000 x 4 ] x 4, [ 1000 x 5, 1 ]
    // [ 1000 3000 ] x 20, [ 1000 1000 ], [ 1 2000, 1 1000, 1 1000 ]
    //
    // split
    //
    // [ 1000 x 2 ]
    // [ 1000 x 3 ] x 2
    // [ 1000 x 4 ] x 4, [ 1000 x 4 ], [ 1000, 1 ]
    // [ 1000 3000 ] x 20, [ 1000 1000, 1 2000 ], [ 1 1000, 1 1000 ]
    #[test]
    fn delete_triggers_internal_split() {
        // Setup
        let (store, _temp_file) = new_file_mmap();
        let writer = store.writer();
        let mut tree = Tree::new(&writer);
        for i in 0u8..=19u8 {
            tree = tree
                .insert(&[i; consts::MAX_KEY_SIZE], &[i; consts::MAX_VALUE_SIZE])
                .unwrap();
        }
        tree = tree
            .insert(&[20], &[20; 1000])
            .unwrap()
            .insert(&[21; 1000], &[21; 1000])
            .unwrap()
            .insert(&[22], &[22; 2000])
            .unwrap()
            .insert(&[23], &[23; 1000])
            .unwrap()
            .insert(&[24], &[24; 1000])
            .unwrap();
        assert_eq!(tree.height().unwrap(), 4);
        assert_eq!(Node::read(&writer, tree.page_num).get_num_keys(), 2);

        // Delete &[20]. This should trigger a split,
        // but the height shouldn't grow.
        // Neither should root change number of keys.
        let tree = tree.delete(&[20]).unwrap();
        assert!(tree.get(&[20]).unwrap().is_none());
        assert_eq!(tree.height().unwrap(), 4);
        assert_eq!(Node::read(&writer, tree.page_num).get_num_keys(), 2);
    }
}
