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

use std::ops::RangeBounds;

use crate::core::error::TreeError;
use crate::core::mmap::{self, Guard, Writer};
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
    guard: &'g G,
    page_num: usize,
}

impl<'g, G: Guard> Tree<'g, G> {
    /// Loads the root of the tree at a specified root page num.
    pub fn new(guard: &'g G, page_num: usize) -> Self {
        Tree { page_num, guard }
    }

    pub fn page_num(&self) -> usize {
        self.page_num
    }

    /// Gets the value corresponding to the key.
    pub fn get(&self, key: &[u8]) -> Result<Option<&'g [u8]>> {
        match Self::read(self.guard, self.page_num) {
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
            Node::Leaf(root) => Ok(root.get(key)),
        }
    }

    /// Iterates through the entire tree in-order.
    pub fn in_order_iter(&self) -> InOrder<'g, 'g, G> {
        InOrder {
            stack: vec![(0, Self::new(self.guard, self.page_num))],
            end_bound: std::ops::Bound::Unbounded,
        }
    }

    /// Iterates through the tree in-order, bounded by `range`.
    pub fn in_order_range_iter<'q, R>(&self, range: &'q R) -> InOrder<'q, 'g, G>
    where
        R: RangeBounds<[u8]>,
    {
        if matches!(range.start_bound(), std::ops::Bound::Unbounded) {
            return InOrder {
                stack: vec![(0, Self::new(self.guard, self.page_num))],
                end_bound: range.end_bound(),
            };
        }

        // Find.
        let start_bound = range.start_bound();
        type LeafPredicate<'q, 'g> = Box<dyn Fn((&'g [u8], &'g [u8])) -> bool + 'q>;
        let (leaf_predicate, start): (LeafPredicate<'q, 'g>, &[u8]) = match start_bound {
            std::ops::Bound::Included(start) => (Box::new(move |pair| pair.0 >= start), start),
            std::ops::Bound::Excluded(start) => (Box::new(move |pair| pair.0 > start), start),
            _ => unreachable!(),
        };
        let mut stack = vec![(0, Self::new(self.guard, self.page_num))];
        while let Some((i, tree)) = stack.pop() {
            let node = Tree::read(tree.guard, tree.page_num);
            let n = node.get_num_keys();
            if i == n {
                break;
            }
            match &node {
                Node::Leaf(leaf) => match leaf.iter().position(&leaf_predicate) {
                    // leaf range < start; try again up the stack
                    None => {}
                    Some(j) => {
                        stack.push((j, tree));
                        break;
                    }
                },
                Node::Internal(internal) => {
                    let child_idx = (i..internal.get_num_keys())
                        .rev()
                        .find(|&j| internal.get_key(j) <= start)
                        .unwrap_or(i);
                    let child_page = internal.get_child_pointer(child_idx);
                    let child = Tree::new(tree.guard, child_page);
                    stack.push((child_idx + 1, tree));
                    stack.push((0, child));
                }
            }
        }
        InOrder {
            stack,
            end_bound: range.end_bound(),
        }
    }

    /// Reads the page at `page_num` and returns it represented as a [`Node`].
    /// This is a convenience wrapper around the unsafe [`Node::read`].
    fn read(guard: &G, page_num: usize) -> Node<'_> {
        // Safety: The tree maintains the invariant that it'll only read nodes
        // that are traversible from the root, including the root itself.
        unsafe { Node::read(guard, page_num) }
    }
}

impl<'g> Tree<'g, Writer<'_>> {
    /// Inserts a key-value pair. The resulting tree won't be visible to
    /// readers until the writer is externally flushed.
    pub fn insert(self, key: &[u8], val: &[u8]) -> Result<Self> {
        let writer = self.guard;
        let new_page_num = match self.insert_helper(key, val)? {
            NodeEffect::Intact(new_root) => new_root.page_num(),
            NodeEffect::Split { left, right } => {
                Self::parent_of_split(writer, &left, &right).page_num()
            }
            _ => unreachable!(),
        };
        Ok(Tree {
            page_num: new_page_num,
            guard: writer,
        })
    }

    /// Finds the node to insert into and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `insert`.
    fn insert_helper(self, key: &[u8], val: &[u8]) -> Result<NodeEffect<'g>> {
        match Self::read(self.guard, self.page_num) {
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
        let writer = self.guard;
        let new_page_num = match self.update_helper(key, val)? {
            NodeEffect::Intact(root) => root.page_num(),
            NodeEffect::Split { left, right } => {
                Self::parent_of_split(writer, &left, &right).page_num()
            }
            _ => unreachable!(),
        };
        Ok(Tree {
            page_num: new_page_num,
            guard: writer,
        })
    }

    /// Finds the node to update and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `update`.
    fn update_helper(self, key: &[u8], val: &[u8]) -> Result<NodeEffect<'g>> {
        match Self::read(self.guard, self.page_num) {
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
        let writer = self.guard;
        let new_page_num = match self.delete_helper(key)? {
            NodeEffect::Empty => mmap::write_empty_leaf(writer),
            NodeEffect::Intact(root) => match node::sufficiency(&root) {
                Sufficiency::Empty => unreachable!(),
                Sufficiency::Underflow => match &root {
                    Node::Leaf(_) => root.page_num(),
                    Node::Internal(internal) => internal.get_child_pointer(0),
                },
                Sufficiency::Sufficient => root.page_num(),
            },
            NodeEffect::Split { left, right } => {
                Self::parent_of_split(writer, &left, &right).page_num()
            }
        };
        Ok(Tree {
            page_num: new_page_num,
            guard: writer,
        })
    }

    /// Finds the node to delete the key from and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `delete`.
    fn delete_helper(self, key: &[u8]) -> Result<NodeEffect<'g>> {
        match Self::read(self.guard, self.page_num) {
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
                match child.delete_helper(key)? {
                    NodeEffect::Empty => {
                        let effect = parent.merge_child_entries(
                            self.guard,
                            &[ChildEntry::Delete { i: child_idx }],
                        );
                        Ok(effect.into())
                    }
                    NodeEffect::Intact(child) => match node::sufficiency(&child) {
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
        parent: Internal<'g>,
        child: Node<'g>,
        child_idx: usize,
    ) -> NodeEffect<'g> {
        // Try to steal from or merge with a sibling.
        let sibling_idx = if child_idx > 0 {
            Some(child_idx - 1)
        } else if child_idx < parent.get_num_keys() - 1 {
            Some(child_idx + 1)
        } else {
            None
        };
        if let Some(sibling_idx) = sibling_idx {
            let sibling = Self::read(self.guard, parent.get_child_pointer(sibling_idx));
            let can_steal_or_merge = node::can_steal(&sibling, &child, sibling_idx < child_idx)
                || node::can_merge(&sibling, &child);
            if can_steal_or_merge {
                return self.steal_or_merge(parent, child, child_idx, sibling, sibling_idx);
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
    fn steal_or_merge(
        &self,
        parent: Internal<'g>,
        child: Node<'g>,
        child_idx: usize,
        sibling: Node<'g>,
        sibling_idx: usize,
    ) -> NodeEffect<'g> {
        let (mut left_idx, mut right_idx) = (sibling_idx, child_idx);
        let (left, right) = if sibling_idx < child_idx {
            (sibling, child)
        } else {
            (left_idx, right_idx) = (child_idx, sibling_idx);
            (child, sibling)
        };
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
                effect.into()
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
                effect.into()
            }
        }
    }

    /// Creates a new internal root node whose children are split nodes
    /// newly-created due to an operation on the tree.
    fn parent_of_split(writer: &'g Writer, left: &Node<'g>, right: &Node<'g>) -> Node<'g> {
        let keys = [left.get_key(0), right.get_key(0)];
        let child_pointers = [left.page_num(), right.page_num()];
        let root = Internal::parent_of_split(writer, keys, child_pointers);
        Node::Internal(root)
    }
}

#[cfg(test)]
impl<G: Guard> Tree<'_, G> {
    /// Gets the height of the tree.
    /// This performs a scan of the entire tree, so it's not really efficient.
    fn height(&self) -> Result<u32> {
        let node = Self::read(self.guard, self.page_num);
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
}

/// An in-order iterator over a tree.
pub struct InOrder<'q, 'g, G: Guard> {
    stack: Vec<(usize, Tree<'g, G>)>,
    end_bound: std::ops::Bound<&'q [u8]>,
}

impl<'g, G: Guard> Iterator for InOrder<'_, 'g, G> {
    type Item = (&'g [u8], &'g [u8]);
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((i, tree)) = self.stack.pop() {
            let node = Tree::read(tree.guard, tree.page_num);
            let n = node.get_num_keys();
            if i == n {
                continue;
            }
            match &node {
                Node::Leaf(leaf) => {
                    let key = leaf.get_key(i);
                    match self.end_bound {
                        std::ops::Bound::Included(end) => {
                            if key > end {
                                return None;
                            }
                        }
                        std::ops::Bound::Excluded(end) => {
                            if key >= end {
                                return None;
                            }
                        }
                        std::ops::Bound::Unbounded => {}
                    }
                    self.stack.push((i + 1, tree));
                    return Some((leaf.get_key(i), leaf.get_value(i)));
                }
                Node::Internal(internal) => {
                    let pn = internal.get_child_pointer(i);
                    let child = Tree {
                        page_num: pn,
                        guard: tree.guard,
                    };
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
    use std::sync::Arc;
    use std::{ops::Range, rc::Rc};

    use rand::rng;
    use rand::seq::SliceRandom;
    use tempfile::NamedTempFile;

    use crate::core::{
        consts,
        mmap::{Mmap, Store},
    };

    use super::*;

    fn new_test_store() -> (Arc<Store>, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        println!("Created temporary file {path:?}");
        let mmap = Mmap::open_or_create(path).unwrap();
        let store = Arc::new(Store::new(mmap));
        (store, temp_file)
    }

    fn u64_to_key(i: u64) -> [u8; consts::MAX_KEY_SIZE] {
        let mut key = [0u8; consts::MAX_KEY_SIZE];
        key[0..8].copy_from_slice(&i.to_be_bytes());
        key
    }

    fn insert_until_height(writer: Writer, height: u32) {
        assert_ne!(height, 0);
        let mut tree = Tree::new(&writer, writer.root_page());
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
        let root = tree.page_num;
        writer.flush(root);
    }

    fn insert_complete(store: &Arc<Store>, height: u32) {
        assert_ne!(height, 0);
        let mut i = 0u64;
        loop {
            let writer = store.writer();
            let tree = Tree::new(&writer, writer.root_page());
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
                writer.abort();
                break;
            }
            let new_page_num = new_tree.page_num;
            writer.flush(new_page_num);
        }
        // Sanity check.
        let reader = store.reader();
        let tree = Tree::new(&reader, reader.root_page());
        assert_eq!(tree.height().unwrap(), height);
    }

    #[test]
    fn insert_into_empty_tree() {
        let (store, _temp_file) = new_test_store();
        {
            let writer = store.writer();
            let tree = Tree::new(&writer, writer.root_page())
                .insert(&[1], &[1])
                .unwrap();
            let root = tree.page_num;
            writer.flush(root);
        }
        let reader = store.reader();
        let _ = Tree::new(&reader, reader.root_page());
    }

    #[test]
    fn insert_until_split() {
        let (store, _temp_file) = new_test_store();
        insert_until_height(store.writer(), 3);
        let reader = store.reader();
        let tree = Tree::new(&reader, reader.root_page());
        let root = Tree::read(&reader, tree.page_num);
        assert!(matches!(root, Node::Internal(_)));
        assert!(root.get_num_keys() >= 2);
        let got = tree
            .in_order_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect::<Vec<(Rc<[u8]>, Rc<[u8]>)>>();
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
        let (store, _temp_file) = new_test_store();
        insert_until_height(store.writer(), 2);
        let reader = store.reader();
        let tree = Tree::new(&reader, reader.root_page());
        let want = &u64_to_key(1u64);
        let got = tree.get(want).unwrap().unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn update_intact() {
        let (store, _temp_file) = new_test_store();
        let writer = store.writer();
        let tree = Tree::new(&writer, writer.root_page())
            .insert(&[0], &[0])
            .unwrap()
            .update(&[0], &[1])
            .unwrap();
        let root = tree.page_num;
        writer.flush(root);
        let reader = store.reader();
        let tree = Tree::new(&reader, reader.root_page());
        let got = tree.get(&[0]).unwrap().unwrap();
        assert_eq!(got, &[1]);
        assert_eq!(tree.height().unwrap(), 1);
    }

    #[test]
    fn update_split() {
        let (store, _temp_file) = new_test_store();
        insert_complete(&store, 2);
        let old_height = {
            let reader = store.reader();
            Tree::new(&reader, reader.root_page()).height().unwrap()
        };
        assert_eq!(old_height, 2);

        let writer = store.writer();
        let key = &u64_to_key(0);
        let new_value = &[1u8; consts::MAX_VALUE_SIZE];
        let tree = Tree::new(&writer, writer.root_page())
            .update(key, new_value)
            .unwrap();
        let got = tree.get(key).unwrap().unwrap();
        assert_eq!(got, new_value);
        assert_eq!(tree.height().unwrap(), old_height + 1);
    }

    #[test]
    fn delete_until_empty() {
        let (store, _temp_file) = new_test_store();
        insert_until_height(store.writer(), 3);
        let max = {
            let reader = store.reader();
            let max_key = Tree::new(&reader, reader.root_page())
                .in_order_iter()
                .last()
                .map(|(k, _)| k)
                .unwrap();
            u64::from_be_bytes([
                max_key[0], max_key[1], max_key[2], max_key[3], max_key[4], max_key[5], max_key[6],
                max_key[7],
            ])
        };

        let writer = store.writer();
        let mut tree = Tree::new(&writer, writer.root_page());
        let mut inds = (0..=max).collect::<Vec<_>>();
        inds.shuffle(&mut rng());
        for i in inds {
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
        let root = tree.page_num;
        writer.flush(root);

        let reader = store.reader();
        let tree = Tree::new(&reader, reader.root_page());
        assert_eq!(tree.height().unwrap(), 1);
        let root = Tree::read(&reader, tree.page_num);
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
        let (store, _temp_file) = new_test_store();
        let writer = store.writer();
        let tree = Tree::new(&writer, writer.root_page())
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
        assert_eq!(Tree::read(&writer, tree.page_num).get_num_keys(), 5);

        // Delete &[1]. This should trigger a split,
        // and the height should grow.
        let tree = tree.delete(&[1]).unwrap();
        assert!(tree.get(&[1]).unwrap().is_none());
        assert_eq!(tree.height().unwrap(), 3);
        assert_eq!(Tree::read(&writer, tree.page_num).get_num_keys(), 2);

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
        let (store, _temp_file) = new_test_store();
        let writer = store.writer();
        let tree = Tree::new(&writer, writer.root_page())
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
        assert_eq!(Tree::read(&writer, tree.page_num).get_num_keys(), 2);

        // Delete &[3]. This should trigger a split,
        // but the height shouldn't grow.
        let tree = tree.delete(&[3]).unwrap();
        assert!(tree.get(&[3]).unwrap().is_none());
        assert_eq!(tree.height().unwrap(), 3);
        assert_eq!(Tree::read(&writer, tree.page_num).get_num_keys(), 3);
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
        let (store, _temp_file) = new_test_store();
        let writer = store.writer();
        let mut tree = Tree::new(&writer, writer.root_page());
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
        assert_eq!(Tree::read(&writer, tree.page_num).get_num_keys(), 2);

        // Delete &[20]. This should trigger a split,
        // but the height shouldn't grow.
        // Neither should root change number of keys.
        let tree = tree.delete(&[20]).unwrap();
        assert!(tree.get(&[20]).unwrap().is_none());
        assert_eq!(tree.height().unwrap(), 4);
        assert_eq!(Tree::read(&writer, tree.page_num).get_num_keys(), 2);
    }

    #[test]
    fn test_in_order_range_iter() {
        let (store, _temp_file) = new_test_store();
        // Setup.
        {
            let writer = store.writer();
            let mut tree = Tree::new(&writer, writer.root_page());
            let mut inds = (1..=100).collect::<Vec<_>>();
            inds.shuffle(&mut rng());
            for i in inds {
                let x = u64_to_key(i);
                tree = tree.insert(&x, &x).unwrap();
            }
            let new_root_ptr = tree.page_num();
            writer.flush(new_root_ptr);
        }
        let reader = store.reader();
        let tree = Tree::new(&reader, reader.root_page());

        // Golang style table-driven tests.
        struct TestCase {
            name: &'static str,
            range: (
                std::ops::Bound<&'static [u8]>,
                std::ops::Bound<&'static [u8]>,
            ),
            want: Range<u64>,
        }
        impl Drop for TestCase {
            fn drop(&mut self) {
                for b in [self.range.0, self.range.1] {
                    match b {
                        std::ops::Bound::Included(b) => {
                            drop(unsafe { Box::from_raw(b.as_ptr() as *mut u8) });
                        }
                        std::ops::Bound::Excluded(b) => {
                            drop(unsafe { Box::from_raw(b.as_ptr() as *mut u8) });
                        }
                        _ => {}
                    }
                }
            }
        }
        let tests = [
            TestCase {
                name: "unbounded unbounded",
                range: (std::ops::Bound::Unbounded, std::ops::Bound::Unbounded),
                want: 1..101,
            },
            TestCase {
                name: "included included",
                range: (
                    std::ops::Bound::Included(Box::leak(Box::new(u64_to_key(5)))),
                    std::ops::Bound::Included(Box::leak(Box::new(u64_to_key(98)))),
                ),
                want: 5..99,
            },
            TestCase {
                name: "excluded included",
                range: (
                    std::ops::Bound::Excluded(Box::leak(Box::new(u64_to_key(5)))),
                    std::ops::Bound::Included(Box::leak(Box::new(u64_to_key(98)))),
                ),
                want: 6..99,
            },
            TestCase {
                name: "excluded excluded",
                range: (
                    std::ops::Bound::Excluded(Box::leak(Box::new(u64_to_key(5)))),
                    std::ops::Bound::Excluded(Box::leak(Box::new(u64_to_key(98)))),
                ),
                want: 6..98,
            },
            TestCase {
                name: "unbounded included",
                range: (
                    std::ops::Bound::Unbounded,
                    std::ops::Bound::Included(Box::leak(Box::new(u64_to_key(98)))),
                ),
                want: 1..99,
            },
            TestCase {
                name: "unbounded excluded",
                range: (
                    std::ops::Bound::Unbounded,
                    std::ops::Bound::Excluded(Box::leak(Box::new(u64_to_key(98)))),
                ),
                want: 1..98,
            },
            TestCase {
                name: "included unbounded",
                range: (
                    std::ops::Bound::Included(Box::leak(Box::new(u64_to_key(5)))),
                    std::ops::Bound::Unbounded,
                ),
                want: 5..101,
            },
            TestCase {
                name: "excluded unbounded",
                range: (
                    std::ops::Bound::Excluded(Box::leak(Box::new(u64_to_key(5)))),
                    std::ops::Bound::Unbounded,
                ),
                want: 6..101,
            },
            TestCase {
                name: "no overlap",
                range: (
                    std::ops::Bound::Excluded(Box::leak(Box::new(u64_to_key(200)))),
                    std::ops::Bound::Unbounded,
                ),
                want: 0..0,
            },
        ];
        for test in tests {
            let got = tree
                .in_order_range_iter(&test.range)
                .map(|(k, _)| u64::from_be_bytes([k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]]))
                .collect::<Vec<_>>();
            let want = test.want.clone().collect::<Vec<_>>();
            assert_eq!(got, want, "Test case \"{}\" failed", test.name);
        }
    }
}
