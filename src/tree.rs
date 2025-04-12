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

pub mod consts;
mod error;
mod node;
mod page_store;

pub use error::TreeError;
use node::{ChildEntry, Internal, Leaf, Node, NodeEffect, Sufficiency};
use page_store::PageStore;
use std::rc::Rc;

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
pub struct Tree<P: PageStore> {
    root: Node<P>,
    store: P,
}

impl<P: PageStore> Tree<P> {
    /// Creates an new empty B+ tree.
    pub fn new(store: P) -> Result<Self> {
        let root = Node::Leaf(Leaf::new(store.clone())?);
        store.flush()?;
        Ok(Self { root, store })
    }

    pub fn read(store: P, page_num: usize) -> Result<Self> {
        let root = Node::read(store.clone(), page_num)?;
        Ok(Self { root, store })
    }

    pub fn page_num(&self) -> usize {
        self.root.page_num()
    }

    fn read_child(&self, child_num: usize) -> Result<Self> {
        Self::read(self.store.clone(), child_num)
    }

    /// Flushes the pending writes and makes available the modified copy of
    /// the tree while leaving the previous tree intact.
    fn flush_modified_copy(&self, new_root: Node<P>) -> Result<Self> {
        self.store.flush()?;
        Ok(Self {
            root: new_root,
            store: self.store.clone(),
        })
    }

    /// Gets the value corresponding to the key.
    pub fn get(&self, key: &[u8]) -> Result<Option<Rc<[u8]>>> {
        match &self.root {
            Node::Internal(root) => {
                let child_idx = root.find(key);
                let child_num = root.get_child_pointer(child_idx);
                let child = self.read_child(child_num)?;
                child.get(key)
            }
            Node::Leaf(root) => Ok(root.find(key).map(|v| v.into())),
        }
    }

    /// Inserts a key-value pair.
    pub fn insert(&self, key: &[u8], val: &[u8]) -> Result<Self> {
        let new_root = match self.insert_helper(key, val)? {
            NodeEffect::Intact(new_root) => new_root,
            NodeEffect::Split { left, right } => self.parent_of_split(left, right)?,
            _ => unreachable!(),
        };
        self.flush_modified_copy(new_root)
    }

    /// Finds the node to insert into and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `insert`.
    fn insert_helper(&self, key: &[u8], val: &[u8]) -> Result<NodeEffect<P>> {
        match &self.root {
            // Base case
            Node::Leaf(leaf) => Ok(leaf.insert(key, val)?.into()),
            // Recursive case
            Node::Internal(internal) => {
                // Find which child to recursively insert into.
                let child_idx = internal.find(key);
                let child_num = internal.get_child_pointer(child_idx);
                let child = self.read_child(child_num)?;
                let child_entries = child.insert_helper(key, val)?.child_entries(child_idx);
                let effect = internal.merge_child_entries(child_entries.as_ref())?;
                Ok(effect.into())
            }
        }
    }

    /// Updates the value corresponding to a key.
    pub fn update(&self, key: &[u8], val: &[u8]) -> Result<Self> {
        let new_root = match self.update_helper(key, val)? {
            NodeEffect::Intact(root) => root,
            NodeEffect::Split { left, right } => self.parent_of_split(left, right)?,
            _ => unreachable!(),
        };
        self.flush_modified_copy(new_root)
    }

    /// Finds the node to update and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `update`.
    fn update_helper(&self, key: &[u8], val: &[u8]) -> Result<NodeEffect<P>> {
        match &self.root {
            // Base case
            Node::Leaf(leaf) => Ok(leaf.update(key, val)?.into()),
            // Recursive case
            Node::Internal(internal) => {
                // Find which child to recursively update at.
                let child_idx = internal.find(key);
                let child_num = internal.get_child_pointer(child_idx);
                let child = self.read_child(child_num)?;
                let child_entries = child.update_helper(key, val)?.child_entries(child_idx);
                let effect = internal.merge_child_entries(child_entries.as_ref())?;
                Ok(effect.into())
            }
        }
    }

    /// Deletes a key and its corresponding value.
    pub fn delete(&self, key: &[u8]) -> Result<Self> {
        match self.delete_helper(key)? {
            NodeEffect::Empty => Self::new(self.store.clone()),
            NodeEffect::Intact(root) => match node::sufficiency(&root) {
                Sufficiency::Empty => unreachable!(),
                Sufficiency::Underflow => match &root {
                    Node::Leaf(_) => self.flush_modified_copy(root),
                    Node::Internal(internal) => {
                        let page_num = internal.get_child_pointer(0);
                        // Need to flush before we can read a recently written page.
                        self.store.flush()?;
                        self.read_child(page_num)
                    }
                },
                Sufficiency::Sufficient => self.flush_modified_copy(root),
            },
            NodeEffect::Split { left, right } => {
                self.flush_modified_copy(self.parent_of_split(left, right)?)
            }
        }
    }

    /// Finds the node to delete the key from and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `delete`.
    fn delete_helper(&self, key: &[u8]) -> Result<NodeEffect<P>> {
        match &self.root {
            // Base case
            Node::Leaf(leaf) => Ok(leaf.delete(key)?.into()),
            // Recursive case
            Node::Internal(parent) => {
                // Find which child to recursively delete from.
                let child_idx = parent.find(key);
                let child_num = parent.get_child_pointer(child_idx);
                let child = self.read_child(child_num)?;
                match child.delete_helper(key)? {
                    NodeEffect::Empty => {
                        let effect =
                            parent.merge_child_entries(&[ChildEntry::Delete { i: child_idx }])?;
                        Ok(effect.into())
                    }
                    NodeEffect::Intact(child) => match node::sufficiency(&child) {
                        Sufficiency::Empty => unreachable!(),
                        Sufficiency::Underflow => self.try_fix_underflow(parent, child, child_idx),
                        Sufficiency::Sufficient => {
                            let child_entries = NodeEffect::Intact(child).child_entries(child_idx);
                            let effect = parent.merge_child_entries(child_entries.as_ref())?;
                            Ok(effect.into())
                        }
                    },
                    NodeEffect::Split { left, right } => {
                        let effect = parent.merge_child_entries(&[
                            ChildEntry::Update {
                                i: child_idx,
                                key: left.get_key(0).into(),
                                page_num: left.page_num(),
                            },
                            ChildEntry::Insert {
                                key: right.get_key(0).into(),
                                page_num: right.page_num(),
                            },
                        ])?;
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
        parent: &Internal<P>,
        child: Node<P>,
        child_idx: usize,
    ) -> Result<NodeEffect<P>> {
        // Try to steal from or merge with the left sibling.
        if child_idx > 0 {
            if let Some(effect) =
                self.try_steal_or_merge(parent, &child, child_idx, child_idx - 1)?
            {
                return Ok(effect);
            }
        }
        // Try to do so with the right sibling.
        if child_idx < parent.get_num_keys() - 1 {
            if let Some(effect) =
                self.try_steal_or_merge(parent, &child, child_idx, child_idx + 1)?
            {
                return Ok(effect);
            }
        }

        // Leave as underflow.
        Ok(parent
            .merge_child_entries(&[ChildEntry::Update {
                i: child_idx,
                key: child.get_key(0).into(),
                page_num: child.page_num(),
            }])?
            .into())
    }

    /// Fixes the underflow of `child` by stealing from or merging from
    /// one of its direct siblings.
    fn try_steal_or_merge(
        &self,
        parent: &Internal<P>,
        child: &Node<P>,
        child_idx: usize,
        sibling_idx: usize,
    ) -> Result<Option<NodeEffect<P>>> {
        let sibling_num = parent.get_child_pointer(sibling_idx);
        let sibling = &Node::read(self.store.clone(), sibling_num)?;

        let can_steal_or_merge = node::can_steal(sibling, child, sibling_idx < child_idx)
            || node::can_merge(sibling, child);
        if !can_steal_or_merge {
            return Ok(None);
        }

        let (mut left, mut right) = (sibling, child);
        let (mut left_idx, mut right_idx) = (sibling_idx, child_idx);
        if sibling_idx > child_idx {
            (right, left) = (sibling, child);
            (right_idx, left_idx) = (sibling_idx, child_idx);
        }
        match node::steal_or_merge(left, right)? {
            NodeEffect::Empty => unreachable!(),
            NodeEffect::Intact(child) => {
                // merged
                let effect = parent.merge_child_entries(&[
                    ChildEntry::Update {
                        i: left_idx,
                        key: child.get_key(0).into(),
                        page_num: child.page_num(),
                    },
                    ChildEntry::Delete { i: right_idx },
                ])?;
                Ok(Some(effect.into()))
            }
            NodeEffect::Split { left, right } => {
                // stolen
                let effect = parent.merge_child_entries(&[
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
                ])?;
                Ok(Some(effect.into()))
            }
        }
    }

    /// Creates a new internal root node whose children are split nodes
    /// newly-created due to an operation on the tree.
    fn parent_of_split(&self, left: Node<P>, right: Node<P>) -> Result<Node<P>> {
        let keys = [left.get_key(0), right.get_key(0)];
        let child_pointers = [left.page_num(), right.page_num()];
        let root = Internal::parent_of_split(keys, child_pointers, self.store.clone())?;
        Ok(Node::Internal(root))
    }
}

#[cfg(test)]
impl<P: PageStore> Tree<P> {
    /// Gets the height of the tree.
    /// This performs a scan of the entire tree, so it's not really efficient.
    fn height(&self) -> Result<u32> {
        match &self.root {
            Node::Leaf(_) => Ok(1),
            Node::Internal(root) => {
                assert!(root.get_num_keys() >= 2);
                let mut height: Option<u32> = None;
                for (_, pn) in root.iter() {
                    let child = self.read_child(pn)?;
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
    fn inorder_iter(&self) -> InOrder<P> {
        let copy = Self::read(self.store.clone(), self.page_num()).unwrap();
        InOrder {
            stack: vec![(0, Rc::new(copy))],
        }
    }
}

#[cfg(test)]
struct InOrder<P: PageStore> {
    stack: Vec<(usize, Rc<Tree<P>>)>,
}

#[cfg(test)]
impl<P: PageStore> Iterator for InOrder<P> {
    type Item = (Rc<[u8]>, Rc<[u8]>);
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((i, tree)) = self.stack.pop() {
            let n = tree.root.get_num_keys();
            if i == n {
                continue;
            }
            match &tree.root {
                Node::Leaf(leaf) => {
                    self.stack.push((i + 1, tree.clone()));
                    return Some((leaf.get_key(i).into(), leaf.get_value(i).into()));
                }
                Node::Internal(internal) => {
                    let pn = internal.get_child_pointer(i);
                    let child = Rc::new(tree.read_child(pn).unwrap());
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
    use page_store::InMemory;

    use super::*;

    fn test_store() -> InMemory {
        InMemory::new()
    }

    fn insert_until_height(height: u32) -> Tree<InMemory> {
        assert_ne!(height, 0);
        let mut tree = Tree::new(test_store()).unwrap();
        let mut i = 0u64;
        loop {
            if tree.height().unwrap() == height {
                break;
            }
            let x = i.to_be_bytes();
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
                matches!(found, Some(v) if v == x.into()),
                "did not find val for {i}"
            );
            i += 1;
        }
        tree
    }

    fn insert_complete(height: u32) -> Tree<InMemory> {
        assert_ne!(height, 0);
        let mut tree = Tree::new(test_store()).unwrap();
        let mut i = 0u64;
        loop {
            let x = i.to_be_bytes();
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
                matches!(found, Some(v) if v == x.into()),
                "did not find val for {i}"
            );
            i += 1;
            if new_tree.height().unwrap() > height {
                break;
            }
            tree = new_tree;
        }
        tree
    }

    #[test]
    fn insert_into_empty_tree() {
        let ps = InMemory::new();
        let tree = Tree::new(ps.clone()).unwrap();
        let tree = tree.insert(&[1], &[1]).unwrap();
        ps.flush().unwrap();
        ps.read_page(tree.page_num()).unwrap();
    }

    #[test]
    fn insert_until_split() {
        let tree = insert_until_height(3);
        assert!(matches!(tree.root, Node::Internal(_)));
        assert!(tree.root.get_num_keys() >= 2);
        let got = tree.inorder_iter().collect::<Vec<_>>();
        let want = (0..got.len() as u64)
            .map(|i| {
                let x = i.to_be_bytes();
                let x: Rc<[u8]> = x.into();
                (x.clone(), x.clone())
            })
            .collect::<Vec<_>>();
        assert_eq!(got, want);
    }

    #[test]
    fn get() {
        let got = insert_until_height(2)
            .get(&1u64.to_be_bytes())
            .unwrap()
            .unwrap();
        assert_eq!(got, 1u64.to_be_bytes().into());
    }

    #[test]
    fn update_intact() {
        let tree = Tree::new(test_store())
            .unwrap()
            .insert(&[0], &[0])
            .unwrap()
            .update(&[0], &[1])
            .unwrap();
        let got = tree.get(&[0]).unwrap().unwrap();
        assert_eq!(got, [1].into());
        assert_eq!(tree.height().unwrap(), 1);
    }

    #[test]
    fn update_split() {
        let old_tree = insert_complete(2);
        let new_tree = old_tree
            .update(&0u64.to_be_bytes(), &[0u8; consts::MAX_VALUE_SIZE])
            .unwrap();
        let got = new_tree.get(&0u64.to_be_bytes()).unwrap().unwrap();
        assert_eq!(got, [0u8; consts::MAX_VALUE_SIZE].into());
        assert_eq!(new_tree.height().unwrap(), old_tree.height().unwrap() + 1);
    }

    #[test]
    fn delete_until_empty() {
        let tree = insert_until_height(3);
        let max_key = tree.inorder_iter().last().map(|(k, _)| k).unwrap();
        let max = u64::from_be_bytes([
            max_key[0], max_key[1], max_key[2], max_key[3], max_key[4], max_key[5], max_key[6],
            max_key[7],
        ]);
        let mut tree = tree;
        for i in 0..=max {
            let key = &i.to_be_bytes();
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
        assert_eq!(tree.height().unwrap(), 1);
        assert_eq!(tree.root.get_num_keys(), 0);
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
        let tree = Tree::new(test_store())
            .unwrap()
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
        assert_eq!(tree.root.get_num_keys(), 5);

        // Delete &[1]. This should trigger a split,
        // and the height should grow.
        let tree = tree.delete(&[1]).unwrap();
        assert!(tree.get(&[1]).unwrap().is_none());
        assert_eq!(tree.height().unwrap(), 3);
        assert_eq!(tree.root.get_num_keys(), 2);

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
        let tree = Tree::new(test_store())
            .unwrap()
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
        assert_eq!(tree.root.get_num_keys(), 2);

        // Delete &[3]. This should trigger a split,
        // but the height shouldn't grow.
        let tree = tree.delete(&[3]).unwrap();
        assert!(tree.get(&[3]).unwrap().is_none());
        assert_eq!(tree.height().unwrap(), 3);
        assert_eq!(tree.root.get_num_keys(), 3);
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
        let mut tree = Tree::new(test_store()).unwrap();
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
        assert_eq!(tree.root.get_num_keys(), 2);

        // Delete &[20]. This should trigger a split,
        // but the height shouldn't grow.
        // Neither should root change number of keys.
        let tree = tree.delete(&[20]).unwrap();
        assert!(tree.get(&[20]).unwrap().is_none());
        assert_eq!(tree.height().unwrap(), 4);
        assert_eq!(tree.root.get_num_keys(), 2);
    }
}
