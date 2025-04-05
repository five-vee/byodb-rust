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

mod buffer_store;
pub mod consts;
mod error;
mod node;
mod page_store;

pub use error::TreeError;
use node::{ChildEntry, Internal, Leaf, Node, NodeEffect, Sufficiency};
use page_store::{InMemory, PageStore};
use std::rc::Rc;

type Result<T> = std::result::Result<T, TreeError>;

/// An enum representing the effect of a tree operation.
enum TreeEffect<P: PageStore = InMemory> {
    /// A tree with 0 keys after a delete was performed on it.
    /// This is a special-case of `Underflow` done to avoid unnecessary
    /// page allocations, since empty non-root nodes aren't allowed.
    Empty,
    /// A newly created tree that remained  "intact", i.e. it did not split.
    Intact(Tree<P>),
    /// The left and right splits of a tree that was created.
    ///
    /// The left and right trees are the same type.
    Split { left: Tree<P>, right: Tree<P> },
}

impl<P: PageStore> TreeEffect<P> {
    /// Converts the tree(s) created during an operation into child
    /// entries of a B+ tree internal node.
    ///
    /// `i` is the index in the internal that the operation was performed on.
    fn child_entries(self, i: usize) -> Rc<[ChildEntry]> {
        match self {
            TreeEffect::Empty => Rc::new([]),
            TreeEffect::Intact(tree) => [ChildEntry::Update {
                i,
                key: tree.root.get_key(0).into(),
                page_num: tree.page_num,
            }]
            .into(),
            TreeEffect::Split { left, right } => [
                ChildEntry::Update {
                    i,
                    key: left.root.get_key(0).into(),
                    page_num: left.page_num,
                },
                ChildEntry::Insert {
                    key: right.root.get_key(0).into(),
                    page_num: right.page_num,
                },
            ]
            .into(),
        }
    }
}

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
#[derive(Clone)]
pub struct Tree<P: PageStore = InMemory> {
    root: Node<P::B>,
    page_num: usize,
    page_store: P,
}

impl Tree<InMemory> {
    pub fn new() -> Result<Self> {
        Self::new_in(InMemory::new())
    }
}

impl<P: PageStore> Tree<P> {
    /// Creates an new empty B+ tree.
    pub fn new_in(page_store: P) -> Result<Self> {
        let root = Node::Leaf(Leaf::new(page_store.buffer_store()));
        let page_num = page_store.write_page(&root)?;
        Ok(Self {
            root,
            page_num,
            page_store,
        })
    }

    /// Gets the value corresponding to the key.
    pub fn get(&self, key: &[u8]) -> Result<Option<Rc<[u8]>>> {
        match &self.root {
            Node::Internal(root) => {
                let child_idx = root.find(key);
                let child_num = root.get_child_pointer(child_idx);
                let child = self.page_store.read_page(child_num)?;
                let child = Self {
                    root: child,
                    page_num: child_num,
                    page_store: self.page_store.clone(),
                };
                child.get(key)
            }
            Node::Leaf(root) => Ok(root.find(key).map(|v| v.into())),
        }
    }

    /// Inserts a key-value pair.
    pub fn insert(&self, key: &[u8], val: &[u8]) -> Result<Self> {
        match self.insert_helper(key, val)? {
            TreeEffect::Intact(tree) => Ok(tree),
            TreeEffect::Split { left, right } => self.parent_of_split(left, right),
            _ => unreachable!(),
        }
    }

    /// Finds the node to insert into and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `insert`.
    fn insert_helper(&self, key: &[u8], val: &[u8]) -> Result<TreeEffect<P>> {
        match &self.root {
            // Base case
            Node::Leaf(leaf) => Ok(self.alloc(leaf.insert(key, val)?.into())?),
            // Recursive case
            Node::Internal(internal) => {
                // Find which child to recursively insert into.
                let child_idx = internal.find(key);
                let child_num = internal.get_child_pointer(child_idx);
                let child = self.page_store.read_page(child_num)?;
                let child = Self {
                    root: child,
                    page_num: child_num,
                    page_store: self.page_store.clone(),
                };
                let child_entries = child.insert_helper(key, val)?.child_entries(child_idx);
                let effect = internal.merge_child_entries(child_entries.as_ref())?;
                Ok(self.alloc(effect.into())?)
            }
        }
    }

    /// Updates the value corresponding to a key.
    pub fn update(&self, key: &[u8], val: &[u8]) -> Result<Self> {
        match self.update_helper(key, val)? {
            TreeEffect::Intact(tree) => Ok(tree),
            TreeEffect::Split { left, right } => self.parent_of_split(left, right),
            _ => unreachable!(),
        }
    }

    /// Finds the node to update and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `update`.
    fn update_helper(&self, key: &[u8], val: &[u8]) -> Result<TreeEffect<P>> {
        match &self.root {
            // Base case
            Node::Leaf(leaf) => self.alloc(leaf.update(key, val)?.into()),
            // Recursive case
            Node::Internal(internal) => {
                // Find which child to recursively update at.
                let child_idx = internal.find(key);
                let child_num = internal.get_child_pointer(child_idx);
                let child = self.page_store.read_page(child_num)?;
                let child = Self {
                    root: child,
                    page_num: child_num,
                    page_store: self.page_store.clone(),
                };
                let child_entries = child.update_helper(key, val)?.child_entries(child_idx);
                let effect = internal.merge_child_entries(child_entries.as_ref())?;
                self.alloc(effect.into())
            }
        }
    }

    /// Deletes a key and its corresponding value.
    pub fn delete(&self, key: &[u8]) -> Result<Self> {
        match self.delete_helper(key)? {
            TreeEffect::Empty => {
                let root = Node::Leaf(Leaf::new(self.page_store.buffer_store()));
                let page_num = self.page_store.write_page(&root)?;
                Ok(Self {
                    root,
                    page_num,
                    page_store: self.page_store.clone(),
                })
            }
            TreeEffect::Intact(tree) => match node::sufficiency(&tree.root) {
                Sufficiency::Empty => unreachable!(),
                Sufficiency::Underflow => match &tree.root {
                    Node::Leaf(_) => Ok(tree),
                    Node::Internal(internal) => {
                        let page_num = internal.get_child_pointer(0);
                        let child = self.page_store.read_page(page_num)?;
                        Ok(Self {
                            root: child,
                            page_num,
                            page_store: self.page_store.clone(),
                        })
                    }
                },
                Sufficiency::Sufficient => Ok(tree),
            },
            TreeEffect::Split { left, right } => self.parent_of_split(left, right),
        }
    }

    /// Finds the node to delete the key from and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `delete`.
    fn delete_helper(&self, key: &[u8]) -> Result<TreeEffect<P>> {
        match &self.root {
            // Base case
            Node::Leaf(leaf) => self.alloc(leaf.delete(key)?.into()),
            // Recursive case
            Node::Internal(parent) => {
                // Find which child to recursively delete from.
                let child_idx = parent.find(key);
                let child_num = parent.get_child_pointer(child_idx);
                let child = self.page_store.read_page(child_num)?;
                let child = Self {
                    root: child,
                    page_num: child_num,
                    page_store: self.page_store.clone(),
                };
                match child.delete_helper(key)? {
                    TreeEffect::Empty => {
                        let effect =
                            parent.merge_child_entries(&[ChildEntry::Delete { i: child_idx }])?;
                        self.alloc(effect.into())
                    }
                    TreeEffect::Intact(child) => match node::sufficiency(&child.root) {
                        Sufficiency::Empty => unreachable!(),
                        Sufficiency::Underflow => self.try_fix_underflow(parent, child, child_idx),
                        Sufficiency::Sufficient => {
                            let child_entries = TreeEffect::Intact(child).child_entries(child_idx);
                            let effect = parent.merge_child_entries(child_entries.as_ref())?;
                            self.alloc(effect.into())
                        }
                    },
                    TreeEffect::Split { left, right } => {
                        let effect = parent.merge_child_entries(&[
                            ChildEntry::Update {
                                i: child_idx,
                                key: left.root.get_key(0).into(),
                                page_num: left.page_num,
                            },
                            ChildEntry::Insert {
                                key: right.root.get_key(0).into(),
                                page_num: right.page_num,
                            },
                        ])?;
                        self.alloc(effect.into())
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
        parent: &Internal<P::B>,
        child: Self,
        child_idx: usize,
    ) -> Result<TreeEffect<P>> {
        // Try to steal from or merge with the left sibling.
        if child_idx > 0 {
            if let Some(effect) =
                self.try_steal_or_merge(parent, &child.root, child_idx, child_idx - 1)?
            {
                return Ok(effect);
            }
        }
        // Try to do so with the right sibling.
        if child_idx < parent.get_num_keys() - 1 {
            if let Some(effect) =
                self.try_steal_or_merge(parent, &child.root, child_idx, child_idx + 1)?
            {
                return Ok(effect);
            }
        }

        // Leave as underflow.
        self.alloc(
            parent
                .merge_child_entries(&[ChildEntry::Update {
                    i: child_idx,
                    key: child.root.get_key(0).into(),
                    page_num: child.page_num,
                }])?
                .into(),
        )
    }

    /// Fixes the underflow of `child` by stealing from or merging from
    /// one of its direct siblings.
    fn try_steal_or_merge(
        &self,
        parent: &Internal<P::B>,
        child: &Node<P::B>,
        child_idx: usize,
        sibling_idx: usize,
    ) -> Result<Option<TreeEffect<P>>> {
        let sibling_num = parent.get_child_pointer(sibling_idx);
        let sibling = &self.page_store.read_page(sibling_num)?;

        if !node::can_steal(sibling, child, sibling_idx < child_idx)
            && !node::can_merge(sibling, child)
        {
            return Ok(None);
        }

        let (mut left, mut right) = (sibling, child);
        let (mut left_idx, mut right_idx) = (sibling_idx, child_idx);
        if sibling_idx > child_idx {
            (right, left) = (sibling, child);
            (right_idx, left_idx) = (sibling_idx, child_idx);
        }
        match self.alloc(node::steal_or_merge(left, right)?)? {
            TreeEffect::Empty => unreachable!(),
            TreeEffect::Intact(child) => {
                // merged
                let effect = parent.merge_child_entries(&[
                    ChildEntry::Update {
                        i: left_idx,
                        key: child.root.get_key(0).into(),
                        page_num: child.page_num,
                    },
                    ChildEntry::Delete { i: right_idx },
                ])?;
                Ok(Some(self.alloc(effect.into())?))
            }
            TreeEffect::Split { left, right } => {
                // stolen
                let effect = parent.merge_child_entries(&[
                    ChildEntry::Update {
                        i: left_idx,
                        key: left.root.get_key(0).into(),
                        page_num: left.page_num,
                    },
                    ChildEntry::Update {
                        i: right_idx,
                        key: right.root.get_key(0).into(),
                        page_num: right.page_num,
                    },
                ])?;
                Ok(Some(self.alloc(effect.into())?))
            }
        }
    }

    /// Returns a new internal root node whose children are split nodes
    /// newly-created due to an operation on the tree.
    fn parent_of_split(&self, left: Self, right: Self) -> Result<Self> {
        let keys = [left.root.get_key(0), right.root.get_key(0)];
        let child_pointers = [left.page_num, right.page_num];
        let root = Internal::parent_of_split(keys, child_pointers, self.page_store.buffer_store())?;
        let root = Node::Internal(root);
        let page_num = self.page_store.write_page(&root)?;
        Ok(Self {
            root,
            page_num,
            page_store: self.page_store.clone(),
        })
    }

    /// Allocates pages for the in-memory nodes created during an upsert.
    fn alloc(&self, effect: NodeEffect<P::B>) -> Result<TreeEffect<P>> {
        match effect {
            NodeEffect::Empty => Ok(TreeEffect::Empty),
            NodeEffect::Intact(root) => {
                let page_num = self.page_store.write_page(&root)?;
                Ok(TreeEffect::Intact(Self {
                    root,
                    page_num,
                    page_store: self.page_store.clone(),
                }))
            }
            NodeEffect::Split { left, right } => {
                let left_page_num = self.page_store.write_page(&left)?;
                let right_page_num = self.page_store.write_page(&right)?;
                let left = Self {
                    root: left,
                    page_num: left_page_num,
                    page_store: self.page_store.clone(),
                };
                let right = Self {
                    root: right,
                    page_num: right_page_num,
                    page_store: self.page_store.clone(),
                };
                Ok(TreeEffect::Split { left, right })
            }
        }
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
                    let child = self.page_store.read_page(pn)?;
                    let child = Self {
                        root: child,
                        page_num: pn,
                        page_store: self.page_store.clone(),
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
    fn inorder_iter(&self) -> InOrder<P> {
        InOrder {
            stack: vec![(0, Rc::new(self.clone()))],
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
                    let child = tree.page_store.read_page(pn).unwrap();
                    let child = Rc::new(Tree {
                        root: child,
                        page_num: pn,
                        page_store: tree.page_store.clone(),
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
    use super::*;
    use page_store::InMemory;

    fn insert_until_height(height: u32) -> Tree {
        assert_ne!(height, 0);
        let mut tree = Tree::new().unwrap();
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

    fn insert_complete(height: u32) -> Tree {
        assert_ne!(height, 0);
        let mut tree = Tree::new().unwrap();
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
        let tree = Tree::new_in(ps.clone()).unwrap();
        let tree = tree.insert(&[1], &[1]).unwrap();
        ps.read_page(tree.page_num).unwrap();
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
        let tree = Tree::new()
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
        let tree = Tree::new()
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
        let tree = Tree::new()
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
        let mut tree = Tree::new().unwrap();
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
