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
mod error;
mod node;
mod page_store;

pub use error::TreeError;
use node::{ChildEntry, Internal, Leaf, Node, NodeEffect, Sufficiency};
pub use node::{MAX_KEY_SIZE, MAX_VALUE_SIZE};
use page_store::PageStore;
use std::{cmp::max, rc::Rc};

type Result<T> = std::result::Result<T, TreeError>;

/// An enum representing the effect of a tree operation.
enum TreeEffect<P: PageStore> {
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
pub struct Tree<P: PageStore> {
    root: Node<P::B>,
    page_num: u64,
    buffer_store: P::B,
    page_store: P,
}

impl<P: PageStore> Tree<P> {
    /// Creates an new empty B+ tree.
    pub fn new(buffer_store: P::B, page_store: P) -> Result<Self> {
        let root = Node::Leaf(Leaf::new(&buffer_store));
        let page_num = page_store.write_page(&root)?;
        Ok(Self {
            root,
            page_num,
            buffer_store,
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
                    buffer_store: self.buffer_store.clone(),
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
                    buffer_store: self.buffer_store.clone(),
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
                    buffer_store: self.buffer_store.clone(),
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
                let root = Node::Leaf(Leaf::new(&self.buffer_store));
                let page_num = self.page_store.write_page(&root)?;
                Ok(Self {
                    root,
                    page_num,
                    buffer_store: self.buffer_store.clone(),
                    page_store: self.page_store.clone(),
                })
            }
            TreeEffect::Intact(tree) => Ok(tree),
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
                    buffer_store: self.buffer_store.clone(),
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
                            let effect = parent.merge_child_entries(&[ChildEntry::Update {
                                i: child_idx,
                                key: child.root.get_key(0).into(),
                                page_num: child.page_num,
                            }])?;
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

    /// Tries to fix the underflow of `child` by stealing from or merging from
    /// one of its direct siblings
    /// (either at `child_idx - 1` or `child_idx + 1`) within the
    /// parent (internal) node.
    ///
    /// If the fix failed, just leaves the child in an underflow state.
    fn try_fix_underflow(
        &self,
        parent: &Internal<P::B>,
        child: Self,
        child_idx: usize,
    ) -> Result<TreeEffect<P>> {
        // Try stealing or merging the left sibling.
        if child_idx > 0 {
            return self.steal_or_merge(parent, &child.root, child_idx, child_idx - 1);
        }
        // Try stealing or merging the right sibling.
        if child_idx < parent.get_num_keys() - 1 {
            return self.steal_or_merge(parent, &child.root, child_idx, child_idx + 1);
        }

        // 5. Just leave child in underflow.
        // Note: This only happens in leaf nodes and not internal nodes.
        // We guarantee that an internal node can fit at least 3 keys,
        // and that an internal node is sufficient if it has at least 2 keys.
        // The child being in underflow means it has only 1 key.
        // If the left/right internal node doesn't have enough to steal from,
        // then the merge of left/right and the child will produce a new node
        // with 3 keys, which will not exceed the page size.
        // Leaf nodes, OTOH, can remain underflow due to variable-length
        // keys and values.
        let effect = parent.merge_child_entries(&[ChildEntry::Update {
            i: child_idx,
            key: child.root.get_key(0).into(),
            page_num: child.page_num,
        }])?;
        self.alloc(effect.into())
    }

    /// Tries to fix the underflow of `child` by stealing from or merging from
    /// one of its direct siblings.
    fn steal_or_merge(
        &self,
        parent: &Internal<P::B>,
        child: &Node<P::B>,
        child_idx: usize,
        sibling_idx: usize,
    ) -> Result<TreeEffect<P>> {
        let sibling_num = parent.get_child_pointer(sibling_idx);
        let sibling = self.page_store.read_page(sibling_num)?;
        let mut left = &sibling;
        let mut right = child;
        if sibling_idx > child_idx {
            (left, right) = (right, left);
        }
        self.alloc(node::steal_or_merge(left, right)?)
    }

    /// Returns a new internal root node whose children are split nodes
    /// newly-created due to an operation on the tree.
    fn parent_of_split(&self, left: Self, right: Self) -> Result<Self> {
        let keys = [left.root.get_key(0), right.root.get_key(0)];
        let child_pointers = [left.page_num, right.page_num];
        let root = Internal::parent_of_split(keys, child_pointers, &self.buffer_store)?;
        let root = Node::Internal(root);
        let page_num = self.page_store.write_page(&root)?;
        Ok(Self {
            root,
            page_num,
            buffer_store: self.buffer_store.clone(),
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
                    buffer_store: self.buffer_store.clone(),
                    page_store: self.page_store.clone(),
                }))
            }
            NodeEffect::Split { left, right } => {
                let left_page_num = self.page_store.write_page(&left)?;
                let right_page_num = self.page_store.write_page(&right)?;
                let left = Self {
                    root: left,
                    page_num: left_page_num,
                    buffer_store: self.buffer_store.clone(),
                    page_store: self.page_store.clone(),
                };
                let right = Self {
                    root: right,
                    page_num: right_page_num,
                    buffer_store: self.buffer_store.clone(),
                    page_store: self.page_store.clone(),
                };
                Ok(TreeEffect::Split { left, right })
            }
        }
    }

    /// Gets the height of the tree.
    /// This performs a scan of the entire tree, so it's not really efficient.
    #[allow(dead_code)]
    fn height(&self) -> Result<u32> {
        match &self.root {
            Node::Leaf(_) => Ok(1),
            Node::Internal(root) => {
                let mut height = 1;
                for (_, pn) in root.iter() {
                    let child = self.page_store.read_page(pn)?;
                    let child = Self {
                        root: child,
                        page_num: pn,
                        buffer_store: self.buffer_store.clone(),
                        page_store: self.page_store.clone(),
                    };
                    height = max(height, 1 + child.height()?);
                }
                Ok(height)
            }
        }
    }

    /// Iterates through the tree in-order.
    /// This is very slow, so be careful.
    #[allow(dead_code)]
    fn inorder_iter(&self) -> InOrder<P> {
        InOrder {
            stack: vec![(0, Rc::new(self.clone()))],
        }
    }
}

struct InOrder<P: PageStore> {
    stack: Vec<(usize, Rc<Tree<P>>)>,
}

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
                        buffer_store: tree.buffer_store.clone(),
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
    use std::sync::OnceLock;

    use super::*;
    use buffer_store::Heap;
    use page_store::InMemory;

    static TEST_BUFFER_STORE: Heap = Heap {};
    static TEST_PAGE_STORE: OnceLock<InMemory> = OnceLock::new();

    fn buffer_store() -> Heap {
        TEST_BUFFER_STORE.clone()
    }

    fn page_store() -> InMemory {
        TEST_PAGE_STORE.get_or_init(InMemory::new).clone()
    }

    #[test]
    fn insert_into_empty_tree() {
        let ps = page_store();
        let tree = Tree::new(buffer_store(), ps.clone()).unwrap();
        let tree = tree.insert(&[1], &[1]).unwrap();
        ps.read_page(tree.page_num).unwrap();
    }

    #[test]
    fn insert_until_split() {
        let ps = page_store();
        let mut tree = Tree::new(buffer_store(), ps.clone()).unwrap();
        let mut i = 0u64;
        loop {
            if tree.height().unwrap() > 2 {
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
        assert!(matches!(tree.root, Node::Internal(_)));
        assert_eq!(tree.root.get_num_keys(), 2);
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
}
