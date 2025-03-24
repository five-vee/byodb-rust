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
mod page_store;

use node::{ChildEntry, Internal, Leaf, Node};
use page_store::Store;
use std::rc::Rc;

pub type Result<T> = std::result::Result<T, ()>;

/// An enum representing the node(s) created during an insert or update
/// (aka an "upsert") operation on a tree.
enum Upsert<S: Store> {
    /// A newly created tree that remained  "intact", i.e. it did not split
    /// after an upsert.
    Intact(Tree<S>),
    /// The left and right splits of a tree that was created after an upsert.
    ///
    /// The left and right nodes are the same type.
    Split { left: Tree<S>, right: Tree<S> },
}

impl<S: Store> Upsert<S> {
    /// Converts the in-memory nodes (created during an upsert) into child
    /// entries of an B+ tree internal node.
    ///
    /// `i` is the index of the node that the upsert was performed on.
    fn child_entries(self, i: usize) -> Rc<[ChildEntry]> {
        match self {
            Upsert::Intact(tree) => [ChildEntry::Update {
                i,
                key: tree.root.get_key(0).into(),
                page_num: tree.page_num,
            }]
            .into(),
            Upsert::Split { left, right } => [
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

/// An enum representing the tree(s), if any, created or destroyed during a
/// deletion operation on a tree.
pub enum Deletion<S: Store> {
    /// A node without 0 keys after a delete was performed on it.
    /// This is a special-case of `Underflow` done to avoid unnecessary
    /// page allocations, since empty non-root nodes aren't allowed.
    Empty,
    /// A node that is sufficiently sized (i.e. has at least 2 keys) even after
    /// a delete was performed on it.
    Sufficient(Tree<S>),
    /// A node that was split due to a delete operation. This can happen
    /// because the node had to delete a key and replace it with another key
    /// that was larger, pushing it beyond the page size,
    /// thus triggering a split.
    ///
    /// Yes, this means the tree can grow in height despite
    /// the deletion operation.
    ///
    /// The left and right nodes are the same type.
    Split { left: Tree<S>, right: Tree<S> },
    /// A node that is NOT sufficiently sized but is not empty
    /// (i.e. has 1 key).
    Underflow(Tree<S>),
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
pub struct Tree<S: Store> {
    root: Node,
    page_num: u64,
    store: Rc<S>,
}

impl<S: Store> Tree<S> {
    /// Gets the value corresponding to the key.
    pub fn get(&self, key: &[u8]) -> Result<Option<Rc<[u8]>>> {
        match &self.root {
            Node::Internal(root) => {
                let child_idx = root.find(key).map_or_else(|| Err(()), |i| Ok(i))?;
                let child_num = root.get_child_pointer(child_idx);
                let child = self.store.read_page(child_num)?;
                let child = Self {
                    root: child,
                    page_num: child_num,
                    store: self.store.clone(),
                };
                child.get(key)
            }
            Node::Leaf(root) => Ok(root.find(key).map(|v| v.into())),
        }
    }

    /// Inserts a key-value pair.
    pub fn insert(&self, key: &[u8], val: &[u8]) -> Result<Self> {
        match self.insert_helper(key, val)? {
            Upsert::Intact(tree) => Ok(tree),
            Upsert::Split { left, right } => self.parent_of_split(left, right),
        }
    }

    /// Finds the node to insert into and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `insert`.
    fn insert_helper(&self, key: &[u8], val: &[u8]) -> Result<Upsert<S>> {
        match &self.root {
            // Base case
            Node::Leaf(leaf) => Ok(self.alloc_upsert(leaf.insert(key, val)?)?),
            // Recursive case
            Node::Internal(internal) => {
                // Find which child to recursively insert into.
                let child_idx = internal.find(key).map_or_else(|| Err(()), |i| Ok(i))?;
                let child_num = internal.get_child_pointer(child_idx);
                let child = self.store.read_page(child_num)?;
                let child = Self {
                    root: child,
                    page_num: child_num,
                    store: self.store.clone(),
                };
                let u = child.insert_helper(key, val)?;
                let child_entries = u.child_entries(child_idx);
                let u = node::Upsert::from(internal.merge_as_upsert(child_entries.as_ref())?);
                Ok(self.alloc_upsert(u)?)
            }
        }
    }

    /// Updates the value corresponding to a key.
    pub fn update(&self, key: &[u8], val: &[u8]) -> Result<Self> {
        match self.update_helper(key, val)? {
            Upsert::Intact(tree) => Ok(tree),
            Upsert::Split { left, right } => self.parent_of_split(left, right),
        }
    }

    /// Finds the node to update and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `update`.
    fn update_helper(&self, key: &[u8], val: &[u8]) -> Result<Upsert<S>> {
        match &self.root {
            // Base case
            Node::Leaf(leaf) => Ok(self.alloc_upsert(leaf.update(key, val)?)?),
            // Recursive case
            Node::Internal(internal) => {
                // Find which child to recursively update at.
                let child_idx = internal.find(key).map_or_else(|| Err(()), |i| Ok(i))?;
                let child_num = internal.get_child_pointer(child_idx);
                let child = self.store.read_page(child_num)?;
                let child = Self {
                    root: child,
                    page_num: child_num,
                    store: self.store.clone(),
                };
                let u = child.update_helper(key, val)?;
                let child_entries = u.child_entries(child_idx);
                let u = node::Upsert::from(internal.merge_as_upsert(child_entries.as_ref())?);
                Ok(self.alloc_upsert(u)?)
            }
        }
    }

    /// Deletes a key and its corresponding value.
    pub fn delete(&self, key: &[u8]) -> Result<Self> {
        match self.delete_helper(key)? {
            Deletion::Empty => {
                let root = Node::Leaf(Leaf::default());
                let page_num = self.store.write_page(&root)?;
                Ok(Self {
                    root,
                    page_num,
                    store: self.store.clone(),
                })
            }
            Deletion::Split { left, right } => self.parent_of_split(left, right),
            Deletion::Sufficient(tree) => Ok(tree),
            Deletion::Underflow(tree) => Ok(tree),
        }
    }

    /// Finds the node to delete the key from and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `delete`.
    fn delete_helper(&self, key: &[u8]) -> Result<Deletion<S>> {
        match &self.root {
            // Base case
            Node::Leaf(leaf) => Ok(self.alloc_deletion(leaf.delete(key)?)?),
            // Recursive case
            Node::Internal(parent) => {
                // Find which child to recursively delete from.
                let child_idx = parent.find(key).map_or_else(|| Err(()), |i| Ok(i))?;
                let child_num = parent.get_child_pointer(child_idx);
                let child = self.store.read_page(child_num)?;
                let child = Self {
                    root: child,
                    page_num: child_num,
                    store: self.store.clone(),
                };
                match child.delete_helper(key)? {
                    Deletion::Empty => {
                        let d = parent.merge_as_deletion(&[ChildEntry::Delete { i: child_idx }])?;
                        Ok(self.alloc_deletion(d)?)
                    }
                    Deletion::Sufficient(child) => {
                        let d = parent.merge_as_deletion(&[ChildEntry::Update {
                            i: child_idx,
                            key: child.root.get_key(0).into(),
                            page_num: child.page_num,
                        }])?;
                        Ok(self.alloc_deletion(d)?)
                    }
                    Deletion::Split {
                        left: left_child,
                        right: right_child,
                    } => {
                        let d = parent.merge_as_deletion(&[
                            ChildEntry::Update {
                                i: child_idx,
                                key: left_child.root.get_key(0).into(),
                                page_num: left_child.page_num,
                            },
                            ChildEntry::Insert {
                                key: right_child.root.get_key(0).into(),
                                page_num: right_child.page_num,
                            },
                        ])?;
                        Ok(self.alloc_deletion(d)?)
                    }
                    Deletion::Underflow(child) => {
                        self.try_fix_underflow(parent, child, child_idx)
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
        parent: &Internal,
        child: Self,
        child_idx: usize,
    ) -> Result<Deletion<S>> {
        // Try stealing or merging the left sibling.
        if child_idx > 0 {
            return self.steal_or_merge(parent, &child.root, child_idx, child_idx - 1)
        }
        // Try stealing or merging the right sibling.
        if child_idx < parent.get_num_keys() - 1 {
            return self.steal_or_merge(parent, &child.root, child_idx, child_idx + 1)
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
        let d = parent.merge_as_deletion(&[ChildEntry::Update {
            i: child_idx,
            key: child.root.get_key(0).into(),
            page_num: child.page_num,
        }])?;
        self.alloc_deletion(d)
    }

    /// Tries to fix the underflow of `child` by stealing from or merging from
    /// one of its direct siblings.
    fn steal_or_merge(
        &self,
        parent: &Internal,
        child: &Node,
        child_idx: usize,
        sibling_idx: usize,
    ) -> Result<Deletion<S>> {
        let sibling_num = parent.get_child_pointer(sibling_idx);
        let sibling = self.store.read_page(sibling_num)?;
        let mut left = &sibling;
        let mut right = child;
        if sibling_idx > child_idx {
            (left, right) = (right, left);
        }
        self.alloc_deletion(node::steal_or_merge(left, right)?)
    }

    /// Returns a new internal root node whose children are split nodes
    /// newly-created due to an operation on the tree.
    fn parent_of_split(&self, left: Self, right: Self) -> Result<Self> {
        let keys = [left.root.get_key(0), right.root.get_key(0)];
        let child_pointers = [left.page_num, right.page_num];
        let root = Internal::parent_of_split(keys, child_pointers)?;
        let root = Node::Internal(root);
        let page_num = self.store.write_page(&root)?;
        Ok(Self {
            root,
            page_num,
            store: self.store.clone(),
        })
    }

    /// Allocates pages for the in-memory nodes created during an upsert.
    fn alloc_upsert(&self, u: node::Upsert) -> Result<Upsert<S>> {
        match u {
            node::Upsert::Intact(root) => {
                let page_num = self.store.write_page(&root)?;
                Ok(Upsert::Intact(Self {
                    root,
                    page_num,
                    store: self.store.clone(),
                }))
            }
            node::Upsert::Split { left, right } => {
                let left_page_num = self.store.write_page(&left)?;
                let right_page_num = self.store.write_page(&right)?;
                let left = Self {
                    root: left,
                    page_num: left_page_num,
                    store: self.store.clone(),
                };
                let right = Self {
                    root: right,
                    page_num: right_page_num,
                    store: self.store.clone(),
                };
                Ok(Upsert::Split { left, right })
            }
        }
    }

    /// Allocates pages for the in-memory nodes created during a deletion.
    fn alloc_deletion(&self, d: node::Deletion) -> Result<Deletion<S>> {
        match d {
            node::Deletion::Empty => Ok(Deletion::Empty),
            node::Deletion::Sufficient(root) => {
                let page_num = self.store.write_page(&root)?;
                Ok(Deletion::Sufficient(Self {
                    root,
                    page_num,
                    store: self.store.clone(),
                }))
            }
            node::Deletion::Split { left, right } => {
                let left_page_num = self.store.write_page(&left)?;
                let right_page_num = self.store.write_page(&right)?;
                let left = Self {
                    root: left,
                    page_num: left_page_num,
                    store: self.store.clone(),
                };
                let right = Self {
                    root: right,
                    page_num: right_page_num,
                    store: self.store.clone(),
                };
                Ok(Deletion::Split { left, right })
            }
            node::Deletion::Underflow(root) => {
                let page_num = self.store.write_page(&root)?;
                Ok(Deletion::Underflow(Self {
                    root,
                    page_num,
                    store: self.store.clone(),
                }))
            }
        }
    }
}
