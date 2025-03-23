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
use std::rc::Rc;

use crate::node;

pub type Result<T> = std::result::Result<T, ()>;

/// An enum representing the node(s) created during an insert or update
/// (aka an "upsert") operation on a tree.
enum Upsert {
    /// A newly created tree that remained  "intact", i.e. it did not split
    /// after an upsert.
    Intact(Tree),
    /// The left and right splits of a tree that was created after an upsert.
    Split { left: Tree, right: Tree },
}

impl Upsert {
    /// Allocates pages for the in-memory nodes created during an upsert.
    fn alloc(u: node::Upsert) -> Result<Self> {
        match u {
            node::Upsert::Intact(root) => {
                let page_num = Tree::write_page(&root)?;
                Ok(Upsert::Intact(Tree { root, page_num }))
            }
            node::Upsert::Split { left, right } => {
                let left_page_num = Tree::write_page(&left)?;
                let right_page_num = Tree::write_page(&right)?;
                let left = Tree {
                    root: left,
                    page_num: left_page_num,
                };
                let right = Tree {
                    root: right,
                    page_num: right_page_num,
                };
                Ok(Upsert::Split { left, right })
            }
        }
    }

    /// Converts the in-memory nodes (created during an upsert) into child
    /// entries of an B+ tree internal node.
    ///
    /// `i` is the index of the node that the upsert was performed on.
    fn child_entries(self, i: u16) -> Rc<[node::ChildEntry]> {
        match self {
            Upsert::Intact(tree) => [node::ChildEntry {
                maybe_i: Some(i),
                key: tree.root.get_key(0).into(),
                page_num: tree.page_num,
            }]
            .into(),
            Upsert::Split { left, right } => [
                node::ChildEntry {
                    maybe_i: Some(i),
                    key: left.root.get_key(0).into(),
                    page_num: left.page_num,
                },
                node::ChildEntry {
                    maybe_i: None,
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
pub enum Deletion {
    /// A node without 0 keys after a delete was performed on it.
    /// This is a special-case of `Underflow` done to avoid unnecessary
    /// page allocations, since empty non-root nodes aren't allowed.
    Empty,
    /// A node that is sufficiently-sized (i.e. has at least 2 keys) even after
    /// a delete was performed on it.
    Sufficient(Tree),
    /// A node that was split due to a delete operation. This can happen
    /// because the node had to delete a key and replace it with another key
    /// that was larger, pushing it beyond the page size,
    /// thus triggering a split.
    ///
    /// Yes, this means the tree can grow in height despite
    /// the deletion operation.
    Split { left: Tree, right: Tree },
    /// A node that is NOT sufficiently-sized but is not empty
    /// (i.e. has 1 key).
    Underflow(Tree),
}

impl Deletion {
    /// Allocates pages for the in-memory nodes created during a deletion.
    fn alloc(d: node::Deletion) -> Result<Self> {
        match d {
            node::Deletion::Empty => Ok(Deletion::Empty),
            node::Deletion::Sufficient(root) => {
                let page_num = Tree::write_page(&root)?;
                Ok(Deletion::Sufficient(Tree { root, page_num }))
            }
            node::Deletion::Split { left, right } => {
                let left_page_num = Tree::write_page(&left)?;
                let right_page_num = Tree::write_page(&right)?;
                let left = Tree {
                    root: left,
                    page_num: left_page_num,
                };
                let right = Tree {
                    root: right,
                    page_num: right_page_num,
                };
                Ok(Deletion::Split { left, right })
            }
            node::Deletion::Underflow(root) => {
                let page_num = Tree::write_page(&root)?;
                Ok(Deletion::Underflow(Tree { root, page_num }))
            }
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
pub struct Tree {
    root: node::Node,
    page_num: u64,
}

impl Tree {
    /// Gets the value corresponding to the key.
    pub fn get(&self, key: &[u8]) -> Result<Option<Rc<[u8]>>> {
        match &self.root {
            node::Node::Internal(root) => {
                let child_idx = root.find(key).map_or_else(|| Err(()), |i| Ok(i))?;
                let child_num = root.get_child_pointer(child_idx)?;
                let child = Self::read_page(child_num)?;
                let child = Self {
                    root: child,
                    page_num: child_num,
                };
                child.get(key)
            },
            node::Node::Leaf(root) => Ok(root.find(key).map(|v| v.into()))
        }
    }

    /// Inserts a key-value pair.
    pub fn insert(&self, key: &[u8], val: &[u8]) -> Result<Self> {
        match self.insert_helper(key, val)? {
            Upsert::Intact(tree) => Ok(tree),
            Upsert::Split { left, right } => Self::new_root(left, right),
        }
    }

    /// Finds the node to insert into and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `insert`.
    fn insert_helper(&self, key: &[u8], val: &[u8]) -> Result<Upsert> {
        match &self.root {
            // Base case
            node::Node::Leaf(leaf) => Ok(Upsert::alloc(leaf.insert(key, val)?)?),
            // Recursive case
            node::Node::Internal(internal) => {
                // Find which child to recursively insert into.
                let child_idx = internal.find(key).map_or_else(|| Err(()), |i| Ok(i))?;
                let child_num = internal.get_child_pointer(child_idx)?;
                let child = Self::read_page(child_num)?;
                let child = Self {
                    root: child,
                    page_num: child_num,
                };
                let u = child.insert_helper(key, val)?;
                let child_entries = u.child_entries(child_idx);
                let u = node::Upsert::from(internal.upsert_child_entries(child_entries.as_ref())?);
                Ok(Upsert::alloc(u)?)
            }
        }
    }

    /// Updates the value corresponding to a key.
    pub fn update(&self, key: &[u8], val: &[u8]) -> Result<Self> {
        match self.update_helper(key, val)? {
            Upsert::Intact(tree) => Ok(tree),
            Upsert::Split { left, right } => Self::new_root(left, right),
        }
    }

    /// Finds the node to update and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `update`.
    fn update_helper(&self, key: &[u8], val: &[u8]) -> Result<Upsert> {
        match &self.root {
            // Base case
            node::Node::Leaf(leaf) => Ok(Upsert::alloc(leaf.update(key, val)?)?),
            // Recursive case
            node::Node::Internal(internal) => {
                // Find which child to recursively update at.
                let child_idx = internal.find(key).map_or_else(|| Err(()), |i| Ok(i))?;
                let child_num = internal.get_child_pointer(child_idx)?;
                let child = Self::read_page(child_num)?;
                let child = Self {
                    root: child,
                    page_num: child_num,
                };
                let u = child.update_helper(key, val)?;
                let child_entries = u.child_entries(child_idx);
                let u = node::Upsert::from(internal.upsert_child_entries(child_entries.as_ref())?);
                Ok(Upsert::alloc(u)?)
            }
        }
    }

    /// Deletes a key and its corresponding value.
    pub fn delete(&self, key: &[u8]) -> Result<Self> {
        match self.delete_helper(key)? {
            Deletion::Empty => {
                let root = node::Node::Leaf(node::Leaf::new(&[], &[])?);
                let page_num = Self::write_page(&root)?;
                Ok(Tree { root, page_num })
            }
            Deletion::Split { left, right } => Self::new_root(left, right),
            Deletion::Sufficient(tree) => Ok(tree),
            Deletion::Underflow(tree) => Ok(tree),
        }
    }

    /// Finds the node to delete the key from and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `delete`.
    fn delete_helper(&self, key: &[u8]) -> Result<Deletion> {
        match &self.root {
            // Base case
            node::Node::Leaf(leaf) => Ok(Deletion::alloc(leaf.delete(key)?)?),
            // Recursive case
            node::Node::Internal(internal) => {
                // Find which child to recursively delete from.
                let child_idx = internal.find(key).map_or_else(|| Err(()), |i| Ok(i))?;
                let child_num = internal.get_child_pointer(child_idx)?;
                let child = Self::read_page(child_num)?;
                let child = Self {
                    root: child,
                    page_num: child_num,
                };
                match child.delete_helper(key)? {
                    Deletion::Empty => {
                        // What if nkeys is now 1? Then delete_child_entry will return Underflow.
                        let d = internal.delete_child_entry(child_idx)?;
                        return Ok(Deletion::alloc(d)?);
                    }
                    Deletion::Sufficient(child) => {
                        let d = internal.update_child_entry(
                            child_idx,
                            child.root.get_key(0),
                            child.page_num,
                        )?;
                        return Ok(Deletion::alloc(d)?);
                    }
                    Deletion::Split {
                        left: child_split_left,
                        right: child_split_right,
                    } => {
                        // Internal is sufficient.
                        todo!();
                    }
                    Deletion::Underflow(child) => {
                        let delta: node::DeletionDelta =
                            Self::try_fix_underflow(internal, child, child_idx)?;
                        let d = internal.merge_delta(delta)?;
                        return Ok(Deletion::alloc(d)?);
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
        parent: &node::Internal,
        child: Tree,
        child_idx: u16,
    ) -> Result<node::DeletionDelta> {
        // Try stealing or merging the left sibling.
        if child_idx > 0 {
            match Self::try_steal_or_merge(parent, &child.root, child_idx, child_idx - 1)? {
                None => {}
                Some(delta) => return Ok(delta),
            }
        }
        // Try stealing or merging the right sibling.
        match Self::try_steal_or_merge(parent, &child.root, child_idx, child_idx + 1)? {
            None => {}
            Some(delta) => return Ok(delta),
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
        return Ok([(child_idx, Some(child.page_num))].into());
    }

    /// Tries to fix the underflow of `child` by stealing from or merging from
    /// one of its direct siblings.
    fn try_steal_or_merge(
        parent: &node::Internal,
        child: &node::Node,
        child_idx: u16,
        sibling_idx: u16,
    ) -> Result<Option<node::DeletionDelta>> {
        let sibling_num = parent.get_child_pointer(sibling_idx)?;
        let sibling = Self::read_page(sibling_num)?;
        if node::can_steal(&sibling, &child) {
            let (new_sibling, new_child) = node::steal(&sibling, child)?;
            let new_sibling_num = Self::write_page(&new_sibling)?;
            let new_child_num = Self::write_page(&new_child)?;
            return Ok(Some(
                [
                    (child_idx - 1, Some(new_sibling_num)),
                    (child_idx, Some(new_child_num)),
                ]
                .into(),
            ));
        }
        if node::can_merge(&child, &sibling) {
            let new_sibling = node::merge(child, &sibling)?;
            let new_sibling_num = Self::write_page(&new_sibling)?;
            return Ok(Some(
                [(child_idx - 1, Some(new_sibling_num)), (child_idx, None)].into(),
            ));
        }
        Ok(None)
    }

    /// Returns a new internal root node whose children are split nodes
    /// newly-created due to an operation on the tree.
    fn new_root(left: Tree, right: Tree) -> Result<Self> {
        let keys = &[left.root.get_key(0), right.root.get_key(0)];
        let child_pointers = &[left.page_num, right.page_num];
        let root = node::Internal::new(keys, child_pointers);
        let root = node::Node::Internal(root);
        let page_num = Self::write_page(&root)?;
        Ok(Self { root, page_num })
    }

    /// Reads a page from disk into an in-memory B+ tree node.
    fn read_page(page_num: u64) -> Result<node::Node> {
        unimplemented!();
    }

    /// Writes an in-memory B+ tree node into a page on disk.
    fn write_page(node: &node::Node) -> Result<u64> {
        unimplemented!();
    }
}
