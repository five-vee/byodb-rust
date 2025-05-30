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
use crate::core::mmap::{Guard, Writer};
use node::{ChildEntry, Internal, Node, NodeEffect, Sufficiency};

use super::mmap::{ImmutablePage, WriterPage};

type Result<T> = std::result::Result<T, TreeError>;

/// A copy-on-write (COW) B+ Tree data structure that stores data in disk.
///
/// Instead of directly updating the existing nodes in the tree when an
/// insertion, deletion, or update occurs, the COW approach creates a modified
/// copy of the node (or the path of nodes leading to the change). The original
/// node remains unchanged. This means that any other readers concurrently
/// accessing the tree will continue to see the consistent, older version of
/// the data until they reach a point where they would naturally access the
/// newly written parts.
pub struct Tree<'g, P: ImmutablePage<'g>, G: Guard<'g, P>> {
    guard: &'g G,
    node: Node<'g, P>,
}

impl<'g, P: ImmutablePage<'g>, G: Guard<'g, P>> Tree<'g, P, G> {
    /// Loads the root of the tree at a specified root page num.
    #[inline]
    pub fn new(guard: &'g G, page_num: usize) -> Self {
        Tree {
            guard,
            node: Self::read(guard, page_num),
        }
    }

    #[inline]
    pub fn page_num(&self) -> usize {
        self.node.page_num()
    }

    /// Reads the page at `page_num` and returns it represented as a [`Node`].
    /// This is a convenience wrapper around the unsafe [`Node::read`].
    #[inline]
    fn read(guard: &'g G, page_num: usize) -> Node<'g, P> {
        // Safety: The tree maintains the invariant that it'll only read nodes
        // that are traversible from the root, including the root itself.
        unsafe { Node::read(guard, page_num) }
    }

    /// Gets the value corresponding to the key.
    pub fn get(&self, key: &[u8]) -> Result<Option<&'g [u8]>> {
        match &self.node {
            Node::Internal(parent) => {
                let child_idx = parent.find(key);
                let child_num = parent.get_child_pointer(child_idx);
                let child = Tree::read(self.guard, child_num);
                let child = Tree {
                    guard: self.guard,
                    node: child,
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
    pub fn in_order_iter(&self) -> InOrder<'g, 'g, P, G> {
        let mut stack = Vec::new();
        stack.reserve_exact(self.height());
        stack.push((
            0,
            Tree {
                guard: self.guard,
                node: Tree::read(self.guard, self.node.page_num()),
            },
        ));
        InOrder {
            stack,
            end_bound: std::ops::Bound::Unbounded,
        }
    }

    /// Iterates through the tree in-order, bounded by `range`.
    pub fn in_order_range_iter<'q, R>(&self, range: &'q R) -> InOrder<'q, 'g, P, G>
    where
        R: RangeBounds<[u8]>,
    {
        if matches!(range.start_bound(), std::ops::Bound::Unbounded) {
            let mut stack = Vec::new();
            stack.reserve_exact(self.height());
            stack.push((
                0,
                Tree {
                    guard: self.guard,
                    node: Tree::read(self.guard, self.node.page_num()),
                },
            ));
            return InOrder {
                stack,
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
        let mut stack = vec![(
            0,
            Tree {
                guard: self.guard,
                node: Tree::read(self.guard, self.node.page_num()),
            },
        )];
        while let Some((i, tree)) = stack.last_mut() {
            if *i == tree.node.get_num_keys() {
                stack.pop();
                break;
            }
            match &tree.node {
                Node::Leaf(leaf) => match leaf.iter().position(&leaf_predicate) {
                    // leaf range < start; try again up the stack
                    None => {
                        stack.pop();
                    }
                    Some(j) => {
                        *i = j;
                        break;
                    }
                },
                Node::Internal(internal) => {
                    let child_idx = (*i..internal.get_num_keys())
                        .rev()
                        .find(|&j| internal.get_key(j) <= start)
                        .unwrap_or(*i);
                    let child_page = internal.get_child_pointer(child_idx);
                    let child = Tree::new(tree.guard, child_page);
                    *i = child_idx + 1;
                    stack.push((0, child));
                }
            }
        }
        InOrder {
            stack,
            end_bound: range.end_bound(),
        }
    }

    /// Finds the height of the tree.
    fn height(&self) -> usize {
        match &self.node {
            Node::Leaf(_) => 1,
            Node::Internal(internal) => {
                1 + Tree::new(self.guard, internal.get_child_pointer(0)).height()
            }
        }
    }
}

impl<'w, 's> Tree<'w, WriterPage<'w, 's>, Writer<'s>> {
    /// Inserts a key-value pair. The resulting tree won't be visible to
    /// readers until the writer is externally flushed.
    pub fn insert(self, key: &[u8], val: &[u8]) -> Result<Self> {
        let writer = self.guard;
        let new_root = match self.insert_helper(key, val)? {
            NodeEffect::Intact(new_root) => new_root,
            NodeEffect::Split { left, right } => Self::parent_of_split(writer, &left, &right),
            _ => unreachable!(),
        };
        Ok(Tree {
            guard: writer,
            node: new_root,
        })
    }

    /// Finds the node to insert into and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `insert`.
    fn insert_helper(self, key: &[u8], val: &[u8]) -> Result<NodeEffect<'w, 's>> {
        match self.node {
            // Base case
            Node::Leaf(leaf) => Ok(leaf.insert(self.guard, key, val)?.into()),
            // Recursive case
            Node::Internal(internal) => {
                // Find which child to recursively insert into.
                let child_idx = internal.find(key);
                let child_num = internal.get_child_pointer(child_idx);
                let child = Tree::read(self.guard, child_num);
                let child = Tree {
                    guard: self.guard,
                    node: child,
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
        let new_root = match self.update_helper(key, val)? {
            NodeEffect::Intact(new_root) => new_root,
            NodeEffect::Split { left, right } => Self::parent_of_split(writer, &left, &right),
            _ => unreachable!(),
        };
        Ok(Tree {
            guard: writer,
            node: new_root,
        })
    }

    /// Finds the node to update and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `update`.
    fn update_helper(self, key: &[u8], val: &[u8]) -> Result<NodeEffect<'w, 's>> {
        match self.node {
            // Base case
            Node::Leaf(leaf) => Ok(leaf.update(self.guard, key, val)?.into()),
            // Recursive case
            Node::Internal(parent) => {
                // Find which child to recursively update at.
                let child_idx = parent.find(key);
                let child_num = parent.get_child_pointer(child_idx);
                let child = Tree::read(self.guard, child_num);
                let child = Tree {
                    guard: self.guard,
                    node: child,
                };
                let child = child.update_helper(key, val)?;
                let child_entries = child.child_entries(child_idx);
                let effect = parent.merge_child_entries(self.guard, child_entries.as_ref());
                Ok(effect.into())
            }
        }
    }

    /// Deletes a key and its corresponding value.
    pub fn delete(self, key: &[u8]) -> Result<Self> {
        let writer = self.guard;
        let new_root = match self.delete_helper(key)? {
            NodeEffect::Empty => Node::<WriterPage<'w, 's>>::new_empty(writer),
            NodeEffect::Intact(root) => match node::sufficiency(&root) {
                Sufficiency::Empty => unreachable!(),
                Sufficiency::Underflow => match root {
                    Node::Leaf(_) => root,
                    Node::Internal(internal) => Tree::read(writer, internal.get_child_pointer(0)),
                },
                Sufficiency::Sufficient => root,
            },
            NodeEffect::Split { left, right } => Self::parent_of_split(writer, &left, &right),
        };
        Ok(Tree {
            guard: writer,
            node: new_root,
        })
    }

    /// Finds the node to delete the key from and creates a modified copy of
    /// the resulting tree.
    ///
    /// This is a recursive implementation of `delete`.
    fn delete_helper(self, key: &[u8]) -> Result<NodeEffect<'w, 's>> {
        match self.node {
            // Base case
            Node::Leaf(leaf) => Ok(leaf.delete(self.guard, key)?.into()),
            // Recursive case
            Node::Internal(parent) => {
                // Find which child to recursively delete from.
                let child_idx = parent.find(key);
                let child_num = parent.get_child_pointer(child_idx);
                let child = Tree::read(self.guard, child_num);
                let child = Tree {
                    guard: self.guard,
                    node: child,
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
                        Sufficiency::Underflow => Ok(Self::try_fix_underflow(
                            self.guard, parent, child, child_idx,
                        )),
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
        writer: &'w Writer<'s>,
        parent: Internal<'w, WriterPage<'w, 's>>,
        child: Node<'w, WriterPage<'w, 's>>,
        child_idx: usize,
    ) -> NodeEffect<'w, 's> {
        // Try to steal from or merge with a sibling.
        let sibling_idx = if child_idx > 0 {
            Some(child_idx - 1)
        } else if child_idx < parent.get_num_keys() - 1 {
            Some(child_idx + 1)
        } else {
            None
        };
        if let Some(sibling_idx) = sibling_idx {
            let sibling = Self::read(writer, parent.get_child_pointer(sibling_idx));
            let can_steal_or_merge = node::can_steal(&sibling, &child, sibling_idx < child_idx)
                || node::can_merge(&sibling, &child);
            if can_steal_or_merge {
                return Self::steal_or_merge(
                    writer,
                    parent,
                    child,
                    child_idx,
                    sibling,
                    sibling_idx,
                );
            }
        }

        // Leave as underflow.
        parent
            .merge_child_entries(
                writer,
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
        writer: &'w Writer<'s>,
        parent: Internal<'w, WriterPage<'w, 's>>,
        child: Node<'w, WriterPage<'w, 's>>,
        child_idx: usize,
        sibling: Node<'w, WriterPage<'w, 's>>,
        sibling_idx: usize,
    ) -> NodeEffect<'w, 's> {
        let (mut left_idx, mut right_idx) = (sibling_idx, child_idx);
        let (left, right) = if sibling_idx < child_idx {
            (sibling, child)
        } else {
            (left_idx, right_idx) = (child_idx, sibling_idx);
            (child, sibling)
        };
        match node::steal_or_merge(left, right, writer) {
            NodeEffect::Empty => unreachable!(),
            NodeEffect::Intact(child) => {
                // merged
                let effect = parent.merge_child_entries(
                    writer,
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
                    writer,
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
    fn parent_of_split(
        writer: &'w Writer<'s>,
        left: &Node<'w, WriterPage<'w, 's>>,
        right: &Node<'w, WriterPage<'w, 's>>,
    ) -> Node<'w, WriterPage<'w, 's>> {
        let keys = [left.get_key(0), right.get_key(0)];
        let child_pointers = [left.page_num(), right.page_num()];
        let root = Internal::parent_of_split(writer, keys, child_pointers);
        Node::Internal(root)
    }
}

#[cfg(test)]
impl<'g, P: ImmutablePage<'g>, G: Guard<'g, P>> Tree<'g, P, G> {
    /// Checks the height invariant of the tree.
    /// This performs a scan of the entire tree, so it's not really efficient.
    #[allow(dead_code)]
    pub fn check_height(&self) -> Result<u32> {
        match &self.node {
            Node::Leaf(_) => Ok(1),
            Node::Internal(node) => {
                assert!(node.get_num_keys() >= 2);
                let mut height: Option<u32> = None;
                for (_, pn) in node.iter() {
                    let child = Tree {
                        guard: self.guard,
                        node: Tree::read(self.guard, pn),
                    };
                    let child_height = child.check_height()?;
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
pub struct InOrder<'q, 'g, P: ImmutablePage<'g>, G: Guard<'g, P>> {
    stack: Vec<(usize, Tree<'g, P, G>)>,
    end_bound: std::ops::Bound<&'q [u8]>,
}

impl<'g, P: ImmutablePage<'g>, G: Guard<'g, P>> Iterator for InOrder<'_, 'g, P, G> {
    type Item = (&'g [u8], &'g [u8]);
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((i, tree)) = self.stack.last_mut() {
            if *i == tree.node.get_num_keys() {
                self.stack.pop();
                continue;
            }
            match &tree.node {
                Node::Leaf(leaf) => {
                    let key = leaf.get_key(*i);
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
                    let val = leaf.get_value(*i);
                    *i += 1;
                    return Some((key, val));
                }
                Node::Internal(internal) => {
                    let pn = internal.get_child_pointer(*i);
                    let child = Tree {
                        guard: tree.guard,
                        node: Tree::read(tree.guard, pn),
                    };
                    *i += 1;
                    self.stack.push((0, child));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    // Tree tests are actually located at crate::api::tests
    // because it's easier to test with the DB/Txn abstraction.
}
