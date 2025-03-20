use std::rc::Rc;

use crate::bplus_node as node;

/// Represents the results of an insert/update operation peformed recursively
/// on a node.
enum NodeResult {
    NonSplit(Tree),
    Split(Split),
}

struct Split {
    left_key: Rc<[u8]>,
    left_page_num: u64,
    right_key: Rc<[u8]>,
    right_page_num: u64,
}

enum Node {
    Leaf(node::Leaf),
    Internal(node::Internal),
}

impl Node {
    fn get_key(&self, i: u16) -> &[u8] {
        match self {
            Node::Leaf(leaf) => leaf.get_key(i),
            Node::Internal(internal) => internal.get_key(i),
        }
    }
}

/// Tree is a copy-on-write B+ Tree.
pub(crate) struct Tree {
    root: Node,
    root_num: u64,
}

impl Tree {
    /// Inserts a key-value pair into the tree.
    /// TODO: return Result if key/val too large.
    pub(crate) fn insert(self, key: &[u8], val: &[u8]) -> Tree {
        match Self::insert_node(self.root, key, val) {
            NodeResult::NonSplit(tree) => tree,
            NodeResult::Split(split) => Self::new_higher_tree(split),
        }
    }

    /// Recursive version of insert.
    fn insert_node(root: Node, key: &[u8], val: &[u8]) -> NodeResult {
        match root {
            Node::Leaf(leaf) => return Self::insert_leaf(leaf, key, val),
            Node::Internal(internal) => return Self::insert_internal(internal, key, val),
        }
    }

    /// Inserts the key-value in a leaf.
    /// Splits the leaf if needed.
    fn insert_leaf(leaf: node::Leaf, key: &[u8], val: &[u8]) -> NodeResult {
        if leaf.needs_split(node::Revision::Insert, key.len() + val.len()) {
            let (mut left, mut right) = leaf.copy_on_split();
            // Insert into left or right?
            if key < right.get_key(0) {
                left.inplace_insert(left.lookup_insert(key), key, val);
            } else {
                right.inplace_insert(right.lookup_insert(key), key, val);
            }
            return NodeResult::Split(Split {
                left_key: left.get_key(0).into(),
                left_page_num: Self::alloc_page(&Node::Leaf(left)),
                right_key: right.get_key(0).into(),
                right_page_num: Self::alloc_page(&Node::Leaf(right)),
            });
        }
        let mut leaf = leaf;
        leaf.inplace_insert(leaf.lookup_insert(key), key, val);
        let root = Node::Leaf(leaf);
        let root_num = Self::alloc_page(&root);
        return NodeResult::NonSplit(Self { root, root_num });
    }

    /// Inserts the key-value in a leaf descendant of an internal node.
    /// Recursively splits if needed.
    fn insert_internal(internal: node::Internal, key: &[u8], val: &[u8]) -> NodeResult {
        let child_idx = internal.lookup_child(key);
        let child_num = internal.get_child_pointer(child_idx);
        let child = Self::get_page(child_num);
        match Self::insert_node(child, key, val) {
            NodeResult::NonSplit(child_tree) => {
                return Self::handle_internal_insertion_nonsplit(internal, child_idx, child_tree);
            }
            NodeResult::Split(split) => {
                return Self::handle_internal_insertion_split(internal, child_idx, split);
            }
        }
    }

    /// Handles an internal node when its child has changed (but not split) upon insertion.
    fn handle_internal_insertion_nonsplit(
        internal: node::Internal,
        child_idx: u16,
        child_tree: Tree,
    ) -> NodeResult {
        let mut internal = internal;
        let child_min_key = child_tree.root.get_key(0);
        if Self::internal_needs_update(&internal, child_idx, child_min_key) {
            if internal.needs_split_before_update(child_min_key.len()) {
                return Self::split_internal_then_update(internal, &child_tree, child_min_key);
            }
            internal.inplace_update(0, child_min_key.as_ref(), child_tree.root_num);
        } else {
            internal.set_child_pointer(child_idx, child_tree.root_num);
        }
        let root = Node::Internal(internal);
        let root_num = Self::alloc_page(&root);
        return NodeResult::NonSplit(Self { root, root_num });
    }

    /// Handles an internal node when its child has changed and split upon insertion.
    fn handle_internal_insertion_split(
        internal: node::Internal,
        child_idx: u16,
        split: Split,
    ) -> NodeResult {
        let mut internal = internal;
        if Self::internal_needs_update(&internal, child_idx, split.left_key.as_ref()) {
            let needs_split = internal
                .needs_split_before_update_and_insert(split.left_key.len(), split.right_key.len());
            if needs_split {
                return Self::split_internal_then_update_and_insert(internal, split);
            }
            internal.inplace_update(0, split.left_key.as_ref(), split.left_page_num);
        } else {
            internal.set_child_pointer(child_idx, split.left_page_num);
        }
        if internal.needs_split_before_insert(split.right_key.len()) {
            return Self::split_internal_then_insert(
                internal,
                split.right_key.clone(),
                split.right_page_num,
            );
        }
        internal.inplace_insert(
            child_idx + 1,
            split.right_key.as_ref(),
            split.right_page_num,
        );
        let root = Node::Internal(internal);
        let new_root_num = Self::alloc_page(&root);
        return NodeResult::NonSplit(Self {
            root: root,
            root_num: new_root_num,
        });
    }

    /// Updates the value corresponding to key in the tree.
    /// TODO: return Result if key doesn't exist or key/val too large.
    pub(crate) fn update(self, key: &[u8], val: &[u8]) -> Tree {
        match Self::update_node(self.root, key, val) {
            NodeResult::NonSplit(tree) => tree,
            NodeResult::Split(split) => Self::new_higher_tree(split),
        }
    }

    /// Recursive version of update.
    fn update_node(root: Node, key: &[u8], val: &[u8]) -> NodeResult {
        match root {
            Node::Leaf(leaf) => return Self::update_leaf(leaf, key, val),
            Node::Internal(internal) => return Self::update_internal(internal, key, val),
        }
    }

    /// Updates the value corresponding to key in a leaf.
    /// Splits if needed.
    fn update_leaf(leaf: node::Leaf, key: &[u8], val: &[u8]) -> NodeResult {
        if leaf.needs_split(
            node::Revision::Update(leaf.lookup_key(key)),
            key.len() + val.len(),
        ) {
            let (mut left, mut right) = leaf.copy_on_split();
            // Update left or right?
            if key < right.get_key(0) {
                left.inplace_update(left.lookup_insert(key), key, val);
            } else {
                right.inplace_update(right.lookup_insert(key), key, val);
            }
            return NodeResult::Split(Split {
                left_key: left.get_key(0).into(),
                left_page_num: Self::alloc_page(&Node::Leaf(left)),
                right_key: right.get_key(0).into(),
                right_page_num: Self::alloc_page(&Node::Leaf(right)),
            });
        }
        let mut leaf = leaf;
        leaf.inplace_insert(leaf.lookup_insert(key), key, val);
        let root = Node::Leaf(leaf);
        let root_num = Self::alloc_page(&root);
        return NodeResult::NonSplit(Self { root, root_num });
    }

    /// Updates the value corresponding to key in a leaf descendant of an internal node.
    /// Recursively splits if needed.
    fn update_internal(internal: node::Internal, key: &[u8], val: &[u8]) -> NodeResult {
        let child_idx = internal.lookup_child(key);
        let child_num = internal.get_child_pointer(child_idx);
        let child = Self::get_page(child_num);
        let result = Self::update_node(child, key, val);
        match result {
            NodeResult::NonSplit(child_tree) => {
                return Self::handle_internal_update_nonsplit(internal, child_idx, child_tree);
            }
            NodeResult::Split(split) => {
                return Self::handle_internal_update_split(internal, child_idx, split);
            }
        }
    }

    /// Handles an internal node when its child has changed (but not split) upon update.
    fn handle_internal_update_nonsplit(
        internal: node::Internal,
        child_idx: u16,
        child_tree: Tree,
    ) -> NodeResult {
        let mut internal = internal;
        let k = Rc::<[u8]>::from(internal.get_key(child_idx));
        internal.inplace_update(child_idx, k.as_ref(), child_tree.root_num);
        let root = Node::Internal(internal);
        let root_num = Self::alloc_page(&root);
        return NodeResult::NonSplit(Self { root, root_num });
    }

    /// Handles an internal node when its child has changed and split upon update.
    fn handle_internal_update_split(
        internal: node::Internal,
        child_idx: u16,
        split: Split,
    ) -> NodeResult {
        let mut internal = internal;
        internal.set_child_pointer(child_idx, split.left_page_num);
        if internal.needs_split_before_insert(split.right_key.len()) {
            return Self::split_internal_then_insert(
                internal,
                split.right_key.clone(),
                split.right_page_num,
            );
        }
        internal.inplace_insert(
            child_idx + 1,
            split.right_key.as_ref(),
            split.right_page_num,
        );
        let root = Node::Internal(internal);
        let root_num = Self::alloc_page(&root);
        return NodeResult::NonSplit(Self { root, root_num });
    }

    /// Returns a new higher tree due to the old root splitting.
    fn new_higher_tree(split: Split) -> Tree {
        let mut root = node::Internal::default();
        root.set_num_keys(2);
        root.set_child_pointer(0, split.left_page_num);
        root.set_child_pointer(1, split.right_page_num);
        root.set_key(0, split.left_key.as_ref());
        root.set_key(1, split.right_key.as_ref());
        let root = Node::Internal(root);
        let root_num = Self::alloc_page(&root);
        return Self { root, root_num };
    }

    /// Returns true if an internal node needs to update.
    ///
    /// Because an internal node models the range `[b, c)` with `b` as its
    /// first (i.e. 0th) key, updating its range to `[a, c)` where `a < b`
    /// requires an update of its 0th key from `b` -> `a`.
    fn internal_needs_update(
        internal: &node::Internal,
        child_idx: u16,
        child_min_key: &[u8],
    ) -> bool {
        child_idx == 0 && internal.get_key(0) > child_min_key.as_ref()
    }

    /// Splits an internal node, then updates the left node.
    fn split_internal_then_update(
        internal: node::Internal,
        child_tree: &Tree,
        child_min_key: &[u8],
    ) -> NodeResult {
        let (mut left, right) = internal.copy_on_split();
        left.inplace_update(0, child_min_key.as_ref(), child_tree.root_num);
        let left_num = Self::alloc_page(&Node::Internal(left));
        let right_num = Self::alloc_page(&Node::Internal(right));
        return NodeResult::Split(Split {
            left_key: child_min_key.into(),
            left_page_num: left_num,
            right_key: right.get_key(0).into(),
            right_page_num: right_num,
        });
    }

    /// Splits an internal node, updates the left node, then inserts into either left or right.
    fn split_internal_then_update_and_insert(internal: node::Internal, split: Split) -> NodeResult {
        let update_key = split.left_key;
        let update_page_num = split.left_page_num;
        let insert_key = split.right_key;
        let insert_page_num = split.right_page_num;
        let (mut top_left, mut top_right) = internal.copy_on_split();
        top_left.inplace_update(0, update_key.as_ref(), update_page_num);
        if insert_key.as_ref() < top_right.get_key(0) {
            top_left.inplace_insert(
                top_left.lookup_insert(insert_key.as_ref()),
                insert_key.as_ref(),
                insert_page_num,
            );
        } else {
            top_right.inplace_insert(
                top_right.lookup_insert(insert_key.as_ref()),
                insert_key.as_ref(),
                insert_page_num,
            );
        }
        let top_left_num = Self::alloc_page(&Node::Internal(top_left));
        let top_right_num = Self::alloc_page(&Node::Internal(top_right));
        return NodeResult::Split(Split {
            left_key: top_left.get_key(0).into(),
            left_page_num: top_left_num,
            right_key: top_right.get_key(0).into(),
            right_page_num: top_right_num,
        });
    }

    /// Splits an internal node, then inserts into either left or right.
    fn split_internal_then_insert(
        internal: node::Internal,
        insert_key: Rc<[u8]>,
        insert_page_num: u64,
    ) -> NodeResult {
        let (mut top_left, mut top_right) = internal.copy_on_split();
        if insert_key.as_ref() < top_right.get_key(0) {
            top_left.inplace_insert(
                top_left.lookup_insert(insert_key.as_ref()),
                insert_key.as_ref(),
                insert_page_num,
            );
        } else {
            top_right.inplace_insert(
                top_right.lookup_insert(insert_key.as_ref()),
                insert_key.as_ref(),
                insert_page_num,
            );
        }
        let top_left_num = Self::alloc_page(&Node::Internal(top_left));
        let top_right_num = Self::alloc_page(&Node::Internal(top_right));
        return NodeResult::Split(Split {
            left_key: top_left.get_key(0).into(),
            left_page_num: top_left_num,
            right_key: top_right.get_key(0).into(),
            right_page_num: top_right_num,
        });
    }

    /// Gets a page from disk.
    fn get_page(page_num: u64) -> Node {
        unimplemented!();
        todo!("make this return a result instead");
    }

    /// Allocates a page on disk with data, and returns its page number.
    fn alloc_page(node: &Node) -> u64 {
        unimplemented!();
        todo!("make this return a result instead");
    }

    /// Deallocates a page on disk.
    fn dealloc_page(page_num: u64) {
        unimplemented!();
        todo!("make this return a result instead");
    }
}
