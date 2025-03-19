use std::rc::Rc;

use crate::bplus_node as node;

enum InsertResult {
    Normal {
        tree: Tree,
        min_key: Rc<[u8]>,
    },
    Split {
        left_key: Rc<[u8]>,
        left_page_num: u64,
        right_key: Rc<[u8]>,
        right_page_num: u64,
    },
}

struct Tree {
    root: node::Node,
    root_num: u64,
}

impl Tree {
    fn insert(self, key: &[u8], val: &[u8]) -> Tree {
        assert!(matches!(&self.root, node::Node::Internal(_)));
        match Self::insert_helper(self.root, key, val) {
            InsertResult::Normal { tree, min_key: _ } => {
                return tree;
            }
            InsertResult::Split {
                left_key: child1_key,
                left_page_num: child1_page_num,
                right_key: child2_key,
                right_page_num: child2_page_num,
            } => {
                // We need a higher root.
                let mut new_root = node::Internal::default();
                new_root.set_num_keys(2);
                new_root.set_child_pointer(0, child1_page_num);
                new_root.set_child_pointer(1, child2_page_num);
                new_root.set_key(0, child1_key.as_ref());
                new_root.set_key(1, child2_key.as_ref());
                let new_root = node::Node::Internal(new_root);
                let root_num = Self::alloc_page(&new_root);
                return Self {
                    root: new_root,
                    root_num,
                };
            }
        }
    }

    fn insert_helper(root: node::Node, key: &[u8], val: &[u8]) -> InsertResult {
        match root {
            node::Node::Leaf(mut leaf) => {
                // Case 1 - split:
                // * Split, then insert into left or right.
                // * Allocate left and right.
                // * Return InsertResult::Split.
                // Case 2 - normal:
                // * Insert into node.
                // * Allocate node.
                // * Return InsertResult::Normal.
                if leaf.needs_split(node::Revision::Insert, key.len() + val.len()) {
                    let (mut left, mut right) = leaf.copy_on_split();
                    // Insert into left or right?
                    let left_num;
                    let right_num;
                    if key < right.get_key(0) {
                        left.inplace_insert(left.lookup_insert(key), key, val);
                        left_num = Self::alloc_page(&node::Node::Leaf(left));
                        right_num = Self::alloc_page(&node::Node::Leaf(right));
                    } else {
                        left_num = Self::alloc_page(&node::Node::Leaf(left));
                        right.inplace_insert(right.lookup_insert(key), key, val);
                        right_num = Self::alloc_page(&node::Node::Leaf(right));
                    }
                    return InsertResult::Split {
                        left_key: left.get_key(0).into(),
                        left_page_num: left_num,
                        right_key: right.get_key(0).into(),
                        right_page_num: right_num,
                    };
                }
                leaf.inplace_insert(leaf.lookup_insert(key), key, val);
                let root_num = Self::alloc_page(&root);
                let min_key: Rc<[u8]> = leaf.get_key(0).into();
                return InsertResult::Normal {
                    tree: Self { root, root_num },
                    min_key,
                };
            }
            node::Node::Internal(mut internal) => {
                // Find child to call insert_helper into.
                // Case A - normal:
                // * Maybe update key.
                // * Update child pointer.
                // Case B - child split:
                // * Maybe update child1_key.
                // * Update child1_pointer.
                // * Insert child2_key.
                // * Insert child2_pointer.
                // Split?
                // Case 1 - split:
                // * Split to left and right.
                // * Allocate left and right.
                // * Return InsertResult::Split.
                // Case 2 - normal:
                // * Allocate node.
                // * Return InsertResult::Normal.
                let child_idx = internal.lookup_child(key);
                let child_num = internal.get_child_pointer(child_idx);
                let child = Self::get_page(child_num);
                let result = Self::insert_helper(child, key, val);
                match result {
                    InsertResult::Normal { tree, min_key } => {
                        if child_idx == 0 && internal.get_key(0) > min_key.as_ref() {
                            if internal.needs_split(
                                (min_key.as_ref().len() as isize)
                                    - (internal.get_key(0).len() as isize),
                            ) {
                                let (mut left, mut right) = internal.copy_on_split();
                                left.inplace_update(0, min_key.as_ref(), tree.root_num);
                                let left_num = Self::alloc_page(&node::Node::Internal(left));
                                let right_num = Self::alloc_page(&node::Node::Internal(right));
                                return InsertResult::Split {
                                    left_key: min_key,
                                    left_page_num: left_num,
                                    right_key: right.get_key(0).into(),
                                    right_page_num: right_num,
                                };
                            }
                            internal.inplace_update(0, min_key.as_ref(), tree.root_num);
                        } else {
                            internal.set_child_pointer(child_idx, tree.root_num);
                        }
                        let new_root_num = Self::alloc_page(&root);
                        let new_min_key = internal.get_key(0);
                        return InsertResult::Normal {
                            tree: Self {
                                root: root,
                                root_num: new_root_num,
                            },
                            min_key: new_min_key.into(),
                        };
                    }
                    InsertResult::Split {
                        left_key,
                        left_page_num,
                        right_key,
                        right_page_num,
                    } => {
                        if child_idx == 0 && internal.get_key(0) > left_key.as_ref() {
                            if internal.needs_split(
                                (left_key.len() as isize) - (internal.get_key(0).len() as isize)
                                    + 8
                                    + (right_key.len() as isize),
                            ) {
                                let (mut top_left, mut top_right) = internal.copy_on_split();
                                top_left.inplace_update(0, left_key.as_ref(), left_page_num);
                                if right_key.as_ref() < top_right.get_key(0) {
                                    top_left.inplace_insert(
                                        top_left.lookup_insert(right_key.as_ref()),
                                        right_key.as_ref(),
                                        right_page_num,
                                    );
                                } else {
                                    top_right.inplace_insert(
                                        top_right.lookup_insert(right_key.as_ref()),
                                        right_key.as_ref(),
                                        right_page_num,
                                    );
                                }
                                let top_left_num =
                                    Self::alloc_page(&node::Node::Internal(top_left));
                                let top_right_num =
                                    Self::alloc_page(&node::Node::Internal(top_right));
                                return InsertResult::Split {
                                    left_key: top_left.get_key(0).into(),
                                    left_page_num: top_left_num,
                                    right_key: top_right.get_key(0).into(),
                                    right_page_num: top_right_num,
                                };
                            }
                            internal.inplace_update(0, left_key.as_ref(), left_page_num);
                        } else {
                            internal.set_child_pointer(child_idx, left_page_num);
                        }
                        if internal.needs_split(8 + (right_key.len() as isize)) {
                            let (mut top_left, mut top_right) = internal.copy_on_split();
                            if right_key.as_ref() < top_right.get_key(0) {
                                top_left.inplace_insert(
                                    top_left.lookup_insert(right_key.as_ref()),
                                    right_key.as_ref(),
                                    right_page_num,
                                );
                            } else {
                                top_right.inplace_insert(
                                    top_right.lookup_insert(right_key.as_ref()),
                                    right_key.as_ref(),
                                    right_page_num,
                                );
                            }
                            let top_left_num = Self::alloc_page(&node::Node::Internal(top_left));
                            let top_right_num = Self::alloc_page(&node::Node::Internal(top_right));
                            return InsertResult::Split {
                                left_key: top_left.get_key(0).into(),
                                left_page_num: top_left_num,
                                right_key: top_right.get_key(0).into(),
                                right_page_num: top_right_num,
                            };
                        }
                        internal.inplace_insert(child_idx + 1, right_key.as_ref(), right_page_num);
                        let new_root_num = Self::alloc_page(&root);
                        let min_key = internal.get_key(0);
                        return InsertResult::Normal {
                            tree: Self {
                                root: root,
                                root_num: new_root_num,
                            },
                            min_key: min_key.into(),
                        };
                    }
                }
            }
        }
    }

    /// Gets a page from disk.
    fn get_page(page_num: u64) -> node::Node {
        unimplemented!();
        todo!("make this return a result instead");
    }

    /// Allocates a page on disk with data, and returns its page number.
    fn alloc_page(node: &node::Node) -> u64 {
        unimplemented!();
        todo!("make this return a result instead");
    }

    /// Deallocates a page on disk.
    fn dealloc_page(page_num: u64) {
        unimplemented!();
        todo!("make this return a result instead");
    }
}
