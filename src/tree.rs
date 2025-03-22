use std::rc::Rc;

use crate::node;

pub type Result<T> = std::result::Result<T, ()>;

enum TreeResult {
    NonSplit(Tree),
    Split(Tree, Tree),
}

impl TreeResult {
    fn alloc(nr: node::NodeResult) -> Self {
        match nr {
            node::NodeResult::NonSplit(root) => {
                let page_num = Tree::alloc_page(&root);
                TreeResult::NonSplit(Tree { root, page_num })
            }
            node::NodeResult::Split(left, right) => {
                let left_page_num = Tree::alloc_page(&left);
                let right_page_num = Tree::alloc_page(&right);
                let left_tree = Tree {
                    root: left,
                    page_num: left_page_num,
                };
                let right_tree = Tree {
                    root: right,
                    page_num: right_page_num,
                };
                TreeResult::Split(left_tree, right_tree)
            }
        }
    }

    fn children(self, i: u16) -> Rc<[node::ChildEntry]> {
        match self {
            TreeResult::NonSplit(tree) => [node::ChildEntry {
                parent_i: Some(i),
                min_key: tree.root.get_key(0).into(),
                page_num: tree.page_num,
            }].into(),
            TreeResult::Split(left, right) => [
                node::ChildEntry {
                    parent_i: Some(i),
                    min_key: left.root.get_key(0).into(),
                    page_num: left.page_num,
                },
                node::ChildEntry {
                    parent_i: None,
                    min_key: right.root.get_key(0).into(),
                    page_num: right.page_num,
                },
            ].into(),
        }
    }
}

pub struct Tree {
    root: node::Node,
    page_num: u64,
}

impl Tree {
    pub fn insert(&self, key: &[u8], val: &[u8]) -> Result<Self> {
        match self.insert_helper(key, val)? {
            TreeResult::NonSplit(tree) => Ok(tree),
            TreeResult::Split(left, right) => {
                let keys = &[left.root.get_key(0), right.root.get_key(0)];
                let child_pointers = &[left.page_num, right.page_num];
                let root = node::Internal::new(keys, child_pointers);
                let root = node::Node::Internal(root);
                let page_num = Self::alloc_page(&root);
                Ok(Self { root, page_num })
            }
        }
    }

    fn insert_helper(&self, key: &[u8], val: &[u8]) -> Result<TreeResult> {
        match &self.root {
            // Base case
            node::Node::Leaf(leaf) => Ok(TreeResult::alloc(leaf.insert(key, val)?)),
            // Recursive case
            node::Node::Internal(internal) => {
                // Find which child to recursively insert into.
                let (child_idx, child_num) = internal.find_child_pointer(key)?;
                let child = Self::get_page(child_num);
                let child = Self {
                    root: child,
                    page_num: child_num,
                };
                let tr = child.insert_helper(key, val)?;
                let new_children = tr.children(child_idx);
                let ir = internal.connect_children(new_children.as_ref())?;
                Ok(TreeResult::alloc(ir))
            }
        }
    }

    pub fn update(&self, key: &[u8], val: &[u8]) -> Result<Self> {
        match self.update_helper(key, val)? {
            TreeResult::NonSplit(tree) => Ok(tree),
            TreeResult::Split(left, right) => {
                let keys = &[left.root.get_key(0), right.root.get_key(0)];
                let child_pointers = &[left.page_num, right.page_num];
                let root = node::Internal::new(keys, child_pointers);
                let root = node::Node::Internal(root);
                let page_num = Self::alloc_page(&root);
                Ok(Self { root, page_num })
            }
        }
    }

    fn update_helper(&self, key: &[u8], val: &[u8]) -> Result<TreeResult> {
        match &self.root {
            // Base case
            node::Node::Leaf(leaf) => Ok(TreeResult::alloc(leaf.update(key, val)?)),
            // Recursive case
            node::Node::Internal(internal) => {
                // Find which child to recursively update at.
                let (child_idx, child_num) = internal.find_child_pointer(key)?;
                let child = Self::get_page(child_num);
                let child = Self {
                    root: child,
                    page_num: child_num,
                };
                let tr = child.update_helper(key, val)?;
                let new_children = tr.children(child_idx);
                let ir = internal.connect_children(new_children.as_ref())?;
                Ok(TreeResult::alloc(ir))
            }
        }
    }

    pub fn delete(&self, key: &[u8]) -> Result<Self> {
        unimplemented!();
    }

    fn get_page(page_num: u64) -> node::Node {
        unimplemented!();
    }

    fn alloc_page(node: &node::Node) -> u64 {
        unimplemented!();
    }
}
