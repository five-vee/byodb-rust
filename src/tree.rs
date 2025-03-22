use std::rc::Rc;

use crate::node;

pub type Result<T> = std::result::Result<T, ()>;

enum Upsert {
    Intact(Tree),
    Split { left: Tree, right: Tree },
}

impl Upsert {
    fn alloc(u: node::Upsert) -> Result<Self> {
        match u {
            node::Upsert::Intact(root) => {
                let page_num = Tree::alloc_page(&root)?;
                Ok(Upsert::Intact(Tree { root, page_num }))
            }
            node::Upsert::Split { left, right } => {
                let left_page_num = Tree::alloc_page(&left)?;
                let right_page_num = Tree::alloc_page(&right)?;
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

pub enum Deletion {
    Empty,
    Sufficient(Tree),
    Split { left: Tree, right: Tree },
    Underflow(Tree),
}

impl Deletion {
    fn alloc(d: node::Deletion) -> Result<Self> {
        match d {
            node::Deletion::Empty => Ok(Deletion::Empty),
            node::Deletion::Sufficient(root) => {
                let page_num = Tree::alloc_page(&root)?;
                Ok(Deletion::Sufficient(Tree { root, page_num }))
            }
            node::Deletion::Split { left, right } => {
                let left_page_num = Tree::alloc_page(&left)?;
                let right_page_num = Tree::alloc_page(&right)?;
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
                let page_num = Tree::alloc_page(&root)?;
                Ok(Deletion::Underflow(Tree { root, page_num }))
            }
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
            Upsert::Intact(tree) => Ok(tree),
            Upsert::Split { left, right } => Self::new_root(left, right)
        }
    }

    fn insert_helper(&self, key: &[u8], val: &[u8]) -> Result<Upsert> {
        match &self.root {
            // Base case
            node::Node::Leaf(leaf) => Ok(Upsert::alloc(leaf.insert(key, val)?)?),
            // Recursive case
            node::Node::Internal(internal) => {
                // Find which child to recursively insert into.
                let (child_idx, child_num) = internal.find_child_pointer(key)?;
                let child = Self::get_page(child_num)?;
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

    pub fn update(&self, key: &[u8], val: &[u8]) -> Result<Self> {
        match self.update_helper(key, val)? {
            Upsert::Intact(tree) => Ok(tree),
            Upsert::Split { left, right } => Self::new_root(left, right)
        }
    }

    fn update_helper(&self, key: &[u8], val: &[u8]) -> Result<Upsert> {
        match &self.root {
            // Base case
            node::Node::Leaf(leaf) => Ok(Upsert::alloc(leaf.update(key, val)?)?),
            // Recursive case
            node::Node::Internal(internal) => {
                // Find which child to recursively update at.
                let (child_idx, child_num) = internal.find_child_pointer(key)?;
                let child = Self::get_page(child_num)?;
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

    pub fn delete(&self, key: &[u8]) -> Result<Self> {
        match self.delete_helper(key)? {
            Deletion::Empty => {
                let root = node::Node::Leaf(node::Leaf::new(&[], &[])?);
                let page_num = Self::alloc_page(&root)?;
                Ok(Tree { root, page_num })
            }
            Deletion::Split { left, right } => Self::new_root(left, right),
            Deletion::Sufficient(tree) => Ok(tree),
            Deletion::Underflow(tree) => Ok(tree)
        }
    }

    fn delete_helper(&self, key: &[u8]) -> Result<Deletion> {
        match &self.root {
            // Base case
            node::Node::Leaf(leaf) => Ok(Deletion::alloc(leaf.delete(key)?)?),
            // Recursive case
            node::Node::Internal(internal) => {
                // Find which child to recursively delete from.
                let (child_idx, child_num) = internal.find_child_pointer(key)?;
                let child = Self::get_page(child_num)?;
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
                            Self::try_fix_underflow(internal, child_idx, child)?;
                        let d = internal.merge_delta(delta)?;
                        return Ok(Deletion::alloc(d)?);
                    }
                }
            }
        }
    }

    fn try_fix_underflow(
        internal: &node::Internal,
        child_idx: u16,
        child: Tree,
    ) -> Result<node::DeletionDelta> {
        // 1. Left (child_idx - 1) exists && left is sufficient
        //    -> [(child_idx - 1, Some(new_left)), (child_idx, Some(new_child))]
        // 2. Left (child_idx - 1) exists && left is underflow
        //    -> [(child_idx - 1, Some(new_left)), (child_idx, None)]
        if child_idx > 0 {
            let left_page_num = internal.get_child_pointer(child_idx - 1)?;
            let left = Self::get_page(left_page_num)?;
            if node::sufficient_steal(&left, &child.root) {
                let (new_left, new_child) = node::steal(left, child.root)?;
                let new_left_page_num = Self::alloc_page(&new_left)?;
                let new_child_page_num = Self::alloc_page(&new_child)?;
                return Ok([
                    (child_idx - 1, Some(new_left_page_num)),
                    (child_idx, Some(new_child_page_num)),
                ]
                .into());
            }
            if node::sufficient_merge(&child.root, &left) {
                let new_left = node::merge(child.root, left)?;
                let new_left_page_num = Self::alloc_page(&new_left)?;
                return Ok([(child_idx - 1, Some(new_left_page_num)), (child_idx, None)].into());
            }
        }

        // 3. Right (child_idx + 1) exists && right is sufficient
        //    -> [(child_idx, Some(new_child)), (child_idx + 1, Some(new_right))]
        // 4. Right (child_idx - 1) exists && right is underflow
        //    -> [(child_idx, None), (child_idx + 1, new_right)]
        let right_page_num = internal.get_child_pointer(child_idx + 1)?;
        let right = Self::get_page(right_page_num)?;
        if node::sufficient_steal(&right, &child.root) {
            let (new_right, new_child) = node::steal(right, child.root)?;
            let new_right_page_num = Self::alloc_page(&new_right)?;
            let new_child_page_num = Self::alloc_page(&new_child)?;
            return Ok([
                (child_idx, Some(new_child_page_num)),
                (child_idx + 1, Some(new_right_page_num)),
            ]
            .into());
        }
        if node::sufficient_merge(&child.root, &right) {
            let new_right: node::Node = node::merge(child.root, right)?;
            let new_right_page_num = Self::alloc_page(&new_right)?;
            return Ok([(child_idx, None), (child_idx + 1, Some(new_right_page_num))].into());
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

    fn new_root(left: Tree, right: Tree) -> Result<Self> {
        let keys = &[left.root.get_key(0), right.root.get_key(0)];
        let child_pointers = &[left.page_num, right.page_num];
        let root = node::Internal::new(keys, child_pointers);
        let root = node::Node::Internal(root);
        let page_num = Self::alloc_page(&root)?;
        Ok(Self { root, page_num })
    }

    fn get_page(page_num: u64) -> Result<node::Node> {
        unimplemented!();
    }

    fn alloc_page(node: &node::Node) -> Result<u64> {
        unimplemented!();
    }
}
