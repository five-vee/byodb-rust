//! The free list is a linked list, where each node is itself a page in the
//! mmap, and each node holds pointers to pages that can be freely used
//! to write new B+ tree pages.
//!
//! The [`Writer`] is responsible for cleaning up garbage unreferenced
//! memory-mapped pages and adding them to the [`FreeList`]. Any page
//! referenced by this list is available for re-use, thus saving on file size.
//!
//! List node format:
//!
//! ```ignore
//! | next | pointers | unused |
//! |  8B  |   n*8B   |   ...  |
//! ```
//!
//! `next` points to the next list node. `pointers` points to free pages.
//!
//! ## Torn writes
//!
//! A write to a free list page is considered "torn" when the write doesn't
//! complete, leaving the page in a partial corrupted state.
//!
//! This is fine because:
//!
//! * The [`Writer::flush`] call must successfully update the [`MetaNode`]
//!   on disk with the updated [`FreeList`] for it to be accessible by
//!   future readers/writers. That is, a partially written free-list is not
//!   accessible upon crash recovery.
//! * _(TODO)_ Upon opening a database file, the free-list is re-built
//!   from scratch, thus reclaiming garbage introduced in an uncommitted
//!   write transaction prior to crashing.

use std::ops::{Deref, DerefMut};

use crate::core::{consts, mmap::WriterPage};

use super::{Guard as _, ImmutablePage as _, Writer, meta_node::MetaNode};

pub(crate) const FREE_LIST_CAP: usize = (consts::PAGE_SIZE - 8) / 8;
const INVALID_NEXT: usize = usize::MAX;

/// An in-memory container of metadata about the free list.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct FreeList {
    pub head_page: usize,
    pub head_seq: usize,
    pub tail_page: usize,
    pub tail_seq: usize,
}

impl FreeList {
    /// Gets 1 item from the list head. Returns None if no head to pop.
    pub fn pop_head(&mut self, writer: &Writer) -> Option<usize> {
        let (ptr, head) = self.pop_helper(writer);
        if let Some(head) = head {
            self.push_tail(writer, head);
        }
        ptr
    }

    /// Pushes a pointer to the list tail.
    pub fn push_tail(&mut self, writer: &Writer, ptr: usize) {
        // add it to the tail node
        // Safety: tail_page points to a page that is guaranteed to never
        // have any concurrent readers accessing it.
        let mut node: ListNode<_> = unsafe { writer.overwrite_page(self.tail_page) }.into();
        node.set_pointer(seq_to_index(self.tail_seq), ptr);
        self.tail_seq += 1;

        if seq_to_index(self.tail_seq) != 0 {
            return;
        }
        // add a new tail node if it's full (the list is never empty)

        // try to reuse from the list head
        let (next, head) = self.pop_helper(writer);
        let next = next.unwrap_or_else(|| {
            // or allocate a new node by appending
            writer.new_page().read_only().page_num()
        });

        // link to the new tail node
        node.set_next(next);
        self.tail_page = next;
        // Safety: tail_page points to a page that is guaranteed to never
        // have any concurrent readers accessing it.
        // Also, &mut self ensures there is only one mutable reference to the
        // underlying page.
        let mut node_page = unsafe { writer.overwrite_page(self.tail_page) };
        init_empty_list_node(&mut node_page);
        node = node_page.into();

        // also add the head node if it's removed
        if let Some(head) = head {
            node.set_pointer(seq_to_index(self.tail_seq), head);
            self.tail_seq += 1;
        }
    }

    /// Removes 1 item from the head node,
    /// and may remove the head node if empty.
    fn pop_helper(&mut self, writer: &Writer) -> (Option<usize>, Option<usize>) {
        if self.head_seq == self.tail_seq {
            return (None, None); // cannot advance
        }
        // A linked list with 0 nodes implies nasty special cases.
        // In practice, it’s easier to design the linked list to have at least
        // 1 node than to deal with special cases.
        if self.head_seq == self.tail_seq - 1 {
            return (None, None); // cannot advance
        }
        // Safety: &mut self ensures exclusive access to the head list node page.
        let node: ListNode<_> = unsafe { writer.read_page(self.head_page) }.into();
        let ptr = node.get_pointer(seq_to_index(self.head_seq));
        self.head_seq += 1;

        // move to the next one if the head node is empty
        let mut head = None;
        if seq_to_index(self.head_seq) == 0 {
            head = Some(self.head_page);
            self.head_page = node.get_next();
            assert_ne!(self.head_page, INVALID_NEXT);
        }
        (Some(ptr), head)
    }
}

impl Default for FreeList {
    /// Creates the initial empty free list node.
    fn default() -> Self {
        FreeList {
            head_page: 1,
            head_seq: 0,
            tail_page: 1,
            tail_seq: 0,
        }
    }
}

impl From<MetaNode> for FreeList {
    fn from(node: MetaNode) -> Self {
        FreeList {
            head_page: node.head_page,
            head_seq: node.head_seq,
            tail_page: node.tail_page,
            tail_seq: node.tail_seq,
        }
    }
}

#[inline]
fn seq_to_index(seq: usize) -> usize {
    seq % FREE_LIST_CAP
}

/// A linked list node representing free pages that can be used to
/// store B+ tree nodes.
struct ListNode<P: Deref<Target = [u8]>> {
    page: P,
}

impl<P: Deref<Target = [u8]>> ListNode<P> {
    /// Gets the page num of the next linked list node.
    fn get_next(&self) -> usize {
        usize::from_le_bytes([
            self.page[0],
            self.page[1],
            self.page[2],
            self.page[3],
            self.page[4],
            self.page[5],
            self.page[6],
            self.page[7],
        ])
    }

    /// Gets the page num of `i`th pointer.
    fn get_pointer(&self, i: usize) -> usize {
        usize::from_le_bytes([
            self.page[8 + 8 * i],
            self.page[8 + 8 * i + 1],
            self.page[8 + 8 * i + 2],
            self.page[8 + 8 * i + 3],
            self.page[8 + 8 * i + 4],
            self.page[8 + 8 * i + 5],
            self.page[8 + 8 * i + 6],
            self.page[8 + 8 * i + 7],
        ])
    }
}

impl<'w> From<WriterPage<'w>> for ListNode<WriterPage<'w>> {
    fn from(page: WriterPage<'w>) -> Self {
        ListNode { page }
    }
}

impl<P: DerefMut<Target = [u8]>> ListNode<P> {
    /// Sets the next pointer.
    fn set_next(&mut self, next: usize) {
        self.page[0..8].copy_from_slice(&next.to_le_bytes());
    }

    /// Sets the value of `i`th pointer.
    fn set_pointer(&mut self, i: usize, ptr: usize) {
        self.page[8 + 8 * i..8 + 8 * (i + 1)].copy_from_slice(&ptr.to_le_bytes());
    }
}

/// Writes into page to initialize it as an empty list node.
pub fn init_empty_list_node(page: &mut [u8]) {
    let mut ln = ListNode { page };
    ln.set_next(INVALID_NEXT);
}
