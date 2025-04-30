//! list node format:
//!
//! ```ignore
//! | next | pointers | unused |
//! |  8B  |   n*8B   |   ...  |
//! ```
//!
//! `next` points to the next list node. `pointers` points to free pages.

use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{consts, mmap::Page};

use super::{Guard as _, ReadOnlyPage, ReusedPageType, Writer};

const FREE_LIST_CAP: usize = (consts::PAGE_SIZE - 8) / 8;

/// An in-memory container of metadata about the free list.
pub struct FreeList {
    head_page_num: usize,
    head_seq: usize,
    tail_page_num: usize,
    tail_seq: usize,
    max_seq: usize,
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
        // Safety: tail_page_num points to a page that is guaranteed to never
        // have any concurrent readers accessing it.
        let mut node: ListNode<'_, _> = unsafe { writer.overwrite_page(self.tail_page_num) }.into();
        node.set_pointer(seq_to_index(self.tail_seq), ptr);
        self.tail_seq += 1;

        // add a new tail node if it's full (the list is never empty)
        if seq_to_index(self.tail_seq) != 0 {
            return;
        }

        // try to reuse from the list head
        let (next, head) = self.pop_helper(writer);
        let next = next.unwrap_or_else(|| {
            // or allocate a new node by appending
            let page = writer.new_page();
            let page = writer.write_page(page);
            page.page_num
        });

        // link to the new tail node
        node.set_next(next);
        self.tail_page_num = next;
        // Safety: tail_page_num points to a page that is guaranteed to never
        // have any concurrent readers accessing it.
        node = unsafe { writer.overwrite_page(self.tail_page_num) }.into();

        // also add the head node if it's removed
        if let Some(head) = head {
            node.set_pointer(0, head);
            self.tail_seq += 1;
        }
    }

    /// Ensures that pages added to the free list cannot be reused
    /// in the same transaction.
    pub fn set_max_seq(&mut self) {
        self.max_seq = self.tail_seq
    }

    // remove 1 item from the head node, and remove the head node if empty.
    fn pop_helper(&mut self, writer: &Writer) -> (Option<usize>, Option<usize>) {
        if self.head_seq == self.tail_seq {
            return (None, None); // cannot advance
        }
        let node: ListNode<'_, _> = writer.read_page(self.head_page_num).into();
        let ptr = node.get_pointer(seq_to_index(self.head_seq));
        self.head_seq += 1;

        // move to the next one if the head node is empty
        let mut head = None;
        if seq_to_index(self.head_seq) == 0 {
            head = Some(self.head_page_num);
            self.head_page_num = node.get_next();
        }
        (Some(ptr), head)
    }
}

#[inline]
fn seq_to_index(seq: usize) -> usize {
    seq % FREE_LIST_CAP
}

/// A linked list node representing free pages that can be used to
/// store B+ tree nodes.
struct ListNode<'a, P: Deref<Target = [u8]>> {
    _phantom: PhantomData<&'a ()>,
    page: P,
}

impl<P: Deref<Target = [u8]>> ListNode<'_, P> {
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

impl<'w> From<ReadOnlyPage<'w>> for ListNode<'w, ReadOnlyPage<'w>> {
    fn from(page: ReadOnlyPage<'w>) -> Self {
        ListNode {
            _phantom: PhantomData,
            page,
        }
    }
}

impl<P: DerefMut<Target = [u8]>> ListNode<'_, P> {
    /// Sets the next pointer.
    fn set_next(&mut self, next: usize) {
        self.page[0..8].copy_from_slice(&next.to_be_bytes());
    }

    /// Sets the value of `i`th pointer.
    fn set_pointer(&mut self, i: usize, ptr: usize) {
        self.page[8 + 8 * i..8 + 8 * (i + 1)].copy_from_slice(&ptr.to_le_bytes());
    }
}

impl<'w> From<Page<'w, ReusedPageType>> for ListNode<'w, Page<'w, ReusedPageType>> {
    fn from(page: Page<'w, ReusedPageType>) -> Self {
        ListNode {
            _phantom: PhantomData,
            page,
        }
    }
}
