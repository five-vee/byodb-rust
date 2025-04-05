//! A memory pool that manages fixed-size buffers (4KB and 8KB) using a
//! pre-allocated memory pool with a fallback to heap allocations.
//! Uses a bitmap to track free pages, allowing for dynamic allocation and
//! implicit merging of freed blocks. Dereferencing pooled buffers does not lock the pool mutex.

use crate::tree::consts;
use core::fmt;
use std::ops::{Deref, DerefMut};

/// The standard page size used for allocations (4096 bytes).
const PAGE_SIZE: usize = consts::PAGE_SIZE;
/// The large page size used for allocations (8192 bytes).
const LARGE_PAGE_SIZE: usize = 2 * PAGE_SIZE;

pub trait BufferStore: Clone + fmt::Debug {
    type B: Buffer;

    fn get_buf(&self, requested: usize) -> Self::B;
}

#[derive(Clone, Debug)]
pub struct Heap;

impl BufferStore for Heap {
    type B = HeapBuffer;

    fn get_buf(&self, requested: usize) -> Self::B {
        let alloc_size = if requested <= PAGE_SIZE {
            PAGE_SIZE
        } else if requested <= LARGE_PAGE_SIZE {
            LARGE_PAGE_SIZE
        } else {
            panic!("HeapStore doesn't support buffer size > {LARGE_PAGE_SIZE}");
        };
        HeapBuffer {
            buf: vec![0u8; alloc_size].into_boxed_slice(),
        }
    }
}

pub trait Buffer: fmt::Debug + Deref<Target = [u8]> + DerefMut<Target = [u8]> {}

#[derive(Debug)]
pub struct HeapBuffer {
    buf: Box<[u8]>,
}

impl Deref for HeapBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.buf.deref()
    }
}

impl DerefMut for HeapBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buf.deref_mut()
    }
}

impl Buffer for HeapBuffer {}
