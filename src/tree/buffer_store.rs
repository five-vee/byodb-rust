//! A memory pool that manages fixed-size buffers (4KB and 8KB) using a
//! pre-allocated memory arena with a fallback to heap allocations.
//! Uses a bitmap to track free pages, allowing for dynamic allocation and
//! implicit merging of freed blocks. Dereferencing pooled buffers does not lock the pool mutex.

use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;
use std::sync::{Arc, Mutex};
use crate::tree::node;

// --- Constants ---

/// The standard page size used for allocations (4096 bytes).
const PAGE_SIZE: usize = node::PAGE_SIZE;
/// The large page size used for allocations (8192 bytes).
const LARGE_PAGE_SIZE: usize = 2 * PAGE_SIZE;

// --- Statistics ---

/// Allocation statistics for an [`ArenaPool`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ArenaStats {
    /// Number of times a 4KB buffer was successfully allocated from the arena.
    pub arena_allocs_4k: usize,
    /// Number of times an 8KB buffer was successfully allocated from the arena.
    pub arena_allocs_8k: usize,
    /// Number of times allocation fell back to the heap.
    pub heap_allocs: usize,
}

// --- Pointer Wrapper for Send/Sync ---

/// A wrapper around a pointer to the arena's base address.
/// Marked Send+Sync under the assumption that the underlying memory is stable
/// and access control is managed by the allocator + PooledBuf lifetimes.
#[derive(Clone, Copy, Debug)]
struct ArenaBasePtr(NonNull<u8>); // Use NonNull<u8>

// SAFETY: This is safe iff:
// 1. The pointer points to a stable heap allocation (`Box<[u8]>`) that lives
//    at least as long as any PooledBuf using it (guaranteed by Arc).
// 2. The allocator logic correctly prevents aliasing mutable access via
//    different PooledBuf instances returned from get_buf.
// 3. Access via deref/deref_mut uses correct index/size from BufferOrigin.
unsafe impl Send for ArenaBasePtr {}
unsafe impl Sync for ArenaBasePtr {}

// --- Private Structs and Enums ---
#[derive(Debug)]
enum BufferOrigin {
    Arena {
        /// The starting byte index within the arena storage.
        index: usize,
        /// The allocated size (PAGE_SIZE or LARGE_PAGE_SIZE).
        size: usize,
        /// Base pointer of the arena allocation (obtained at allocation time).
        base_ptr: ArenaBasePtr, // Store the wrapped pointer
    },
    Heap {
        data: Box<[u8]>,
    },
}

/// Internal state of the Arena pool, protected by a `Mutex`.
#[derive(Debug)]
struct ArenaState {
    /// Owns the contiguous memory block for the arena.
    arena_storage: Box<[u8]>,
    /// Bitmap tracking allocation status of each PAGE_SIZE chunk.
    /// `false` = free, `true` = allocated.
    bitmap: Vec<bool>,
    /// Total number of PAGE_SIZE pages in the arena.
    num_pages: usize,
    /// Allocation statistics.
    stats: ArenaStats,
}

impl ArenaState {
    /// Creates a new Arena sized for `num_pages` of `PAGE_SIZE` each.
    fn new(num_pages: usize) -> Self {
        assert!(num_pages > 0, "Arena must have at least one page");

        let size_in_bytes = num_pages * PAGE_SIZE;
        let arena_storage = vec![0u8; size_in_bytes].into_boxed_slice();
        // Initialize bitmap with all pages marked as free.
        let bitmap = vec![false; num_pages];

        println!(
            "Arena initialized: {} Pages ({} bytes), All pages free.",
            num_pages, size_in_bytes,
        );

        ArenaState {
            arena_storage,
            bitmap,
            num_pages,
            stats: ArenaStats::default(),
        }
    }

    // Helper to get base pointer (avoids calling as_mut_ptr repeatedly)
    // This doesn't *need* a lock if the Box reference itself is stable,
    // but getting it under lock is simplest during initialization/allocation.
    fn get_base_ptr(&self) -> Option<NonNull<u8>> {
        NonNull::new(self.arena_storage.as_ptr() as *mut u8) // Cast const to mut for NonNull<u8>
   }

    /// Scans the bitmap to find a free chunk, marks it allocated, and updates stats.
    fn get_arena_chunk(&mut self, requested_size: usize) -> Option<(usize, usize)> {
        // Determine number of contiguous pages needed
        let pages_needed = if requested_size <= PAGE_SIZE {
            1
        } else {
            // requested_size must be <= LARGE_PAGE_SIZE asserted in get_buf
            2
        };

        // --- Scan Bitmap ---
        let mut found_idx: Option<usize> = None;
        // Iterate checking for `pages_needed` contiguous free slots
        for i in 0..=(self.num_pages.saturating_sub(pages_needed)) {
            let mut contiguous_free = true;
            for j in 0..pages_needed {
                if self.bitmap[i + j] {
                    // Found an allocated page, this sequence won't work
                    contiguous_free = false;
                    break; // Break inner loop, continue outer scan
                }
            }

            if contiguous_free {
                // Found a suitable sequence starting at index i
                found_idx = Some(i);
                break; // Stop scanning
            }
            // If contiguous_free is false, the outer loop continues scanning
        }
        // --- End Scan ---


        if let Some(start_page_idx) = found_idx {
            // --- Mark pages as allocated ---
            for j in 0..pages_needed {
                self.bitmap[start_page_idx + j] = true;
            }

            // --- Update Stats ---
            let allocated_size = pages_needed * PAGE_SIZE;
            if allocated_size == PAGE_SIZE {
                self.stats.arena_allocs_4k += 1;
            } else {
                self.stats.arena_allocs_8k += 1;
            }

            // --- Return byte index and allocated size ---
            let byte_index = start_page_idx * PAGE_SIZE;
            Some((byte_index, allocated_size))
        } else {
            // No suitable chunk found in the arena
            None
        }
    }

    /// Marks the corresponding pages in the bitmap as free.
    fn return_arena_chunk(&mut self, byte_index: usize, size: usize) {
        // Validate inputs (basic checks)
        assert!(byte_index % PAGE_SIZE == 0, "Returned index not page aligned");
        assert!(
            size == PAGE_SIZE || size == LARGE_PAGE_SIZE,
            "Invalid size returned"
        );
        assert!(
            byte_index + size <= self.num_pages * PAGE_SIZE,
            "Returned block out of bounds"
        );

        let start_page_idx = byte_index / PAGE_SIZE;
        let pages_to_free = size / PAGE_SIZE;

        for j in 0..pages_to_free {
            let current_page_idx = start_page_idx + j;
            if !self.bitmap[current_page_idx] {
                // This indicates a double-free or logic error
                eprintln!(
                    "WARNING: Attempting to free already free page at index {}",
                    current_page_idx
                );
                // Optionally panic here, or just log and continue
            }
            self.bitmap[current_page_idx] = false; // Mark as free
        }
    }

    /// Records a heap allocation fallback.
    fn record_heap_alloc(&mut self) {
        self.stats.heap_allocs += 1;
    }

    /// Gets a copy of the current statistics.
    fn get_stats(&self) -> ArenaStats {
        self.stats
    }
}

// --- Public Pool Structure ---
pub trait BufferStore : Clone {
    fn get_buf(&self, requested_size: usize) -> PooledBuf;
}

/// A thread-safe memory pool managing fixed-size buffers (`4KB` or `8KB`).
/// Uses a bitmap allocator over a pre-allocated memory arena.
#[derive(Clone, Debug)]
pub struct ArenaPool {
    state: Arc<Mutex<ArenaState>>,
}
// Mark as Send + Sync because ArenaState is Send (contains Box, Vec<bool>, usize, Stats)
unsafe impl Send for ArenaPool {}
unsafe impl Sync for ArenaPool {}

impl ArenaPool {
    /// Creates a new pool with an arena sized for `num_pages` of `PAGE_SIZE`.
    pub fn new(num_pages: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(ArenaState::new(num_pages))),
        }
    }

    /// Returns a copy of the current allocation statistics.
    pub fn get_stats(&self) -> ArenaStats {
        let state = self.state.lock().unwrap();
        state.get_stats()
    }

    /// Used internally by PooledBuf Drop for Arena buffers.
    fn return_arena_chunk(&self, index: usize, size: usize) {
        let mut state = self.state.lock().unwrap();
        state.return_arena_chunk(index, size);
    }
}

impl BufferStore for ArenaPool {
     /// Gets a buffer from the pool, preferring the arena, falling back to the heap.
     fn get_buf(&self, requested_size: usize) -> PooledBuf {
        assert!(
            requested_size > 0 && requested_size <= LARGE_PAGE_SIZE,
            "Requested buffer size ({}) invalid or exceeds maximum ({})",
            requested_size,
            LARGE_PAGE_SIZE
        );

        { // Lock scope for allocation attempt
            let mut state = self.state.lock().unwrap();
            if let Some((index, actual_size)) = state.get_arena_chunk(requested_size) {
                // Get base pointer *once* under lock
                if let Some(base_ptr) = state.get_base_ptr() {
                     let origin = BufferOrigin::Arena {
                        index,
                        size: actual_size,
                        base_ptr: ArenaBasePtr(base_ptr), // Store wrapped NonNull ptr
                    };
                    return PooledBuf { pool: self.clone(), origin };
                } else {
                    // This should ideally not happen if arena_storage exists
                    eprintln!("Error: Failed to get base pointer from arena storage!");
                    // Fall through to heap allocation as a safe fallback
                     state.record_heap_alloc(); // Record even if base_ptr failed
                }
            } else {
                 state.record_heap_alloc(); // Record heap fallback if chunk not found
            }
        } // Lock released

        // --- Heap Fallback ---
        let alloc_size = if requested_size <= PAGE_SIZE { PAGE_SIZE } else { LARGE_PAGE_SIZE };
        println!(
            "Arena pool falling back to HEAP allocation for size {} (requested {})",
            alloc_size, requested_size
        );
        let heap_data = vec![0u8; alloc_size].into_boxed_slice();
        let origin = BufferOrigin::Heap { data: heap_data };
        PooledBuf { pool: self.clone(), origin }
    }
}

// --- Public Buffer Handle ---

/// A smart pointer representing a buffer allocated from an [`ArenaPool`].
///
/// This buffer either points to a slice within the pool's arena or owns
/// a heap allocation (`Box<[u8]>`).
///
/// It implements `Deref` and `DerefMut` for easy access to the underlying byte slice.
/// When `PooledBuf` is dropped, it automatically returns the memory to the
/// arena's free list or lets the heap allocation be freed, respectively.
#[derive(Debug)]
pub struct PooledBuf {
    pub pool: ArenaPool,
    /// Tracks the origin (Arena or Heap) and holds necessary data for management.
    origin: BufferOrigin,
}

impl Drop for PooledBuf {
    /// Returns the buffer to the pool or frees heap memory.
    /// Acquires pool lock if Arena.
    fn drop(&mut self) {
        match &mut self.origin {
            BufferOrigin::Arena { index, size , .. } => { // base_ptr not needed for drop
                self.pool.return_arena_chunk(*index, *size);
            }
            BufferOrigin::Heap { .. } => {} // Box drops automatically
        }
    }
}

// --- Deref / DerefMut ---

impl Deref for PooledBuf {
    type Target = [u8];
    /// Provides immutable access to the buffer's byte slice.
    fn deref(&self) -> &Self::Target {
        match &self.origin {
            BufferOrigin::Arena { index, size, base_ptr } => {
                // Get pointer directly from the stored base_ptr
                let ptr = unsafe { base_ptr.0.as_ptr().add(*index) }; // Use NonNull::as_ptr()
                // SAFETY: Pointer validity relies on Arc keeping pool alive,
                // index/size correctness relies on allocator logic.
                // Concurrent reads are safe.
                unsafe { slice::from_raw_parts(ptr, *size) }
            }
            BufferOrigin::Heap { data, .. } => data.deref(),
        }
    }
}

impl DerefMut for PooledBuf {
    /// Provides mutable access to the buffer's byte slice.
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut self.origin {
            BufferOrigin::Arena { index, size, base_ptr } => {
                // Get pointer directly from the stored base_ptr
                let ptr = unsafe { base_ptr.0.as_ptr().add(*index) };
                // SAFETY: Pointer validity relies on Arc keeping pool alive.
                // Index/size correctness relies on allocator logic.
                // Exclusivity relies on &mut self *and* correct allocator logic
                // preventing aliased mutable access from other PooledBuf instances.
                unsafe { slice::from_raw_parts_mut(ptr, *size) }
            }
            BufferOrigin::Heap { data, .. } => data.deref_mut(),
        }
    }
}

// --- Public Methods for PooledBuf ---
impl PooledBuf {
    /// Returns a string indicating the origin of the buffer's memory ("Arena" or "Heap").
    pub fn origin_type(&self) -> &'static str {
        match &self.origin {
            BufferOrigin::Arena { .. } => "Arena",
            BufferOrigin::Heap { .. } => "Heap",
        }
    }

    /// Returns the total capacity of the underlying allocated buffer
    /// (either [`PAGE_SIZE`] or [`LARGE_PAGE_SIZE`]).
    pub fn capacity(&self) -> usize {
        match &self.origin {
            BufferOrigin::Arena { size, .. } => *size,
            BufferOrigin::Heap { data, .. } => data.len(),
        }
    }
}