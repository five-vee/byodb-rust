//! Constants related to memory page size and offsets.

/// Size of a B+ tree node page.
// 4KB
#[cfg(test)]
pub(crate) const PAGE_SIZE: usize = 4096;

/// Size of a B+ tree node page.
// 16KB
#[cfg(not(test))]
pub(crate) const PAGE_SIZE: usize = 16384;

/// The maximum allowed key size in a tree.
pub const MAX_KEY_SIZE: usize = 1000;
/// The maximum allowed value size in a tree.
pub const MAX_VALUE_SIZE: usize = 1000;

const _: () = {
    assert!(PAGE_SIZE <= (1 << 16), "page size is within 16 bits");
    assert!(
        (PAGE_SIZE as isize)
            - 2 // type
            - 2 // nkeys
            // 3 keys + overhead
            - 3 * (8 + 2 + MAX_KEY_SIZE as isize)
            >= 0,
        "3 keys of max size cannot fit into an internal node page"
    );
    assert!(
        (PAGE_SIZE as isize)
            - 2 // type
            - 2 // nkeys
            // 2 key-value pairs + overhead
            - 2*(2 + 2 + 2 + MAX_KEY_SIZE as isize + MAX_VALUE_SIZE as isize)
            >= 0,
        "2 key-value pairs of max size cannot fit into a leaf node page"
    );
};
