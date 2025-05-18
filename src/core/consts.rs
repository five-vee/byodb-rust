//! Constants related to memory page size and offsets.

const AARCH64_MACOS_NON_TEST: bool =
    cfg!(all(target_arch = "aarch64", target_os = "macos", not(test)));

/// Size of a page.
pub(crate) const PAGE_SIZE: usize = if AARCH64_MACOS_NON_TEST {
    16384 // 16KB
} else {
    4096 // 4KB
};

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
