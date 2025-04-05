/// Size of a B+ tree node page.
pub(crate) const PAGE_SIZE: usize = 4096;
/// The maximum allowed key size in a tree.
pub const MAX_KEY_SIZE: usize = 1000;
/// The maximum allowed value size in a tree.
pub const MAX_VALUE_SIZE: usize = 3000;

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
            // 1 key-value pair + overhead
            - (2 + 2 + 2 + MAX_KEY_SIZE as isize + MAX_VALUE_SIZE as isize)
            >= 0,
        "1 key-value pair of max size cannot fit into a leaf node page"
    );
};
