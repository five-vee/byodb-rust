# Build Your Own Database from Scratch (in Rust)

https://build-your-own.org/database/, but instead of Go, use Rust. This a personal project to learn both database internals and the Rust programming language.

## Database Design

This database implements a **Copy-on-Write (COW) B+ Tree** stored within a **memory-mapped file**. This design provides several key features for robustness, concurrency, and efficient data management.

1.  **Memory-Mapped File**:
    *   The entire database resides in a single file on disk. This file is **memory-mapped**, allowing the database to treat portions of the file as if they were directly in memory. This approach minimizes explicit read/write system calls for data access, leveraging the operating system's virtual memory management.
    *   The file structure begins with a special **meta page**, followed by a series of fixed-size data pages used for B+ Tree nodes and free list management.

1.  **Copy-on-Write (COW) B+ Tree**:
    *   The primary data structure is a B+ Tree, optimized for efficient key-value storage, lookups, and range scans.
    *   A **Copy-on-Write (COW)** strategy is employed for all modifications (inserts, updates, deletes). Instead of altering existing B+ Tree nodes in place, the system creates modified copies of the affected nodes (and their parent nodes up to the root). The original nodes remain unchanged.
    *   This COW mechanism is crucial for enabling snapshot isolation and simplifying concurrent access, as readers can continue to see a stable version of the tree while modifications are in progress.

1.  **Durability via Meta Node**:
    *   The **meta page**, located at the beginning of the database file, is vital for ensuring data **durability** and atomicity of writes. It stores essential metadata, including a pointer to the current root page of the B+ Tree and information about the free list's state.
    *   Database updates are committed by first ensuring all modified data pages are written to disk, and then atomically updating the meta page to reflect the new state (e.g., the new root page). If a crash occurs, the database can recover to the last successfully committed state by reading the meta page.

1.  **Multiversion Concurrency Control (MVCC)**:
    *   The database supports concurrent read and write operations using an MVCC-like approach.
    *   **Readers** (operating within read-only transactions) are provided with a consistent **snapshot** of the database. They do not block writers, and writers do not block them.
    *   This is achieved by allowing readers to access older versions of the memory-mapped data if a writer has concurrently made changes. The `arc_swap` mechanism is used to manage different versions of the memory map, ensuring that active readers can continue using their snapshot even if the underlying file is extended or its active version changes due to a writer's commit. Readers only see data that has been fully "flushed" and committed.
    *   **Writers** (operating within read-write transactions) obtain exclusive access for modifications, ensuring that write operations are serialized.

1.  **Free List and Garbage Collection**:
    *   When data is updated or deleted, B+ Tree pages previously holding that data become unused.
    *   These unused pages are not immediately overwritten or removed from the file. Instead, they are marked for **garbage collection**.
    *   A **free list** is maintained within the database file. This is a linked list structure where each node (itself a page) points to other pages that are free and can be reused for new data.
    *   A reclamation mechanism (utilizing the `seize` crate) ensures that pages are only added to the free list once it's guaranteed that no active transaction is still referencing them. This allows for efficient reuse of disk space.

## API Usage

Interacting with the database is done through the `DB` struct and `Txn` (transaction) struct.

The `DB` struct represents the database itself. You use `DB::open_or_create(path)` to either open an existing database file or create a new one at the specified path. Once you have a `DB` instance, all interactions with the data are performed through transactions.

Transactions provide a consistent view of the database and ensure atomicity for operations. There are two types of transactions:
-   **Read-only transactions (`r_txn`)**: Created using `db.r_txn()`. These allow you to read data (e.g., using `get()`, `in_order_iter()`) without locking out other readers. Multiple read-only transactions can occur concurrently.
-   **Read-write transactions (`rw_txn`)**: Created using `db.rw_txn()`. These provide exclusive access for modifying the database. You can perform operations like `insert()`, `update()`, and `delete()`. Changes made in a read-write transaction are isolated until `commit()` is called. If `commit()` is not called (e.g., due to an error, or if the transaction is simply dropped), all changes are automatically aborted.

### Example

The following example demonstrates basic usage, including opening a database, performing read operations, and performing write operations within transactions.

```rust,no_run
# use byodb_rust::{DB, Result, Txn};
# fn main() -> Result<()> {
let path = "/path/to/a/db/file";
let db: DB = DB::open_or_create(path)?;

// Perform reads in a read transaction.
{
    let r_txn: Txn<_> = db.r_txn();
    for (k, v) in r_txn.in_order_iter() {
        println!("key: {k:?}, val: {v:?}");
    }
} // read transaction is dropped at the end of scope.

// Perform reads and writes in a read-write transaction.
{
    let mut rw_txn: Txn<_> = db.rw_txn();
    if rw_txn.get("some_key".as_bytes())?.is_some() {
        rw_txn.update("some_key".as_bytes(), "some_new_val".as_bytes())?;
    }
    // If rw_txn.commit() is not called b/c there was error in any of the
    // above steps, then when rw_txn is dropped, it is equivalent to doing
    // rw_txn.abort().
    rw_txn.commit();
}
# Ok(())
# }
```

## Caveats/Limitations

*   **Not production ready**: This project is primarily a learning exercise and lacks the robustness, extensive testing, and feature completeness required for production environments.
*   **No checksum in pages yet**: Data pages do not currently include checksums, making it harder to detect silent data corruption on disk.
*   **No disaster/corruption recovery**: Beyond basic meta page integrity for atomic commits, there are no advanced mechanisms for recovering from significant file corruption or disasters.
*   **No network replication/CDC**: The database operates as a single-node instance; there's no support for replicating data to other nodes or Change Data Capture (CDC) for external systems.
*   **No journaling mode (for performance)**: Lacks a write-ahead log (WAL) or similar journaling, which could offer different performance trade-offs and recovery strategies.
*   **No profiling/monitoring**: No built-in tools or hooks for performance profiling or operational monitoring.
*   **No robust testing or CI/CD**: While some tests exist, comprehensive testing (e.g., stress testing, fuzz testing) and a CI/CD pipeline are not implemented.
*   **No buffer caching mode**: Relies solely on the OS's mmap capabilities for page caching. An explicit buffer cache could offer more control over memory usage and caching strategies.

## Completed tasks

### [01. From Files To Databases](https://build-your-own.org/database/01_files)

* [x] 1.1 Updating files in-place
* [x] 1.2 Atomic renaming

### [04. B+Tree Node and Insertion](https://build-your-own.org/database/04_btree_code_1)

* [x] 4.1 B+tree node
* [x] 4.2 Decode the B+tree nodes
* [x] 4.3 Create B+tree nodes
* [X] 4.4 Insert or update the leaf node
* [X] 4.5 Split a node
* [X] 4.6 B+tree data structure

### [05. B+Tree Deletion and Testing](https://build-your-own.org/database/05_btree_code_2)

* [x] 5.1 High-level interfaces
* [x] 5.2 Merge nodes
* [x] 5.3 B+tree deletion
* [x] 5.4 Test the B+tree

### [06. Append-Only KV Store](https://build-your-own.org/database/06_btree_disk)

* [x] 6.2 Two-phase update
* [x] 6.3 Database on a file
* [x] 6.4 Manage disk pages
* [x] 6.5 The meta page
* [x] 6.6 Error handling

### [0.7 Free List: Recyle & Reuse](https://build-your-own.org/database/07_free_list)

* [x] 7.1 Memory management techniques
* [x] 7.2 Linked list on disk
* [x] 7.3 Free list implementation
* [x] 7.4 KV with a free list
