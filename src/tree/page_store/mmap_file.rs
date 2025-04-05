use std::{
    cmp::max,
    fs::{File, OpenOptions},
    io::Write,
    ops::Deref,
    path::Path,
    rc::Rc,
    sync::{Arc, Mutex},
};

use arc_swap::{ArcSwap, Guard};
use mmap_rs::{Mmap, MmapMut, MmapOptions};

use crate::tree::{
    buffer_store::Heap,
    consts,
    page_store::{PageStore, PageStoreError},
};

const META_OFFSET: u64 = 2 * consts::META_SIZE as u64;

type Result<T> = std::result::Result<T, PageStoreError>;

// This struct holds the state visible to readers.
// It's wrapped in an Arc and managed by ArcSwap.
pub struct ReaderState {
    // Readers can safely read any data from offset 0 up to this offset.
    mmap: Arc<Mmap>,
}

// A guard structure for reading, ensuring stable access to a snapshot.
// It holds the Guard from arc_swap.
pub struct ReaderGuard {
    guard: Guard<Arc<ReaderState>>,
}

impl Deref for ReaderGuard {
    type Target = ReaderState;
    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl ReaderState {
    pub fn get_page(&self, page_num: usize) -> &[u8] {
        &self.mmap[page_num * consts::PAGE_SIZE..(page_num + 1) * consts::PAGE_SIZE]
    }
}

// This struct holds the state exclusive to writers.
struct WriterState {
    // Shared file handle. Needed by the writer to resize and create new mmaps.
    file: File,
    mmap_mut: MmapMut,
    pending: Vec<Rc<[u8]>>,
}

struct MmapFileState {
    reader: ArcSwap<ReaderState>,
    writer: Mutex<WriterState>,
}

#[derive(Clone)]
pub struct MmapFile {
    state: Arc<MmapFileState>,
}

impl MmapFile {
    pub fn open_or_create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true) // create file only if not exists
            .open(path)?;
        let file_len = file.metadata()?.len() as usize;
        if file_len < META_OFFSET as usize {
            return Err(PageStoreError::InvalidFile(
                "file must be at least 2 meta page sizes".into(),
            ));
        }
        if (file_len - META_OFFSET as usize) % consts::PAGE_SIZE != 0 {
            return Err(PageStoreError::InvalidFile(
                "file size must be 2 meta pages, plus a multiple of page size".into(),
            ));
        }
        if file_len == 0 {
            return Self::create(file);
        }
        todo!()
    }

    fn create(mut file: File) -> Result<Self> {
        // TODO: have a concept of meta node.
        file.write_all(&[0u8; META_OFFSET as usize])?;
        let mmap = unsafe { MmapOptions::new(0)?.with_file(&file, META_OFFSET).map()? };
        let state = ReaderState { mmap: mmap.into() };
        let mmap_mut = unsafe {
            MmapOptions::new(0)?
                .with_file(&file, META_OFFSET)
                .map_mut()?
        };
        Ok(MmapFile {
            state: Arc::new(MmapFileState {
                reader: ArcSwap::new(Arc::new(state)),
                writer: Mutex::new(WriterState {
                    file,
                    mmap_mut,
                    pending: Vec::default(),
                }),
            }),
        })
    }

    pub fn append_page(&self) -> Result<()> {
        let reader = self.state.reader.load();
        let mut writer = self.state.writer.lock().unwrap();
        if writer.pending.len() * consts::PAGE_SIZE > writer.mmap_mut.len() {
            // Grow file.
            let reader_len = reader.mmap.len();
            let writer_len = writer.mmap_mut.len();
            let expand = max(
                writer.pending.len() * consts::PAGE_SIZE - writer_len,
                writer_len,
            );
            let expand = max(expand, 1 << 26 /* 64MB */);
            writer
                .file
                .set_len(META_OFFSET + (reader_len + writer_len + expand) as u64)?;
            // Merge to preserve existing virtual address and OS page caching.
            let expanded = unsafe {
                MmapOptions::new(expand)?
                    .with_file(&writer.file, META_OFFSET + (reader_len + writer_len) as u64)
                    .map_mut()?
            };
            let result = writer.mmap_mut.merge(expanded);
            if let Err((e, mmap_mut)) = result {
                writer.mmap_mut = mmap_mut;
                return Err(e.into());
            }
        }
        let mut pending = std::mem::take(&mut writer.pending);
        let mut copied = 0usize;
        for page in pending.iter() {
            writer.mmap_mut[copied..copied + consts::PAGE_SIZE].copy_from_slice(page.as_ref());
            copied += consts::PAGE_SIZE;
        }
        // Only make changes visible if the flush succeeded.
        if let Err(e) = writer.mmap_mut.flush(0..copied) {
            writer.pending = pending;
            return Err(e.into());
        }
        pending.clear();
        writer.pending = pending;
        match writer.mmap_mut.split_to(copied)?.make_read_only() {
            Err((mut mmap_mut, e)) => {
                std::mem::swap(&mut writer.mmap_mut, &mut mmap_mut);
                writer.mmap_mut.merge(mmap_mut).unwrap(); // If this fails, panic b/c this is unhandleable.
                Err(e.into())
            }
            Ok(split_off_mmap) => {
                let mut new_mmap = unsafe {
                    MmapOptions::new(reader.mmap.len())?
                        .with_address(reader.mmap.start())
                        .with_file(&writer.file, META_OFFSET)
                        .map()?
                };
                if let Err((e, _)) = new_mmap.merge(split_off_mmap) {
                    return Err(e.into());
                }
                self.state.reader.store(Arc::new(ReaderState {
                    mmap: Arc::new(new_mmap),
                }));
                Ok(())
            }
        }
    }
}

impl PageStore for MmapFile {
    type B = Heap;

    fn read_page(&self, page_num: usize) -> Result<Rc<[u8]>> {
        let reader = self.state.reader.load();
        if page_num * consts::PAGE_SIZE > reader.mmap.len() {
            return Err(PageStoreError::Read(
                format!(
                    "page_num = {} must be < {}",
                    page_num,
                    reader.mmap.len() / consts::PAGE_SIZE
                )
                .into(),
            ));
        }
        Ok(reader.mmap[page_num * consts::PAGE_SIZE..(page_num + 1) * consts::PAGE_SIZE].into())
    }

    fn write_page(&self, page: Rc<[u8]>) -> Result<usize> {
        let mut writer = self.state.writer.lock().unwrap();
        writer.pending.push(page);
        Ok(writer.pending.len() - 1)
    }

    fn buffer_store(&self) -> &Self::B {
        &Heap
    }
}
