use std::{
    cell::RefCell,
    cmp::max,
    fs::{File, OpenOptions},
    io::{Read, Write},
    ops::{Deref, DerefMut},
    path::Path,
    rc::Rc,
    sync::Arc,
};

use arc_swap::ArcSwap;
use mmap_rs::{MmapMut, MmapOptions};

use crate::tree::{
    consts,
    page_store::{PageStore, PageStoreError, ReadOnlyPage},
};

const META_OFFSET: u64 = 2 * consts::META_SIZE as u64;

type Result<T> = std::result::Result<T, PageStoreError>;

// This struct holds the state visible to readers.
// It's wrapped in an Arc and managed by ArcSwap.
pub struct ReaderState {
    mmap: Rc<MmapMut>,
    end: usize,
}

// This struct holds the state exclusive to writers.
struct WriterState {
    // Shared file handle. Needed by the writer to resize and create new mmaps.
    file: File,
    mmap: Rc<MmapMut>,
    start: usize,
    new_pointer: usize,
    write_pointer: usize,
}

impl WriterState {
    fn flush(&mut self) -> Result<()> {
        self.mmap.flush(self.start..self.write_pointer)?;
        self.start = self.write_pointer;
        Ok(())
    }

    fn grow_if_needed(&mut self, num_pages: usize) -> Result<bool> {
        if self.new_pointer + num_pages * consts::PAGE_SIZE <= self.mmap.len() {
            return Ok(false);
        }
        let expand = max(
            self.new_pointer + num_pages * consts::PAGE_SIZE - self.mmap.len(),
            self.mmap.len(),
        );
        let expand = max(expand, 1 << 26 /* 64MB */);
        let new_len = META_OFFSET as usize + self.mmap.len() + expand;
        self.file.set_len(new_len as u64)?;
        let mmap = unsafe {
            MmapOptions::new(new_len)?
                .with_file(&self.file, META_OFFSET)
                .map_mut()?
        };
        self.mmap = Rc::new(mmap);
        Ok(true)
    }
}

struct MmapFileState {
    reader: ArcSwap<ReaderState>,
    writer: RefCell<WriterState>,
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
        Self::open(file)
    }

    fn create(mut file: File) -> Result<Self> {
        // TODO: have a concept of meta node.
        file.write_all(&[0u8; META_OFFSET as usize])?;
        let mmap = unsafe {
            MmapOptions::new(0)?
                .with_file(&file, META_OFFSET)
                .map_mut()?
        };
        let mmap = Rc::new(mmap);
        Ok(MmapFile {
            state: Arc::new(MmapFileState {
                reader: ArcSwap::new(Arc::new(ReaderState {
                    mmap: mmap.clone(),
                    end: 0,
                })),
                writer: RefCell::new(WriterState {
                    file,
                    mmap: mmap.clone(),
                    start: 0,
                    new_pointer: 0,
                    write_pointer: 0,
                }),
            }),
        })
    }

    fn open(mut file: File) -> Result<Self> {
        // TODO: have a concept of meta node.
        let mut meta_page_a = [0u8; consts::META_SIZE];
        let read_a = file.read(&mut meta_page_a)?;
        if read_a != consts::META_SIZE {
            return Err(PageStoreError::InvalidFile(
                format!("file read only {read_a} bytes, must be at least META_OFFSET").into(),
            ));
        }
        let mut meta_page_b = [0u8; consts::META_SIZE];
        let read_b = file.read(&mut meta_page_b)?;
        if read_b != consts::META_SIZE {
            return Err(PageStoreError::InvalidFile(
                format!(
                    "file read only {} bytes, must be at least META_OFFSET",
                    read_a + read_b
                )
                .into(),
            ));
        }
        // TODO: Figure out if a or b is the most up to date meta page.
        let mmap = unsafe {
            MmapOptions::new(todo!())?
                .with_file(&file, META_OFFSET)
                .map_mut()?
        };
        let mmap = Rc::new(mmap);
        Ok(MmapFile {
            state: Arc::new(MmapFileState {
                writer: RefCell::new(WriterState {
                    file,
                    mmap: mmap.clone(),
                    new_pointer: todo!(),
                    start: todo!(),
                    write_pointer: todo!(),
                }),
                reader: ArcSwap::new(Arc::new(ReaderState {
                    mmap: mmap.clone(),
                    end: todo!(),
                })),
            }),
        })
    }
}

impl PageStore for MmapFile {
    type Page = MmapPage;
    type ReadOnlyPage = ReadOnlyMmapPage;

    fn read_page(&self, page_num: usize) -> Result<ReadOnlyMmapPage> {
        let reader = self.state.reader.load();
        if page_num * consts::PAGE_SIZE >= reader.end {
            return Err(PageStoreError::Read(
                format!(
                    "page_num = {} must be < {}",
                    page_num,
                    reader.end / consts::PAGE_SIZE
                )
                .into(),
            ));
        }
        Ok(ReadOnlyMmapPage {
            mmap: reader.mmap.clone(),
            offset: page_num * consts::PAGE_SIZE,
        })
    }

    fn new_page(&self) -> Result<Self::Page> {
        let mut writer = self.state.writer.borrow_mut();
        writer.grow_if_needed(1)?;
        let page = MmapPage {
            mmap: writer.mmap.clone(),
            offset: writer.new_pointer,
        };
        writer.new_pointer += consts::PAGE_SIZE;
        Ok(page)
    }

    fn write_page(&self, page: Self::Page) -> Self::ReadOnlyPage {
        let mut writer = self.state.writer.borrow_mut();
        writer.write_pointer += consts::PAGE_SIZE;
        let page = ReadOnlyMmapPage {
            mmap: page.mmap,
            offset: page.offset,
        };
        page
    }

    fn flush(&self) -> Result<()> {
        let mut writer = self.state.writer.borrow_mut();

        if writer.write_pointer < writer.new_pointer {
            let delta = (writer.new_pointer - writer.write_pointer) / consts::PAGE_SIZE;
            panic!("write_page() was called {delta} fewer times than new_page()");
        }

        writer.flush()?;
        self.state.reader.store(Arc::new(ReaderState {
            mmap: writer.mmap.clone(),
            end: writer.start,
        }));

        Ok(())
    }
}

pub struct MmapPage {
    mmap: Rc<MmapMut>,
    offset: usize,
}

impl Deref for MmapPage {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.mmap[self.offset..self.offset + consts::PAGE_SIZE]
    }
}

impl DerefMut for MmapPage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: The buffer is mutable b/c it's guaranteed that no two
        // MmapPage's will overlap.
        // SAFETY: MmapFile guarantees that offset..offset+PAGE_SIZE
        // is within the bounds of the mmap.
        unsafe {
            let ptr = self.mmap.as_ptr().add(self.offset) as *mut u8;
            &mut *std::ptr::slice_from_raw_parts_mut(ptr, consts::PAGE_SIZE)
        }
    }
}

pub struct ReadOnlyMmapPage {
    mmap: Rc<MmapMut>,
    offset: usize,
}

impl Deref for ReadOnlyMmapPage {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.mmap[self.offset..self.offset + consts::PAGE_SIZE]
    }
}

impl ReadOnlyPage for ReadOnlyMmapPage {
    fn page_num(&self) -> usize {
        self.offset / consts::PAGE_SIZE
    }
}
