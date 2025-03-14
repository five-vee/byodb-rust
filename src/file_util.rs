use std::fs::File;
use std::rc::Rc;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SaveError {
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    #[error("I/O error on tmp file {0}: {1}")]
    TmpFileError(Rc<str>, std::io::Error),
}

/// Creates a temporary file.
fn create_tmp_file(path: &str) -> Result<(File, Rc<str>), std::io::Error> {
    use rand::Rng as _;
    use std::os::unix::fs::PermissionsExt as _;
    let tmp = format!("{}.tmp.{}", path, rand::rng().random::<u32>());
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)?;

    let metadata = std::fs::metadata(&tmp)?;
    let mut permissions = metadata.permissions();
    permissions.set_mode(0o664);
    std::fs::set_permissions(&tmp, permissions)?;
    println!("Created tmp file: {}", tmp);
    Ok((file, tmp.into()))
}

/// Writes to a file, syncs it, then renames it.
fn write_sync_rename(
    file: &mut File,
    tmp: &str,
    path: &str,
    data: &[u8],
) -> Result<(), std::io::Error> {
    use std::io::Write as _;
    file.write_all(data)?;
    file.sync_all()?;
    println!("Wrote to tmp file: {}", tmp);
    std::fs::rename(tmp, path)?;
    println!("Renamed tmp file to {}", path);
    Ok(())
}

/// Replaces data at `path` "atomically" via a rename.
pub fn save_data2(path: &str, data: &[u8]) -> Result<(), SaveError> {
    let (mut file, tmp) = create_tmp_file(path)?;
    if let Err(e) = write_sync_rename(&mut file, tmp.as_ref(), path, data) {
        return Err(SaveError::TmpFileError(tmp, e));
    }
    Ok(())
}
