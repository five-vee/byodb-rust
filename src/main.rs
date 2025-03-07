use std::fs::File;
use std::rc::Rc;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SaveError {
    #[error("I/O error: {0}")]
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

fn main() {
    use std::{fs, process::Command};
    let output = match Command::new("mktemp").output() {
        Ok(it) => it,
        Err(err) => {
            eprintln!("Error running mktemp: {}", err);
            std::process::exit(1);
        }
    };

    let temp_file_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    println!("Created file: {}", temp_file_path);

    match save_data2(temp_file_path.as_str(), "hello_world".as_bytes()) {
        Ok(()) => {}
        Err(save_error) => {
            if let SaveError::TmpFileError(tmp, _) = &save_error {
                // Attempt to discard the temporary file if it still exists.
                std::fs::remove_file(tmp.as_ref()).unwrap_or_default();
            }
            eprintln!("Error saving data: {}", &save_error);
            std::process::exit(1);
        }
    }
    println!("Saved 'hello world' to {}", temp_file_path);

    let contents = match fs::read_to_string(&temp_file_path) {
        Ok(contents) => contents,
        Err(err) => {
            eprintln!("Error reading data: {}", err);
            std::process::exit(1);
        }
    };
    println!("File contents: {}", contents);

    if let Err(e) = fs::remove_file(&temp_file_path) {
        eprintln!("Error deleting file: {}", e);
        std::process::exit(1);
    }
    println!("Deleted file: {}", temp_file_path);
}
