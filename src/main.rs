use std::{io::Write as _, os::unix::fs::PermissionsExt as _};

fn save_data1(path: &str, data: &[u8]) -> Result<(), std::io::Error> {
  let file = std::fs::OpenOptions::new()
    .write(true)
    .create(true)
    .truncate(true)
    .open(path)?;

  let metadata = std::fs::metadata(path)?;
  let mut permissions = metadata.permissions();
  permissions.set_mode(0o664);
  std::fs::set_permissions(path, permissions)?;

  let mut file = file;
  file.write_all(data)?;
  file.sync_all()?;
  Ok(())
}

use std::{process::Command, fs};

fn main() -> Result<(), std::io::Error> {
    let output = Command::new("mktemp")
        .output()?;

    let temp_file_path = String::from_utf8_lossy(&output.stdout).trim().to_string();

    if let Err(e) = save_data1(&temp_file_path, "hello world".as_bytes()) {
        eprintln!("Error saving data: {}", e);
        std::process::exit(1);
    }
    println!("Saved 'hello world' to {}", temp_file_path);

    // Read the contents of the file
    let contents = fs::read_to_string(&temp_file_path)?;
    println!("File contents: {}", contents);

    // Delete the file
    fs::remove_file(&temp_file_path)?;
    println!("Deleted file: {}", temp_file_path);

    Ok(())
}
