use std::fs;
use std::process::Command;

mod file_util;
mod bplus_tree;

fn main() {
    let output = match Command::new("mktemp").output() {
        Ok(it) => it,
        Err(err) => {
            eprintln!("Error running mktemp: {}", err);
            std::process::exit(1);
        }
    };

    let temp_file_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    println!("Created file: {}", temp_file_path);

    match file_util::save_data2(temp_file_path.as_str(), "hello_world".as_bytes()) {
        Ok(()) => {}
        Err(save_error) => {
            if let file_util::SaveError::TmpFileError(tmp, _) = &save_error {
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
