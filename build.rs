use fs_extra::copy_items;
use fs_extra::dir::CopyOptions;
use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=res/*");

    let out_dir = env::var("OUT_DIR")?;
    let mut copy_options = CopyOptions::new();
    copy_options.overwrite = true;
    let paths_to_copy = vec!["res/"];
    copy_items(&paths_to_copy, out_dir, &copy_options)?;

    Ok(())
}