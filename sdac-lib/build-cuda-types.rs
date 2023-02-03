extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};

// Install NVIDIA CUDA prior to building the bindings with `cargo build`.
// https://docs.rs/bindgen/latest/bindgen/struct.Builder.html
fn main() {
    let cdir = std::env::var("CUDA_DIR").unwrap_or("/usr/local/cuda-11.8".to_string());
    let cuda_dir = Path::new(&cdir);

    let bindings = bindgen::Builder::default()
        .header(cuda_dir.join("include/cuda.h").display().to_string())
        .header(cuda_dir.join("include/cuda_runtime_api.h").display().to_string())
        .allowlist_type("CU.*")
        .allowlist_type("cuda.*")
        .derive_eq(true)
        .array_pointers_in_arguments(true)
        .generate()
        .unwrap();

    let target_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(target_path.join("cuda_types.rs"))
        .expect("Couldn't write bindings!");

    println!(
        "Wrote bindings to {}",
        target_path.join("cuda_types.rs").display()
    );
}
