extern crate bindgen;

use std::env;
use std::path::PathBuf;

// Install NVIDIA CUDA prior to building the bindings with `cargo build`.
// https://docs.rs/bindgen/latest/bindgen/struct.Builder.html
// TODO(sdake): I am hopeful we can autogenerate network objects as well via:
//              https://docs.rs/bindgen/latest/bindgen/callbacks/trait.ParseCallbacks.html
// TODO(sdake): Pretty sure we need somethign like this: .allowlist_type("^cuuint(32|64)_t)"
// TODO(sdake): We may want some variant of: .rustified_enum<T: AsRef<str>>(self, arg: T) -> Builder

fn main() {
    let bindings = bindgen::Builder::default()
        .header("/usr/local/cuda-12.0/include/cuda.h")
        .allowlist_type("^CU.*")
        .allowlist_type("^cudaError_enum")
        .allowlist_type("^cu.*Complex$")
        .allowlist_type("^cuda.*")
        .allowlist_type("^libraryPropertyTYpe.*")
        .allowlist_var("^CU.*")
        .allowlist_function("^cu.*")
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .array_pointers_in_arguments(true)
        .generate()
        .unwrap();

    let target_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(target_path.join("cuda_driver_bindings.rs"))
        .expect("Couldn't write bindings!");
    println!(
        "Wrote bindings to {}\n",
        target_path.join("cuda_driver_bindings.rs").display()
    )
}
