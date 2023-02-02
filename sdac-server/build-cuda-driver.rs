extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};

use bindgen::callbacks::*;

macro_rules! p {
    ($($tokens: tt)*) => {
        println!("cargo:warning={}", format!($($tokens)*))
    }
}

// Install NVIDIA CUDA prior to building the bindings with `cargo build`.
// https://docs.rs/bindgen/latest/bindgen/struct.Builder.html
// TODO(sdake): I am hopeful we can autogenerate network objects as well via:
//              https://docs.rs/bindgen/latest/bindgen/callbacks/trait.ParseCallbacks.html
// TODO(sdake): We may want some variant of: .rustified_enum<T: AsRef<str>>(self, arg: T) -> Builder

#[derive(Debug)]
struct NetworkEmitter {}

impl ParseCallbacks for NetworkEmitter {
    fn generated_name_override(&self, _item_info: ItemInfo<'_>) -> Option<String> {
        // p!("f {:#?}\n", item_info.name);
        None
    }

    fn item_name(&self, _name: &str) -> Option<String> {
        // p!("v {}\n", name);
        None
    }
}

fn main() {
    let cdir = std::env::var("CUDA_DIR").unwrap_or("/usr/local/cuda-11.8".to_string());
    let cuda_dir = Path::new(&cdir);
    let cuda_lib = cuda_dir.join("targets/x86_64-linux/lib/stubs");

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());

    let bindings = bindgen::Builder::default()
        .header(cuda_dir.join("include/cuda.h").display().to_string())
        .header(
            cuda_dir
                .join("include/cuda_runtime_api.h")
                .display()
                .to_string(),
        )
        .allowlist_function("cu.*")
        .allowlist_type("CU.*")
        .allowlist_type("cuda.*")
        .derive_eq(true)
        .array_pointers_in_arguments(true)
        .parse_callbacks(Box::new(NetworkEmitter {}))
        .generate()
        .unwrap();

    let target_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(target_path.join("cuda_driver_bindings.rs"))
        .expect("Couldn't write bindings!");

    p!(
        "Wrote bindings to {}",
        target_path.join("cuda_driver_bindings.rs").display()
    );
}
