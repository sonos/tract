use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=TRACT_DYLIB_SEARCH_PATH");
    println!("cargo:rerun-if-changed=tract.h");
    if let Ok(path) = std::env::var("TRACT_DYLIB_SEARCH_PATH") {
        println!("cargo:rustc-link-search={path}");
    }
    println!("cargo:rustc-link-lib=tract");

    let bindings = bindgen::Builder::default()
        .header("tract.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs")).expect("Couldn't write bindings!");
}
