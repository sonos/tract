fn main() {
    println!("cargo:rerun-if-env-changed=TRACT_DYLIB_SEARCH_PATH");
    if let Ok(path) = std::env::var("TRACT_DYLIB_SEARCH_PATH") {
        println!("cargo:rustc-link-search={path}");
    }
    println!("cargo:rustc-link-lib=tract");
}
