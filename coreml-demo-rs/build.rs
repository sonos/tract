fn main() {
    // Bake @rpath into the dylib's install_name so the .app can find it via
    // its embedded `LD_RUNPATH_SEARCH_PATHS` rather than an absolute build path.
    if cfg!(target_os = "macos") {
        println!(
            "cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libtract_coreml_demo_rs.dylib"
        );
    }
}
