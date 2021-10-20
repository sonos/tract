fn main() {
    let mut cc = cc::Build::new();
    cc
        .file("c/tile_1x1.c")
        .file("c/tile_2x2.c")
        .file("c/tile_4x4.c")
        .file("c/packed_tile_4x4.c")
        .file("c/tile_8x8.c")
        .file("c/packed_tile_8x8.c");
    if std::env::var("TARGET").unwrap().starts_with("aarch64") {
        cc.flag("-mtune=cortex-a53");
    } else {
        cc.flag("-mtune=haswell");
    }
    cc.flag("-funsafe-math-optimizations").compile("libmatmulbench");
}
