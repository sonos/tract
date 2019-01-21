extern crate cc;

use std::env::var;

fn main() {
    let arch = var("CARGO_CFG_TARGET_ARCH").unwrap();
    if arch == "arm" || arch == "armv7" {
        cc::Build::new()
            .file("arm-vfp2/arm-vfpv2_mm_s4x4.c")
            .flag("-marm")
            .flag("-mfpu=vfp")
            .static_flag(true)
            .compile("armvfpv2");
        cc::Build::new()
            .file("armv7neon/armv7neon_mm_s8x4.c")
            .flag("-marm")
            .flag("-mfpu=neon")
            .flag("-mfloat-abi=hard")
            .static_flag(true)
            .compile("armv7neon");
    }
}
