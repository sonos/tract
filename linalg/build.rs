extern crate cc;

use std::env::var;

fn main() {
    let arch = var("CARGO_CFG_TARGET_ARCH").unwrap();
    if arch == "x86_64" {
        cc::Build::new()
            .file("x86_64/fma/stile16x6.S")
            .flag("-mfma")
            .static_flag(true)
            .compile("x86_64_fma");
    }
    if arch == "arm" || arch == "armv7" {
        cc::Build::new()
            .file("arm32/armvfpv2/armvfpv2_conv_s4x4.S")
            .file("arm32/armvfpv2/armvfpv2_mm_s4x4.S")
            .flag("-marm")
            .flag("-mfpu=vfp")
            .static_flag(true)
            .compile("armvfpv2");
        cc::Build::new()
            .file("arm32/armv7neon/armv7neon_conv_s8x4.S")
            .file("arm32/armv7neon/armv7neon_mm_s8x4.S")
            .file("arm32/armv7neon/armv7neon_tile_s8x4.S")
            .flag("-marm")
            .flag("-mfpu=neon")
            .static_flag(true)
            .compile("armv7neon");
    }
    if arch == "aarch64" {
        cc::Build::new()
            .file("arm64/arm64simd/arm64simd_conv_s8x8.S")
            .file("arm64/arm64simd/arm64simd_mm_s8x8.S")
            .static_flag(true)
            .compile("arm64");
    }
}
