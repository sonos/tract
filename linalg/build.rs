extern crate cc;

use std::env::var;

fn main() {
    if var("TARGET").unwrap() == "arm-unknown-linux-gnueabihf" {
            cc::Build::new()
        .file("arm-vfp2/arm-vfpv2_mm_s4x4.c")
        .flag("-marm")
        .flag("-mfpu=vfp")
        .flag("-mfloat-abi=hard")
        .static_flag(true)
        .compile("armvfpv2");
    }
}
