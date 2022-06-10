#![allow(dead_code, non_upper_case_globals, unused_macros, non_snake_case, unused_assignments)]

use std::arch::asm;

mod nano;

#[repr(C, align(64))]
struct Floats([f32; 4096]);
const _F32: Floats = Floats([12.; 4096]);
const F32: *const f32 = (&_F32) as *const Floats as *const f32;

lazy_static::lazy_static! {
    static ref TICK: f64 = unsafe { b8192!(asm!("or rax, rax", out("rax") _)) };
}

macro_rules! kloop {
    ($filter: expr, $geo: literal, $n: expr, $path: literal, $ww: expr) => {
        let label = $path.split("/").last().unwrap().split_once(".").unwrap().0;
        let full_label = format!("{:8} {:40}", $geo, label);
        if full_label.contains($filter.unwrap_or("")) {
            let time = b2!({
                let mut p = F32;
                let mut q = F32;
                r128!(asm!(include_str!(concat!("../x86_64/fma/", $path)),
						 inout("rax") p, inout("rcx") q,
				out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
                out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
                out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
                out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
                out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _,
                out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
                ));
            }) / 128.;


            println!("{} {:3.0}% ({:>5.2 }/{:3 } cy) {:.2} GFLOP/s", full_label, ($n as f64 / $ww as f64) / time * 100. * *TICK, time / *TICK, $n as f64 / $ww as f64, $n as f64 / time / 1e9 );
        }
    }
}

unsafe fn packed_packed_1x8(f: Option<&str>) {
    println!("-- 1x8 kernels");
    kloop!(f, "1x8x1", (8 * 8), "8x8/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "1x8x2", (8 * 8 * 2), "8x8/packed_packed_loop1/avx-unroll.tmpli", 8);
    println!("");
}

unsafe fn packed_packed_2x6(f: Option<&str>) {
    println!("-- 2x6 kernels");
    kloop!(f, "2x6x1", (16 * 6), "2x6/packed_packed_loop1/original.tmpli", 8);
    kloop!(f, "2x6x2", (16 * 6 * 2), "2x6/packed_packed_loop1/original-unroll.tmpli", 8);
    // if std::is_x86_feature_detected!("avx512f") {
    //     kloop!(f, "2x6x1", (32 * 6), "2x6/packed_packed_loop1/avx-512.tmpli", 16);
    //     kloop!(f, "2x6x2", (32 * 6 * 2), "2x6/packed_packed_loop1/avx-512-unroll.tmpli", 16);
    // }
    println!("");
}

unsafe fn packed_packed_2x5(f: Option<&str>) {
    println!("-- 2x5 kernels");
    kloop!(f, "2x5x1", (16 * 5), "2x5/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "2x5x2", (16 * 5 * 2), "2x5/packed_packed_loop1/avx-unroll.tmpli", 8);
    // if std::is_x86_feature_detected!("avx512f") {
    //     kloop!(f, "2x5x1", (32 * 5), "2x5/packed_packed_loop1/avx-512.tmpli", 16);
    //     kloop!(f, "2x5x2", (32 * 5 * 2), "2x5/packed_packed_loop1/avx-512-unroll.tmpli", 16);
    // }
    println!("");
}

unsafe fn packed_packed_3x4(f: Option<&str>) {
    println!("-- 3x4 kernels");
    kloop!(f, "3x4x1", (24 * 4), "3x4/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "3x4x2", (24 * 4 * 2), "3x4/packed_packed_loop1/avx-unroll.tmpli", 8);
    // if std::is_x86_feature_detected!("avx512f") {
    //     kloop!(f, "3x4x1", (48 * 4), "3x4/packed_packed_loop1/avx-512.tmpli", 16);
    //     kloop!(f, "3x4x2", (48 * 4 * 2), "3x4/packed_packed_loop1/avx-512-unroll.tmpli", 16);
    // }
    println!("");
}

unsafe fn packed_packed_4x3(f: Option<&str>) {
    println!("-- 4x3 kernels");
    kloop!(f, "4x3x1", (32 * 3), "4x3/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "4x3x2", (32 * 3 * 2), "4x3/packed_packed_loop1/avx-unroll.tmpli", 8);
    // if std::is_x86_feature_detected!("avx512f") {
    //     kloop!(f, "4x3x1", (64 * 3), "4x3/packed_packed_loop1/avx-512.tmpli", 16);
    //     kloop!(f, "4x3x2", (64 * 3 * 2), "4x3/packed_packed_loop1/avx-512-unroll.tmpli", 16);
    // }
    println!("");
}

unsafe fn packed_packed_5x2(f: Option<&str>) {
    println!("-- 5x2 kernels");
    kloop!(f, "5x2x1", (40 * 2), "5x2/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "5x2x1", (40 * 2 * 2), "5x2/packed_packed_loop1/avx-unroll.tmpli", 8);
    // if std::is_x86_feature_detected!("avx512f") {
    //     kloop!(f, "5x2x1", (80 * 2), "5x2/packed_packed_loop1/avx-512.tmpli", 16);
    //     kloop!(f, "5x2x2", (80 * 2 * 2), "5x2/packed_packed_loop1/avx-512-unroll.tmpli", 16);
    // }
    println!("");
}

unsafe fn packed_packed_6x2(f: Option<&str>) {
    println!("-- 6x2 kernels");
    kloop!(f, "6x2x1", (40 * 2), "6x2/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "6x2x2", (40 * 2 * 2), "6x2/packed_packed_loop1/avx-unroll.tmpli", 8);
    // if std::is_x86_feature_detected!("avx512f") {
    //     kloop!(f, "6x2x1", (80 * 2), "6x2/packed_packed_loop1/avx-512.tmpli", 16);
    //     kloop!(f, "6x2x2", (80 * 2 * 2), "6x2/packed_packed_loop1/avx-512-unroll.tmpli", 16);
    // }
    println!("");
}

unsafe fn packed_packed_8x1(f: Option<&str>) {
    println!("-- 8x1 kernels");
    kloop!(f, "8x1x1", (64 * 1), "8x1/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "8x1x2", (64 * 1 * 2), "8x1/packed_packed_loop1/avx-unroll.tmpli", 8);
    // if std::is_x86_feature_detected!("avx512f") {
    //     kloop!(f, "8x1x1", (128 * 1), "8x1/packed_packed_loop1/avx-512.tmpli", 16);
    //     kloop!(f, "8x1x2", (128 * 1 * 2), "8x1/packed_packed_loop1/avx-512-unroll.tmpli", 16);
    // }
    println!("");
}

fn main() {
    let filter = std::env::args().skip(1).filter(|a| a != "--bench").next();
    unsafe {
        packed_packed_1x8(filter.as_deref());
        packed_packed_2x6(filter.as_deref());
        packed_packed_2x5(filter.as_deref());
        packed_packed_3x4(filter.as_deref());
        packed_packed_4x3(filter.as_deref());
        packed_packed_5x2(filter.as_deref());
        packed_packed_6x2(filter.as_deref());
        packed_packed_8x1(filter.as_deref());
    }
}
