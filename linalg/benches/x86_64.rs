#![allow(dead_code, non_upper_case_globals, unused_macros, non_snake_case, unused_assignments)]

use std::arch::asm;

mod nano;

#[repr(C, align(64))]
struct Floats([f32; 256 * 1024 * 64]);
const _F32: Floats = Floats([12.; 256 * 1024 * 64]);
const F32: *const f32 = (&_F32) as *const Floats as *const f32;

lazy_static::lazy_static! {
    static ref TICK: f64 = unsafe { b8192!(asm!("or rax, rax", out("rax") _)) };
}

macro_rules! kloop {
    ($filter: expr, $geo: literal, $n: expr, $path: literal, $ww: expr, $u: expr, $arch: expr) => {
        let label = $path.split("/").last().unwrap().split_once(".").unwrap().0;
        let full_label = format!("{:8} {:40}", $geo, label);
		let repeats = 32;
		let ks = 256;
        if full_label.contains($filter.unwrap_or("")) {
            let time = b1!({

				let mut p = F32;
				let mut q = F32;
				let mut k = ks;
				let mut r = repeats;
				asm!(
					concat!(r#"
2:
      mov rax, r9
      mov rcx, r10
      mov r8, r12
3:
    "#, include_str!(			concat!("../x86_64/", $arch, "/", $path)), "\n sub r8, ", $u, r#"
jnz 3b

sub r11, 1
jnz 2b
"#),
					inout("r9") p, inout("r10") q, inout("r12") k, inout("r11") r, out("rax") _, out("rcx") _,
					out("r8") _,
					out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
					out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
					out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
					out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
					out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _,
					out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
                    out("zmm28") _,  out("zmm29") _,  out("zmm30") _,  out("zmm31") _,
				);
            });

			// We have k=1024 * 64 but some tests step twice per iteration
			let iterations = (ks * repeats / $u);
			// Those that step twice process twice as many elements per iteration
			let elems_per_iteration = $n * $u;

			let time_per_iteration = time / iterations  as f64;

			let total_floats = elems_per_iteration * iterations;
			let flops = total_floats as f64 / time;

			let total_time_ms = time * 1e6;
			let fmas_per_iteration = ($n as f64 / $ww as f64) * $u as f64;
			let ticks_per_iteration = time_per_iteration / *TICK;
            println!("{} {:3.5} {:3.0}% ({:>5.2 }/{:3 } cy) {:.2} GFLOP/s", full_label, total_time_ms, fmas_per_iteration / ticks_per_iteration * 100., ticks_per_iteration, fmas_per_iteration, flops / 1e9 );
        }
    };

	($filter: expr, $geo: literal, $n: expr, $path: literal, $ww: expr) => {
		kloop!($filter, $geo, $n, $path, $ww, 1, "fma")
	};
	($filter: expr, $geo: literal, $n: expr, $path: literal, $ww: expr, $u: expr) => {
		kloop!($filter, $geo, $n, $path, $ww, $u, "fma")
	};
}

unsafe fn packed_packed_1x12(f: Option<&str>) {
    println!("-- 1x12 kernels");
    if std::is_x86_feature_detected!("avx512f") {
        kloop!(f, "1x12x1", (16 * 1 * 12), "1x12/packed_packed_loop1/avx-512.tmpli", 16, 1, "avx512");
    }
    println!();
}

unsafe fn packed_packed_1x8(f: Option<&str>) {
    println!("-- 1x8 kernels");
    kloop!(f, "1x8x1", (8 * 8), "8x8/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "1x8x2", (8 * 8), "8x8/packed_packed_loop1/avx-unroll.tmpli", 8, 2);
    if std::is_x86_feature_detected!("avx512f") {
        kloop!(f, "1x8x1", (16 * 1 * 8), "8x8/packed_packed_loop1/avx-512.tmpli", 16, 1, "avx512");
    }
    println!();
}

unsafe fn packed_packed_2x6(f: Option<&str>) {
    println!("-- 2x6 kernels");
    kloop!(f, "2x6x1", (16 * 6), "2x6/packed_packed_loop1/original.tmpli", 8);
    kloop!(f, "2x6x2", (16 * 6), "2x6/packed_packed_loop1/original-unroll.tmpli", 8, 2);
    if std::is_x86_feature_detected!("avx512f") {
        kloop!(f, "2x6x1", (16 * 2 * 6), "2x6/packed_packed_loop1/avx-512.tmpli", 16, 1, "avx512");
        kloop!(f, "2x6x2", (16 * 2 * 6), "2x6/packed_packed_loop1/avx-512-unroll.tmpli", 16, 2, "avx512");
    }
    println!();
}

unsafe fn packed_packed_2x5(f: Option<&str>) {
    println!("-- 2x5 kernels");
    kloop!(f, "2x5x1", (16 * 5), "2x5/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "2x5x2", (16 * 5), "2x5/packed_packed_loop1/avx-unroll.tmpli", 8, 2);
    if std::is_x86_feature_detected!("avx512f") {
        kloop!(f, "2x5x1", (32 * 5), "2x5/packed_packed_loop1/avx-512.tmpli", 16, 1, "avx512");
        kloop!(f, "2x5x2", (32 * 5), "2x5/packed_packed_loop1/avx-512-unroll.tmpli", 16, 2, "avx512");
    }
    println!();
}

unsafe fn packed_packed_3x4(f: Option<&str>) {
    println!("-- 3x4 kernels");
    kloop!(f, "3x4x1", (24 * 4), "3x4/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "3x4x2", (24 * 4), "3x4/packed_packed_loop1/avx-unroll.tmpli", 8, 2);
    if std::is_x86_feature_detected!("avx512f") {
        kloop!(f, "3x4x1", (16 * 3 * 4), "3x4/packed_packed_loop1/avx-512.tmpli", 16, 1, "avx512");
        kloop!(f, "3x4x2", (16 * 3 * 4), "3x4/packed_packed_loop1/avx-512-unroll.tmpli", 16, 2, "avx512");
    }
    println!();
}

unsafe fn packed_packed_4x3(f: Option<&str>) {
    println!("-- 4x3 kernels");
    kloop!(f, "4x3x1", (32 * 3), "4x3/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "4x3x2", (32 * 3), "4x3/packed_packed_loop1/avx-unroll.tmpli", 8, 2);
    if std::is_x86_feature_detected!("avx512f") {
        kloop!(f, "4x3x1", (16 * 4 * 3), "4x3/packed_packed_loop1/avx-512.tmpli", 16, 1, "avx512");
        kloop!(f, "4x3x2", (16 * 4 * 3), "4x3/packed_packed_loop1/avx-512-unroll.tmpli", 16, 2, "avx512");
    }
    println!();
}

unsafe fn packed_packed_5x2(f: Option<&str>) {
    println!("-- 5x2 kernels");
    kloop!(f, "5x2x1", (40 * 2), "5x2/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "5x2x1", (40 * 2), "5x2/packed_packed_loop1/avx-unroll.tmpli", 8, 2);
    if std::is_x86_feature_detected!("avx512f") {
        kloop!(f, "5x2x1", (16 * 5 * 2), "5x2/packed_packed_loop1/avx-512.tmpli", 16, 1, "avx512");
        kloop!(f, "5x2x2", (16 * 5 * 2), "5x2/packed_packed_loop1/avx-512-unroll.tmpli", 16, 2, "avx512");
    }
    println!();
}

unsafe fn packed_packed_6x2(f: Option<&str>) {
    println!("-- 6x2 kernels");
    kloop!(f, "6x2x1", (48 * 2), "6x2/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "6x2x2", (48 * 2), "6x2/packed_packed_loop1/avx-unroll.tmpli", 8, 2);
    if std::is_x86_feature_detected!("avx512f") {
        kloop!(f, "6x2x1", (16 * 6 * 2), "6x2/packed_packed_loop1/avx-512.tmpli", 16, 1, "avx512");
        kloop!(f, "6x2x2", (16 * 6 * 2), "6x2/packed_packed_loop1/avx-512-unroll.tmpli", 16, 2, "avx512");
    }
    println!();
}

unsafe fn packed_packed_8x2(f: Option<&str>) {
    println!("-- 8x2 kernels");
    if std::is_x86_feature_detected!("avx512f") {
        kloop!(f, "8x2x1", (16 * 8 * 2), "8x2/packed_packed_loop1/avx-512.tmpli", 16, 1, "avx512");
    }
    println!();
}

unsafe fn packed_packed_8x1(f: Option<&str>) {
    println!("-- 8x1 kernels");
    kloop!(f, "8x1x1", (64 * 1), "8x1/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "8x1x2", (64 * 1), "8x1/packed_packed_loop1/avx-unroll.tmpli", 8, 2);
    if std::is_x86_feature_detected!("avx512f") {
        kloop!(f, "8x1x1", (16 * 8 * 1), "8x1/packed_packed_loop1/avx-512.tmpli", 16, 1, "avx512");
        kloop!(f, "8x1x2", (16 * 8 * 1), "8x1/packed_packed_loop1/avx-512-unroll.tmpli", 16, 2, "avx512");
    }
    println!();
}

unsafe fn packed_packed_6x1(f: Option<&str>) {
    println!("-- 6x1 kernels");
    kloop!(f, "6x1x1", (48 * 1), "6x1/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "6x1x2", (48 * 1), "6x1/packed_packed_loop1/avx-unroll.tmpli", 8, 2);
    if std::is_x86_feature_detected!("avx512f") {
        kloop!(f, "6x1x1", (16 * 6 * 1), "6x1/packed_packed_loop1/avx-512.tmpli", 16, 1, "avx512");
        kloop!(f, "6x1x2", (16 * 6 * 1), "6x1/packed_packed_loop1/avx-512-unroll.tmpli", 16, 2, "avx512");
    }
    println!();
}

unsafe fn packed_packed_7x1(f: Option<&str>) {
    println!("-- 7x1 kernels");
    if std::is_x86_feature_detected!("avx512f") {
        kloop!(f, "7x1x1", (16 * 7 * 1), "7x1/packed_packed_loop1/avx-512.tmpli", 16, 1, "avx512");
        kloop!(f, "7x1x2", (16 * 7 * 1), "7x1/packed_packed_loop1/avx-512-unroll.tmpli", 16, 2, "avx512");
    }
    println!();
}

unsafe fn packed_packed_1x1(f: Option<&str>) {
    if std::is_x86_feature_detected!("avx512f") {
        kloop!(f, "1x1x1", (16 * 1 * 1), "1x1/packed_packed_loop1/avx-512.tmpli", 16, 1, "avx512");
        kloop!(f, "1x1x2", (16 * 1 * 1), "1x1/packed_packed_loop1/unroll.tmpli", 16, 2, "avx512");
        kloop!(f, "1x1x4", (16 * 1 * 1), "1x1/packed_packed_loop1/unroll-4.tmpli", 16, 4, "avx512");
        kloop!(f, "1x1x8", (16 * 1 * 1), "1x1/packed_packed_loop1/unroll-8.tmpli", 16, 8, "avx512");
        kloop!(f, "1x1x16", (16 * 1 * 1), "1x1/packed_packed_loop1/unroll-16.tmpli", 16, 16, "avx512");
    }
    println!();
}

unsafe fn packed_packed_10x1(f: Option<&str>) {
    println!("-- 10x1 kernels");
    kloop!(f, "10x1x1", (80 * 1), "10x1/packed_packed_loop1/avx.tmpli", 8);
    kloop!(f, "10x1x2", (80 * 1), "10x1/packed_packed_loop1/avx-unroll.tmpli", 8, 2);
    if std::is_x86_feature_detected!("avx512f") {
        kloop!(f, "10x1x1", (16 * 10 * 1), "10x1/packed_packed_loop1/avx-512.tmpli", 16, 1, "avx512");
        kloop!(f, "10x1x2", (16 * 10 * 1), "10x1/packed_packed_loop1/avx-512-unroll.tmpli", 16, 2, "avx512");
    }
    println!();
}

fn main() {
    let filter = std::env::args().skip(1).find(|a| a != "--bench");
    unsafe {
        packed_packed_1x1(filter.as_deref());
        packed_packed_1x12(filter.as_deref());
        packed_packed_1x8(filter.as_deref());
        packed_packed_2x6(filter.as_deref());
        packed_packed_2x5(filter.as_deref());
        packed_packed_3x4(filter.as_deref());
        packed_packed_4x3(filter.as_deref());
        packed_packed_5x2(filter.as_deref());
        packed_packed_6x2(filter.as_deref());
        packed_packed_8x2(filter.as_deref());
        packed_packed_6x1(filter.as_deref());
        packed_packed_7x1(filter.as_deref());
        packed_packed_8x1(filter.as_deref());
        packed_packed_10x1(filter.as_deref());
    }
}
