use rayon::prelude::*;
use tract_data::prelude::Blob;

use super::runner;

static mut HAS_AVX512: bool = false;

#[cfg(target_arch = "x86_64")]
#[inline]
fn load_a_slice(slice: &[u8]) {
    unsafe {
        let mut ptr = slice.as_ptr();
        let end = ptr.add(slice.len());
        if HAS_AVX512 {
            while ptr < end {
                std::arch::asm!("
                vmovaps zmm0, [rsi]
                vmovaps zmm1, [rsi + 64]
                    ", inout("rsi") ptr,
                out("zmm0") _,
                out("zmm1") _,
                );
                ptr = ptr.add(128);
            }
        } else {
            while ptr < end {
                std::arch::asm!("
                vmovaps ymm0, [rsi]
                vmovaps ymm1, [rsi + 32]
                vmovaps ymm2, [rsi + 64]
                vmovaps ymm3, [rsi + 96]
                    ", inout("rsi") ptr,
                out("ymm0") _,
                out("ymm1") _,
                out("ymm2") _,
                out("ymm3") _,
                );
                ptr = ptr.add(128);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn load_a_slice(slice: &mut [u8]) {
    unsafe {
        let mut ptr = slice.as_ptr();
        let end = ptr.add(slice.len());
        while ptr < end {
            std::arch::asm!("
                ld1 {{v0.16b-v3.16b}}, [x0], #64
                ld1 {{v4.16b-v7.16b}}, [x0], #64
                    ", inout("x0") ptr,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v7") _,
            );
        }
    }
}

fn bandwidth_seq(slice_len: usize, threads: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        HAS_AVX512 = std::is_x86_feature_detected!("avx512f");
    }
    let buffer = unsafe { Blob::new_for_size_and_align(slice_len, 256) };
    let b = (0..threads)
        .into_par_iter()
        .map(|_| runner::run_bench(|| load_a_slice(&buffer)))
        .sum::<f64>();
    (slice_len * threads) as f64 / b
}

pub fn what_is_big() -> usize {
    1024 * 1024 * if cfg!(target_arch = "armv7") { 64 } else { 256 }
}

pub fn l1_bandwidth_seq(threads: usize) -> f64 {
    [1024, 2048, 4096, 8192, 16384]
        .into_iter()
        .map(|p| bandwidth_seq(p, threads))
        .max_by_key(|x| *x as i64)
        .unwrap()
}

pub fn main_memory_bandwith_seq(threads: usize) -> f64 {
    bandwidth_seq(what_is_big(), threads)
}

#[ignore]
#[test]
fn b() {
    let max = what_is_big();
    for threads in [1, 2, 3, 4] {
        println!("Threads: {}", threads);
        for size in (0..)
            .flat_map(|po2| (0..2).map(move |f| (1024 + 512 * f) * (1 << po2)))
            .take_while(|&s| s < max)
        {
            let bw = bandwidth_seq(size, threads);
            println!(
                "threads: {threads} slice: {} KiB bandwidth: {} GiB/s",
                size as f64 / 1024.,
                (bw / (1024. * 1024. * 1024.)) as usize
            );
        }
    }
}
