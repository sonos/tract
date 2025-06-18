use tract_data::itertools::Itertools;
use tract_data::prelude::Blob;

use super::runner;

#[cfg(target_arch = "x86_64")]
static mut HAS_AVX512: bool = false;

#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn load_a_slice(slice: &[u8], loops: usize) {
    unsafe {
        if HAS_AVX512 {
            for _ in 0..loops {
                let mut ptr = slice.as_ptr();
                let end = ptr.add(slice.len());
                while ptr < end {
                    std::arch::asm!("
                vmovaps zmm0, [rsi]
                vmovaps zmm1, [rsi + 64]
                vmovaps zmm2, [rsi + 128]
                vmovaps zmm3, [rsi + 192]
                vmovaps zmm4, [rsi + 256]
                vmovaps zmm5, [rsi + 320]
                vmovaps zmm6, [rsi + 384]
                vmovaps zmm7, [rsi + 448]
                    ", inout("rsi") ptr,
                    out("zmm0") _,
                    out("zmm1") _,
                    );
                    ptr = ptr.add(512);
                }
            }
        } else {
            let mut ptr = slice.as_ptr();
            let end = ptr.add(slice.len());
            for _ in 0..loops {
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
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn load_a_slice(slice: &[u8], loops: usize) {
    unsafe {
        for _ in 0..loops {
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
}

#[cfg(target_arch = "arm")]
#[inline(never)]
fn load_a_slice(slice: &[u8], loops: usize) {
    unsafe {
        for _ in 0..loops {
            let mut ptr = slice.as_ptr();
            let end = ptr.add(slice.len());
            while ptr < end {
                std::arch::asm!("
                vldmia r1!, {{q0-q3}}
                vldmia r1!, {{q4-q7}}
                    ", inout("r1") ptr,
                out("d0") _, out("d1") _, out("d2") _, out("d3") _,
                out("d4") _, out("d5") _, out("d6") _, out("d7") _,
                out("d8") _, out("d9") _, out("d10") _, out("d11") _,
                out("d12") _, out("d13") _, out("d14") _, out("d15") _,
                );
            }
        }
    }
}

fn bandwidth_seq(slice_len: usize, threads: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        HAS_AVX512 = std::is_x86_feature_detected!("avx512f");
    }
    std::thread::scope(|s| {
        let gards = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let buffer = unsafe { Blob::new_for_size_and_align(slice_len, 1024) };
                    runner::run_bench(|loops| load_a_slice(&buffer, loops))
                })
            })
            .collect_vec();
        let time = gards.into_iter().map(|t| t.join().unwrap()).sum::<f64>() / threads as f64;
        (slice_len * threads * 1) as f64 / time
    })
}

pub fn what_is_big() -> usize {
    1024 * 1024 * if cfg!(target_arch = "arm") { 64 } else { 256 }
}

pub fn l1_bandwidth_seq(threads: usize) -> f64 {
    // [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    [1024]
        .into_iter()
        .map(|slice_len| bandwidth_seq(slice_len, threads))
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
