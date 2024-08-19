use rayon::prelude::*;
use tract_data::prelude::Blob;

#[path = "../../benches/nano.rs"]
mod nano;

static mut HAS_AVX512: bool = false;

#[cfg(target_arch = "x86_64")]
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
fn load_a_slice(slice: &[u8]) {
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

fn main() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        HAS_AVX512 = std::is_x86_feature_detected!("avx512f");
    }
    let buffer = unsafe { Blob::new_for_size_and_align(1024 * 1024 * 1024, 256) };
    for threads in [1 /*, 2, 3, 4 */] {
        println!("Threads: {}", threads);
        for size in (0..)
            .flat_map(|po2| (0..2).map(move |f| (1024 + 512 * f) * (1 << po2)))
            .take_while(|&s| s < buffer.len())
        {
            let b = (0..threads)
                .into_par_iter()
                .map(|_| nano::run_bench(|| load_a_slice(&buffer[0..size])))
                .sum::<f64>();
            let bw: f64 = size as f64 * threads as f64 / b;
            // println!("{:12} B : {:4.0} GB/s", size, (bw / (1024. * 1024. * 1024.)) as usize);
            println!("{} {}", size, (bw / (1024. * 1024. * 1024.)) as usize);
        }
    }
}
