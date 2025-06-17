#![allow(unused_macros)]

use std::time::Duration;
use std::time::Instant;

#[macro_export]
macro_rules! r1 { ($($stat:stmt)*) => { $( $stat )* } }
#[macro_export]
macro_rules! r2 { ($($stat:stmt)*) => { $( $stat )* $( $stat )* } }
#[macro_export]
macro_rules! r4 { ($($stat:stmt)*) => { r2!(r2!($($stat)*)) }}
#[macro_export]
macro_rules! r8 { ($($stat:stmt)*) => { r2!(r4!($($stat)*)) }}
#[macro_export]
macro_rules! r16 { ($($stat:stmt)*) => { r2!(r8!($($stat)*)) }}
#[macro_export]
macro_rules! r32 { ($($stat:stmt)*) => { r2!(r16!($($stat)*)) }}
#[macro_export]
macro_rules! r64 { ($($stat:stmt)*) => { r2!(r32!($($stat)*)) }}
#[macro_export]
macro_rules! r128 { ($($stat:stmt)*) => { r2!(r64!($($stat)*)) }}
#[macro_export]
macro_rules! r256 { ($($stat:stmt)*) => { r2!(r128!($($stat)*)) }}
#[macro_export]
macro_rules! r512 { ($($stat:stmt)*) => { r2!(r256!($($stat)*)) }}
#[macro_export]
macro_rules! r1024 { ($($stat:stmt)*) => { r2!(r512!($($stat)*)) }}
#[macro_export]
macro_rules! r2048 { ($($stat:stmt)*) => { r2!(r1024!($($stat)*)) }}
#[macro_export]
macro_rules! r4096 { ($($stat:stmt)*) => { r2!(r2048!($($stat)*)) }}
#[macro_export]
macro_rules! r8192 { ($($stat:stmt)*) => { r2!(r4096!($($stat)*)) }}

#[macro_export]
macro_rules! b1 { ($($stat:stmt)*) => { nano::run_bench(|| { r1!($($stat)*); }) / 1.0 } }
#[macro_export]
macro_rules! b2 { ($($stat:stmt)*) => { nano::run_bench(|| { r2!($($stat)*); }) / 2.0 } }
#[macro_export]
macro_rules! b4 { ($($stat:stmt)*) => { nano::run_bench(|| { r4!($($stat)*); }) / 4.0 } }
#[macro_export]
macro_rules! b8 { ($($stat:stmt)*) => { nano::run_bench(|| { r8!($($stat)*); }) / 8.0 } }
#[macro_export]
macro_rules! b16 { ($($stat:stmt)*) => { nano::run_bench(|| { r16!($($stat)*); }) / 16.0 } }
#[macro_export]
macro_rules! b32 { ($($stat:stmt)*) => { nano::run_bench(|| { r32!($($stat)*); }) / 32.0 } }
#[macro_export]
macro_rules! b64 { ($($stat:stmt)*) => { nano::run_bench(|| { r64!($($stat)*); }) / 64.0 } }
#[macro_export]
macro_rules! b128 { ($($stat:stmt)*) => { nano::run_bench(|| { r128!($($stat)*); }) / 128.0 } }
#[macro_export]
macro_rules! b256 { ($($stat:stmt)*) => { nano::run_bench(|| { r256!($($stat)*); }) / 256.0 } }
#[macro_export]
macro_rules! b512 { ($($stat:stmt)*) => { nano::run_bench(|| { r512!($($stat)*); }) / 512.0 } }
#[macro_export]
macro_rules! b1024 { ($($stat:stmt)*) => { nano::run_bench(|| { r1024!($($stat)*); }) / 1024.0 } }
#[macro_export]
macro_rules! b2048 { ($($stat:stmt)*) => { nano::run_bench(|| { r2048!($($stat)*); }) / 2048.0 } }
#[macro_export]
macro_rules! b4096 { ($($stat:stmt)*) => { nano::run_bench(|| { r4096!($($stat)*); }) / 4096.0 } }
#[macro_export]
macro_rules! b8192 { ($($stat:stmt)*) => { nano::run_bench(|| { r8192!($($stat)*); }) / 8192.0 } }

#[inline]
fn black_box<T>(dummy: T) -> T {
    unsafe {
        let ret = std::ptr::read_volatile(&dummy);
        std::mem::forget(dummy);
        ret
    }
}

pub fn run_bench<T, F: FnMut(usize) -> T + Copy>(f: F) -> f64 {
    let start = Instant::now();
    let mut f = black_box(f);
    black_box(f(1));
    let once = start.elapsed();
    let evaled = if once < Duration::from_millis(1) {
        let start = Instant::now();
        black_box(f)(1000);
        start.elapsed().as_secs_f64() / 1000.
    } else {
        once.as_secs_f64()
    };

    // we want each individual sample to run for no less than
    let minimum_sampling_time_s = 0.01;
    let minimum_samples = 25;
    let desired_bench_time = 1.0;

    let inner_loops = (minimum_sampling_time_s / evaled).max(1.0) as usize;

    let samples =
        ((desired_bench_time / (inner_loops as f64 * evaled)) as usize).max(minimum_samples);
    let warmup = (1.0 / evaled) as usize;

    // println!(
    //     "evaled: {:?} samples:{samples} inner_loops:{inner_loops} time:{}",
    //     Duration::from_secs_f64(evaled),
    //     (samples * inner_loops) as f64 * evaled
    // );
    let mut measures = vec![0.0; samples];

    black_box(f(warmup));
    for m in &mut measures {
        let start = Instant::now();
        black_box(black_box(f))(inner_loops);
        *m = start.elapsed().as_secs_f64() / inner_loops as f64
    }
    measures
        .sort_by(|a, b| if a < b { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater });
    let q1 = measures[samples / 4];
    let q3 = measures[samples - samples / 4];
    let iq = q3 - q1;
    measures.retain(|&x| x >= q1 - 3. * iq && x <= q3 + 3. * iq);
    measures.iter().copied().sum::<f64>() / measures.len() as f64
}
