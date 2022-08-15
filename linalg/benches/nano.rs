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

fn black_box<T>(dummy: T) -> T {
    unsafe {
        let ret = std::ptr::read_volatile(&dummy);
        std::mem::forget(dummy);
        ret
    }
}

pub fn run_bench<T, F: FnMut() -> T>(mut f: F) -> f64 {
    let start = Instant::now();
    black_box(f());
    let once = start.elapsed();
    let evaled = if once < Duration::from_millis(1) {
        let start = Instant::now();
        for _ in 0..1000 {
            black_box(f());
        }
        start.elapsed().as_secs_f64() / 1000.
    } else {
        once.as_secs_f64()
    };
    let warmup = (0.3 / evaled) as usize;
    let iters = (0.3 / evaled) as usize;
    let chunks = 1000;
    let chunk = (iters / chunks).max(50);
    let chunks = (iters / chunk).max(50);
    let mut measures = vec![0.0; chunks];
    for _ in 0..warmup {
        black_box(f());
    }
    for m in &mut measures {
        let start = Instant::now();
        for _ in 0..chunk {
            black_box(f());
        }
        *m = start.elapsed().as_secs_f64() / chunk as f64
    }
    measures
        .sort_by(|a, b| if a < b { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater });
    let q1 = measures[chunks / 4];
    let q3 = measures[chunks - chunks / 4];
    let iq = q3 - q1;
    measures.retain(|&x| x >= q1 - 3. * iq && x <= q3 + 3. * iq);
    measures.iter().copied().sum::<f64>() / measures.len() as f64
}
