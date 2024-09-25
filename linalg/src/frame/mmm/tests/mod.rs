use crate::LADatum;

#[macro_use]
pub mod fuse;
#[macro_use]
pub mod frame;
#[macro_use]
pub mod packed_packed;
#[macro_use]
pub mod q_scale;

#[cfg(test)]
macro_rules! test_mmm_kernel {
    (f16, $ker:expr, $cond: expr) => {
        test_mmm_kernel_f16!($ker, $cond);
    };
    (f32, $ker:expr, $cond: expr) => {
        test_mmm_kernel_f32!($ker, $cond);
    };
    (f64, $ker:expr, $cond: expr) => {
        test_mmm_kernel_f64!($ker, $cond);
    };
    (i32, $ker:expr, $cond: expr) => {
        test_mmm_kernel_i32!($ker, $cond);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_f16 {
    ($k: ident, $ker: expr, $cond: expr) => {
        mmm_packed_packed_tests!($cond, $k, &*$ker, f16f16:0);
        mmm_frame_tests!($cond, $k, &*$ker, f16, f16, f16, f16);
        mmm_kernel_fuse_tests!($cond, &*$ker, f16, f16);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_f32 {
    ($ker: expr, $cond: expr) => {
        mmm_packed_packed_tests!($cond, &*$ker, f32f32:0);
        mmm_frame_tests!($cond, &*$ker, f32, f32, f32, f32);
        mmm_kernel_fuse_tests!($cond, &*$ker, f32, f32);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_f64 {
    ($ker:expr, $cond: expr) => {
        mmm_packed_packed_tests!($cond, &*$ker, f64f64:0);
        mmm_frame_tests!($cond, &*$ker, f64, f64, f64, f64);
        mmm_kernel_fuse_tests!($cond, &*$ker, f64, f64);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_i32 {
    ($ker: expr, $cond: expr) => {
        mmm_packed_packed_tests!($cond, &*$ker, i32i32:0);
        mmm_kernel_fuse_tests!($cond, &*$ker, i32, i32);
        mmm_frame_tests!($cond, &*$ker, i32, i32, i32, i32);
        mmm_q_scale_tests!($cond, &*$ker);
    };
}

fn display_error<TC: LADatum>(v: &[TC], expected: &[TC], m: usize, n: usize) {
    if v != expected {
        for ixm in 0..m {
            for ixn in 0..n {
                use nu_ansi_term::Color::*;
                let f = v[ixm * n + ixn];
                let e = expected[ixm * n + ixn];
                let color = if f != e { Red } else { Green };
                print!("{} ", color.paint(format!("{:4}", f)));
            }
            print!("      ");
            for ixn in 0..n {
                print!("{:4} ", expected[ixm * n + ixn]);
            }
            println!();
        }
    }
}
