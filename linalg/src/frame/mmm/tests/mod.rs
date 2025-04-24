use crate::LADatum;

#[macro_use]
pub mod fuse;
#[macro_use]
pub mod frame;
#[macro_use]
pub mod packed_packed;
#[macro_use]
pub mod q_scale;
#[macro_use]
pub mod store;

#[cfg(test)]
macro_rules! test_mmm_kernel {
    (f16, $ker:expr) => {
        test_mmm_kernel_f16!($ker);
    };
    (f32, $ker:expr) => {
        test_mmm_kernel_f32!($ker);
    };
    (f64, $ker:expr) => {
        test_mmm_kernel_f64!($ker);
    };
    (i32, $ker:expr) => {
        test_mmm_kernel_i32!($ker);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_f16 {
    ($ker: expr) => {
        mmm_packed_packed_tests!(&*$ker, f16f16:0);
        mmm_frame_tests!(&*$ker, f16, f16, f16, f16);
        mmm_kernel_fuse_tests!(&*$ker, f16, f16);
        mmm_store_test!(&*$ker, f16);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_f32 {
    ($ker: expr) => {
        mmm_packed_packed_tests!(&*$ker, f32f32:0);
        mmm_frame_tests!(&*$ker, f32, f32, f32, f32);
        mmm_kernel_fuse_tests!(&*$ker, f32, f32);
        mmm_store_test!(&*$ker, f32);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_f64 {
    ($ker:expr) => {
        mmm_packed_packed_tests!(&*$ker, f64f64:0);
        mmm_frame_tests!(&*$ker, f64, f64, f64, f64);
        mmm_kernel_fuse_tests!(&*$ker, f64, f64);
        mmm_store_test!(&*$ker, f64);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_i32 {
    ($ker: expr) => {
        mmm_packed_packed_tests!(&*$ker, i32i32:0);
        mmm_kernel_fuse_tests!(&*$ker, i32, i32);
        mmm_frame_tests!(&*$ker, i32, i32, i32, i32);
        mmm_q_scale_tests!(&*$ker);
        mmm_store_test!(&*$ker, i32);
    };
}

pub fn display_error<TC: LADatum>(v: &[TC], expected: &[TC], m: usize, n: usize) {
    if v != expected {
        for ixm in 0..m {
            print!("|");
            for ixn in 0..n {
                use nu_ansi_term::Color::*;
                let f = v[ixm * n + ixn];
                let e = expected[ixm * n + ixn];
                let color = if f != e { Red.bold() } else { Green.into() };
                print!("{}|", color.paint(format!("{f:5}")));
            }
            print!("  #  ");
            for ixn in 0..n {
                print!("{:5} ", expected[ixm * n + ixn]);
            }
            println!();
        }
    }
}
