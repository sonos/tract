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
    (f16, $func:ident, $cond: expr) => {
        test_mmm_kernel_f16!($func, $cond);
    };
    (f32, $func:ident, $cond: expr) => {
        test_mmm_kernel_f32!($func, $cond);
    };
    (f64, $func:ident, $cond: expr) => {
        test_mmm_kernel_f64!($func, $cond);
    };
    (i32, $func:ident, $cond: expr) => {
        test_mmm_kernel_i32!($func, $cond);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_f16 {
    ($k: ident, $cond: expr) => {
        mmm_packed_packed_tests!($cond, $k, f16f16:0, f16, f16, f16, f16);
        mmm_frame_tests!($cond, $k, f16, f16, f16, f16);
        mmm_kernel_fuse_tests!($cond, $k, f16, f16);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_f32 {
    ($k: ident, $cond: expr) => {
        mmm_packed_packed_tests!($cond, $k, f32f32:0, f32, f32, f32, f32);
        mmm_frame_tests!($cond, $k, f32, f32, f32, f32);
        mmm_kernel_fuse_tests!($cond, $k, f32, f32);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_f64 {
    ($k: ident, $cond: expr) => {
        mmm_packed_packed_tests!($cond, $k, f64f64:0, f64, f64, f64, f64);
        mmm_frame_tests!($cond, $k, f64, f64, f64, f64);
        mmm_kernel_fuse_tests!($cond, $k, f64, f64);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_i32 {
    ($k: ident, $cond: expr) => {
        mmm_packed_packed_tests!($cond, $k, i32i32:0, i32, i32, i32, i32);
        mmm_kernel_fuse_tests!($cond, $k, i32, i32);
        mmm_frame_tests!($cond, $k, i32, i32, i32, i32);
        mmm_q_scale_tests!($cond, $k);
    };
}


