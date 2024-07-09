use proptest::prelude::*;
use tract_data::internal::*;

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
        mmm_packed_packed_tests!($cond, $k, f16f16:0);
        mmm_frame_tests!($cond, $k, f16, f16, f16, f16);
        mmm_kernel_fuse_tests!($cond, $k, f16, f16);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_f32 {
    ($k: ident, $cond: expr) => {
        mmm_packed_packed_tests!($cond, $k, f32f32:0);
        mmm_frame_tests!($cond, $k, f32, f32, f32, f32);
        mmm_kernel_fuse_tests!($cond, $k, f32, f32);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_f64 {
    ($k: ident, $cond: expr) => {
        mmm_packed_packed_tests!($cond, $k, f64f64:0);
        mmm_frame_tests!($cond, $k, f64, f64, f64, f64);
        mmm_kernel_fuse_tests!($cond, $k, f64, f64);
    };
}

#[macro_export]
macro_rules! test_mmm_kernel_i32 {
    ($k: ident, $cond: expr) => {
        mmm_packed_packed_tests!($cond, $k, i32i32:0);
        mmm_kernel_fuse_tests!($cond, $k, i32, i32);
        mmm_frame_tests!($cond, $k, i32, i32, i32, i32);
        mmm_q_scale_tests!($cond, $k);
    };
}

fn tensor(dt: DatumType, shape: Vec<usize>) -> BoxedStrategy<Tensor> {
    let len = shape.iter().product::<usize>();
    // for f16, positive numbers only to avoid worst rounding side effects
    // and not too big either to avoid overflow :)
    let number = if dt == f16::datum_type() {
        (0i16..100).boxed()
    } else {
        any::<i8>().prop_map(|i| i as i16).boxed()
    };
    proptest::collection::vec(number, len..=len)
        .prop_map(move |vec| {
            tract_ndarray::ArrayD::from_shape_vec(shape.clone(), vec)
                .unwrap()
                .into_tensor()
                .cast_to_dt(dt)
                .unwrap()
                .into_owned()
        })
        .boxed()
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
