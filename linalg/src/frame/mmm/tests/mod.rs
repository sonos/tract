use crate::LADatum;
use proptest::prelude::Strategy;
use proptest::test_runner::{Config, TestCaseError, TestRunner};
use tract_data::internal::*;

/// One registered mmm kernel test case: a body already monomorphised over its kernel, plus
/// the predicate saying whether this host can run it. The `mmm_kernels` harness turns each into
/// a trial, marking as ignored — rather than silently passing — those it cannot run.
pub struct MmmTestCase {
    /// [`crate::mmm::MatMatMul::name`] of the kernel under test.
    pub kernel: fn() -> &'static str,
    /// Test family the case comes from, e.g. `frame`.
    pub family: &'static str,
    pub case: &'static str,
    /// Whether this host can run the case: the kernel's own support predicate, and whatever
    /// else the case needs. `run` must not be called when it answers false.
    pub runnable: fn() -> bool,
    pub run: fn() -> TractResult<()>,
}

inventory::collect!(MmmTestCase);

/// Every kernel test case this build registered, in link order.
pub fn cases() -> impl Iterator<Item = &'static MmmTestCase> {
    inventory::iter::<MmmTestCase>()
}

/// Register one test case for `$ker` under `$family`, its body evaluated only when the harness
/// runs it. The `if` form adds a condition beyond the kernel's own support predicate, for cases
/// the kernel cannot serve — an op it does not fuse, a tile too small for the shape.
#[macro_export]
macro_rules! mmm_test_case {
    ($ker:expr, $family:expr, $case:expr, if($cond:expr), $body:expr) => {
        $crate::inventory::submit! {
            $crate::mmm::tests::MmmTestCase {
                kernel: || $crate::mmm::MatMatMul::name($ker),
                family: $family,
                case: $case,
                runnable: || $crate::mmm::MatMatMul::is_supported_here($ker) && $cond,
                run: || $body,
            }
        }
    };
    ($ker:expr, $family:expr, $case:expr, $body:expr) => {
        $crate::mmm_test_case!($ker, $family, $case, if(true), $body);
    };
}

/// Run a proptest strategy from inside a registered case. `proptest!` can only build `#[test]`
/// functions, so cases drive the runner directly; `source_file` must be the caller's `file!()`
/// for the regression file to land beside it.
pub fn run_proptest<T: std::fmt::Debug>(
    source_file: &'static str,
    strategy: impl Strategy<Value = T>,
    check: impl Fn(T) -> TractResult<()>,
) -> TractResult<()> {
    let config = Config { source_file: Some(source_file), ..Config::default() };
    TestRunner::new(config)
        .run(&strategy, |value| check(value).map_err(|e| TestCaseError::fail(e.to_string())))
        .map_err(|e| format_err!("{e}"))
}

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

#[cfg(feature = "test-kernels")]
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
