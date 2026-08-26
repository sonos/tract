use std::fmt::Debug;
use std::marker::PhantomData;

use tract_data::TractResult;

use crate::LADatum;

use super::element_wise_helper::map_slice_with_alignment;

// An element-wise kernel from a `run` body. A leading arch ident is for bodies that are
// inline arch asm or intrinsics, which will not even compile elsewhere: those builds get a
// signature-matched panic stub instead, so the kernel struct exists everywhere.
/// Declare an element-wise routine whose body is written here: the kernel, its registry descriptor
/// and its accuracy tests, from one statement. `func` says which cell of the registry it fills and
/// which tests it answers to; `param` marks a kernel taking a scalar of its own type; `isa` says
/// which machines may run it, and therefore which may test it; `boost` is for a kernel that must
/// never be chosen.
macro_rules! routine_ew_rust {
    (arm; $($rest:tt)*) => { routine_ew_rust!(@ arm, target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { routine_ew_rust!(@ aarch64, target_arch = "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*) => { routine_ew_rust!(@ x86_64, target_arch = "x86_64"; $($rest)*); };
    (riscv64; $($rest:tt)*) => {
        routine_ew_rust!(@ riscv64, target_arch = "riscv64"; $($rest)*);
    };
    (wasm32; $($rest:tt)*) => {
        routine_ew_rust!(@ wasm32, all(target_arch = "wasm32", target_feature = "simd128");
            $($rest)*);
    };
    (generic; $($rest:tt)*) => { routine_ew_rust!(@ generic, all(); $($rest)*); };

    // A scalar-parameter kernel takes its own datum type as the parameter, and answers in the
    // parameter shape; a plain one takes nothing.
    (@ $arch:ident, $built:meta; $ti:ident, $ker:ident, $nr:expr, $alignment_items:expr,
     $run:item, func($f:ident), param $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        paste! {
            routine_ew_rust!(@@ $arch, $built; $ti, $ker, $nr, $alignment_items, $ti, $run, $f,
                [<$ti:upper Param>] $(, isa($($isa),+))? $(, boost($boost))?);
        }
    };
    (@ $arch:ident, $built:meta; $ti:ident, $ker:ident, $nr:expr, $alignment_items:expr,
     $run:item, func($f:ident) $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        paste! {
            routine_ew_rust!(@@ $arch, $built; $ti, $ker, $nr, $alignment_items, (), $run, $f,
                [<$ti:upper>] $(, isa($($isa),+))? $(, boost($boost))?);
        }
    };

    (@@ $arch:ident, $built:meta; $ti:ident, $ker:ident, $nr:expr, $alignment_items:expr,
     $params:ty, $run:item, $f:ident, $factory:ident
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        ew_kernel!(@ $built; $ti, $ker, $nr, $alignment_items, $params, $run);
        paste! {
            submit_routine!($arch; $factory, $f, $ker $(, isa($($isa),+))? $(, boost($boost))?);
            #[cfg(test)]
            mod [<test_ $ker:snake>] {
                use super::*;
                crate::[<$f:snake _frame_tests>]!(
                    cfg!($built)
                        && $crate::isa::IsaReq::ANY
                            $(.needing(&[$($crate::isa::Isa::$isa),+]))?
                            .satisfied_by($crate::isa::native()),
                    $ti,
                    $ker
                );
            }
        }
    };
}

macro_rules! ew_kernel {
    (arm; $($rest:tt)*) => { ew_kernel!(@ target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { ew_kernel!(@ target_arch = "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*) => { ew_kernel!(@ target_arch = "x86_64"; $($rest)*); };
    (riscv64; $($rest:tt)*) => { ew_kernel!(@ target_arch = "riscv64"; $($rest)*); };
    (wasm32; $($rest:tt)*) => { ew_kernel!(@ all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*); };

    (@ $built:meta; $ti:ident, $func:ident, $nr:expr, $alignment_items:expr, $params:ty, $run:item) => {
        #[cfg($built)]
        ew_kernel!($ti, $func, $nr, $alignment_items, $params, $run);
        #[cfg(not($built))]
        ew_kernel!($ti, $func, $nr, $alignment_items, $params,
            fn run(_vec: &mut [$ti], _params: $params) {
                panic!(concat!(stringify!($func), ": kernel not built for this target"))
            }
        );
    };

    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $params: ty, $run: item) => {
        paste! {
            #[derive(Copy, Clone, Debug)]
            #[allow(non_camel_case_types)]
            pub struct $func;

            impl crate::frame::element_wise::ElementWiseKer<$ti, $params> for $func {
                #[inline(always)]
                fn name() -> &'static str {
                    stringify!($func)
                }
                #[inline(always)]
                fn nr() -> usize {
                    $nr
                }
                #[inline(always)]
                fn alignment_items() -> usize {
                    $alignment_items
                }
                $run
            }
        }
    };
}

/// Define an f16 element-wise kernel for cores without native f16 arithmetic by
/// round-tripping through an existing f32 kernel: convert each `CHUNK`-sized f16
/// slice into an aligned f32 scratch, run the f32 kernel in place, convert back.
///
/// Callers supply the `unsafe` f16<->f32 conversion fns (their target-feature
/// gating, if any, lives on those fns — this macro is architecture-agnostic), the
/// f32 kernel to reuse, the f32-scratch `CHUNK`, and the scratch alignment (must
/// satisfy the f32 kernel's input-alignment contract, since `run` is called
/// directly, bypassing `map_slice_with_alignment`). The remaining arguments match
/// `ew_kernel!`.
///
/// `CHUNK` must be a multiple of `nr`: the f32 kernel steps `nr` lanes with no
/// tail, and each chunk length passed to it is a multiple of `nr` only because
/// both `CHUNK` and every buffer length are.
///
/// The param arm converts the f16-side param into the f32 kernel's param via
/// `$pname => $pconv` (e.g. `f16, alpha => alpha.to_f32()`), computed once per call.
/// Declare an element-wise routine whose body round-trips through an f32 kernel: the kernel, its
/// registry descriptor and its accuracy tests, from one statement. The arguments after the
/// alignment are the round-trip's -- scratch length, scratch alignment, the two conversions and
/// the f32 kernel to reuse -- and `param` names the f16 parameter and how it converts.
macro_rules! routine_ew_via_f32 {
    (aarch64; $($rest:tt)*) => {
        routine_ew_via_f32!(@ aarch64, target_arch = "aarch64"; $($rest)*);
    };
    (x86_64; $($rest:tt)*) => {
        routine_ew_via_f32!(@ x86_64, target_arch = "x86_64"; $($rest)*);
    };
    (arm; $($rest:tt)*) => { routine_ew_via_f32!(@ arm, target_arch = "arm"; $($rest)*); };
    (wasm32; $($rest:tt)*) => {
        routine_ew_via_f32!(@ wasm32, all(target_arch = "wasm32", target_feature = "simd128");
            $($rest)*);
    };

    (@ $arch:ident, $built:meta; $ker:ident, $nr:expr, $alignment_items:expr, $chunk:expr,
     $scratch_align:literal, $cvt_in:path, $cvt_out:path, $f32_kernel:ty, func($f:ident),
     param($pname:ident => $pconv:expr) $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        ew_kernel_via_f32!($ker, $nr, $alignment_items, $chunk, $scratch_align,
            $cvt_in, $cvt_out, $f32_kernel, f16, $pname => $pconv);
        routine_ew_via_f32!(@@ $arch, $built; $ker, $f, F16Param
            $(, isa($($isa),+))? $(, boost($boost))?);
    };

    (@ $arch:ident, $built:meta; $ker:ident, $nr:expr, $alignment_items:expr, $chunk:expr,
     $scratch_align:literal, $cvt_in:path, $cvt_out:path, $f32_kernel:ty, func($f:ident)
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        ew_kernel_via_f32!($ker, $nr, $alignment_items, $chunk, $scratch_align,
            $cvt_in, $cvt_out, $f32_kernel);
        routine_ew_via_f32!(@@ $arch, $built; $ker, $f, F16
            $(, isa($($isa),+))? $(, boost($boost))?);
    };

    (@@ $arch:ident, $built:meta; $ker:ident, $f:ident, $factory:ident
     $(, isa($($isa:ident),+))? $(, boost($boost:expr))?) => {
        submit_routine!($arch; $factory, $f, $ker
            $(, isa($($isa),+))? $(, boost($boost))?, round_trip(true));
        paste! {
            #[cfg(test)]
            mod [<test_ $ker:snake>] {
                use super::*;
                crate::[<$f:snake _frame_tests>]!(
                    cfg!($built)
                        && $crate::isa::IsaReq::ANY
                            $(.needing(&[$($crate::isa::Isa::$isa),+]))?
                            .satisfied_by($crate::isa::native()),
                    f16,
                    $ker
                );
            }
        }
    };
}

macro_rules! ew_kernel_via_f32 {
    ($func:ident, $nr:expr, $alignment_items:expr, $chunk:expr, $scratch_align:literal,
     $cvt_in:path, $cvt_out:path, $f32_kernel:ty) => {
        ew_kernel_via_f32!(@build $func, $nr, $alignment_items, $chunk, $scratch_align,
            $cvt_in, $cvt_out, $f32_kernel, (), _params, ());
    };
    ($func:ident, $nr:expr, $alignment_items:expr, $chunk:expr, $scratch_align:literal,
     $cvt_in:path, $cvt_out:path, $f32_kernel:ty, $params:ty, $pname:ident => $pconv:expr) => {
        ew_kernel_via_f32!(@build $func, $nr, $alignment_items, $chunk, $scratch_align,
            $cvt_in, $cvt_out, $f32_kernel, $params, $pname, $pconv);
    };
    (@build $func:ident, $nr:expr, $alignment_items:expr, $chunk:expr, $scratch_align:literal,
     $cvt_in:path, $cvt_out:path, $f32_kernel:ty, $params:ty, $pname:ident, $pconv:expr) => {
        ew_kernel!(
            f16, $func, $nr, $alignment_items, $params,
            #[inline(never)]
            fn run(buf: &mut [f16], $pname: $params) {
                const _: () = assert!(
                    $chunk % $nr == 0,
                    "CHUNK must be a multiple of nr; the f32 kernel steps nr lanes with no tail"
                );
                #[repr(C, align($scratch_align))]
                struct AlignedScratch([f32; $chunk]);
                debug_assert!(buf.len() % Self::nr() == 0);
                debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
                if buf.is_empty() {
                    return;
                }
                let f32_params = $pconv;
                let mut scratch = std::mem::MaybeUninit::<AlignedScratch>::uninit();
                // SAFETY: f32 has no invalid bit patterns, and every `s[..n]` element is
                // written by `$cvt_in` before the f32 kernel or `$cvt_out` reads it, so the
                // scratch never needs zero-initialising.
                let s = unsafe { &mut (*scratch.as_mut_ptr()).0 };
                let mut i = 0;
                while i < buf.len() {
                    let n = ($chunk).min(buf.len() - i);
                    unsafe { $cvt_in(&buf[i..i + n], &mut s[..n]) };
                    <$f32_kernel>::run(&mut s[..n], f32_params);
                    unsafe { $cvt_out(&s[..n], &mut buf[i..i + n]) };
                    i += n;
                }
            }
        );
    };
}

// An element-wise kernel whose body is an asm extern. A leading arch ident emits that extern
// only in builds carrying the arch's instructions, replaced elsewhere by a bail stub of the
// same signature, so the module links everywhere.

/// Declare an element-wise routine: the kernel, the registry descriptor, and the accuracy tests,
/// from one statement. The leading architecture ident is the one every kernel-declaration macro
/// takes, omitted for generic Rust; `isa` names what the architecture must offer beyond
/// itself.
///
/// The instruction set is declared once and answers both questions it used to be asked twice:
/// which machines may select the kernel, and which machines may test it. A kernel whose tests
/// are skipped everywhere it could run has nothing left to say, so the two must not be able to
/// disagree.
macro_rules! routine_ew_extern {
    (arm; $($rest:tt)*) => { routine_ew_extern!(@ arm, target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { routine_ew_extern!(@ aarch64, target_arch = "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*) => { routine_ew_extern!(@ x86_64, target_arch = "x86_64"; $($rest)*); };
    (riscv64; $($rest:tt)*) => { routine_ew_extern!(@ riscv64, target_arch = "riscv64"; $($rest)*); };
    (wasm32; $($rest:tt)*) => {
        routine_ew_extern!(@ wasm32,
            all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*);
    };

    (@ $arch:ident, $built:meta; $func:ident, $ti:ident, $ker:ident,
     $nr:expr, $alignment_items:expr $(, isa($($isa:ident),+))?) => {
        ew_kernel_extern!($arch; $ti, $ker, $nr, $alignment_items);
        paste! {
            submit_routine!($arch; [<$ti:upper>], $func, $ker $(, isa($($isa),+))?);
        }
        #[cfg(test)]
        paste! {
            mod [<test_ $ker:snake>] {
                use super::*;
                [<$func:snake _frame_tests>]!(
                    cfg!($built)
                        && $crate::isa::IsaReq::ANY
                            $(.needing(&[$($crate::isa::Isa::$isa),+]))?
                            .satisfied_by($crate::isa::native()),
                    $ti,
                    $ker
                );
            }
        }
    };
}

macro_rules! ew_kernel_extern {
    (arm; $($rest:tt)*) => { ew_kernel_extern!(@ target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { ew_kernel_extern!(@ target_arch = "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*) => { ew_kernel_extern!(@ target_arch = "x86_64"; $($rest)*); };
    (riscv64; $($rest:tt)*) => { ew_kernel_extern!(@ target_arch = "riscv64"; $($rest)*); };
    (wasm32; $($rest:tt)*) => { ew_kernel_extern!(@ all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*); };

    (@ $built:meta; $ti:ident, $func:ident, $nr:expr, $alignment_items:expr) => {
        paste! {
            mod [<sys_ $func>] {
                #[allow(unused_imports)]
                use tract_data::prelude::f16;

                #[cfg($built)]
                extern_kernel!(fn $func(ptr: *mut $ti, count: usize) -> ());

                #[cfg(not($built))]
                #[allow(dead_code)]
                pub unsafe fn $func(_ptr: *mut $ti, _count: usize) {
                    panic!(concat!(stringify!($func), ": activation kernel not built for this target"))
                }
            }
            ew_kernel!($ti, $func, $nr, $alignment_items, (),
                #[inline(never)]
                fn run(buf: &mut [$ti], _params: ()) {
                    unsafe { [<sys_ $func>]::$func(buf.as_mut_ptr(), buf.len()) }
                }
            );
        }
    };

}

pub trait ElementWise<T, Params = ()>: Send + Sync + Debug + dyn_clone::DynClone
where
    Params: Copy + Send + Sync + Debug + 'static + Default,
    T: Copy + Debug + PartialEq + Send + Sync,
{
    fn name(&self) -> &'static str;
    fn run(&self, vec: &mut [T]) -> TractResult<()> {
        self.run_with_params(vec, Params::default())
    }
    fn run_with_params(&self, vec: &mut [T], params: Params) -> TractResult<()>;
}

dyn_clone::clone_trait_object!(<T, Params> ElementWise<T, Params> where T: Copy, Params: Copy);

#[derive(Debug, Clone, new)]
pub struct ElementWiseImpl<K, T, Params = ()>
where
    T: LADatum,
    Params: Copy + Send + Sync + Debug + 'static + Default,
    K: ElementWiseKer<T, Params> + Clone,
{
    phantom: PhantomData<(K, T, Params)>,
}

impl<K, T, Params> ElementWise<T, Params> for ElementWiseImpl<K, T, Params>
where
    T: LADatum,
    Params: Copy + Send + Sync + Debug + 'static + Default,
    K: ElementWiseKer<T, Params> + Clone,
{
    fn name(&self) -> &'static str {
        K::name()
    }
    fn run_with_params(&self, vec: &mut [T], params: Params) -> TractResult<()> {
        map_slice_with_alignment(vec, |data| K::run(data, params), K::nr(), K::alignment_bytes())
    }
}

pub trait ElementWiseKer<T, Params = ()>:
    Send + Sync + Debug + dyn_clone::DynClone + Clone + 'static
where
    Params: Copy + Send + Sync + Debug + 'static + Default,
    T: LADatum,
{
    fn name() -> &'static str;
    fn alignment_bytes() -> usize {
        Self::alignment_items() * T::datum_type().size_of()
    }
    fn alignment_items() -> usize;
    fn nr() -> usize;
    fn run(vec: &mut [T], params: Params);
    fn ew() -> Box<dyn ElementWise<T, Params>> {
        Box::new(ElementWiseImpl::<Self, T, Params>::new())
    }
}

#[cfg(test)]
pub mod test {
    use crate::{LADatum, frame::element_wise::*};
    use num_traits::AsPrimitive;
    use proptest::test_runner::{TestCaseError, TestCaseResult};
    use tract_data::internal::*;

    /// Every finite `f16`, or a 1/4096 grid of `[-30, 30]` for wider types.
    ///
    /// The grid samples where the `f16` set is enumerated, but it reaches past every input
    /// clamp the f32 kernels apply, so no input outside its bounds takes a path it has not
    /// already exercised.
    ///
    /// The step has to stay fine because these invariants break on value-specific
    /// rounding, not over a contiguous region: the Tanh kernels leave `[-1, 1]` only
    /// inside a band about 0.3 wide, and a 1/256 grid steps clean over
    /// `arm64simd_tanh_f32_4n`'s share of it.
    fn invariant_sweep<T: LADatum>() -> Vec<T>
    where
        f32: AsPrimitive<T>,
    {
        if T::datum_type() == f16::datum_type() {
            let all: Vec<f16> =
                (0..=u16::MAX).map(f16::from_bits).filter(|x| x.is_finite()).collect();
            let all = tensor1(&all).cast_to::<T>().unwrap().into_owned();
            return all.try_as_plain().unwrap().as_slice::<T>().unwrap().to_vec();
        }
        (-30 * 4096..=30 * 4096).map(|i| (i as f32 / 4096.).as_()).collect()
    }

    /// Assert `invariant` holds of every `(input, output)` pair a kernel produces over
    /// [`invariant_sweep`], reporting `expected` on the first pair that breaks it.
    ///
    /// The accuracy tests cannot stand in for this on the saturating tails: there the true
    /// value is smaller than the rounding error of the kernels' own arithmetic, so an
    /// output that violates the range or the sign still compares close to the reference.
    pub fn test_element_wise_invariant<K: ElementWiseKer<T>, T: LADatum>(
        expected: &str,
        invariant: impl Fn(T, T) -> bool,
    ) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        crate::setup_test_logger();
        let values = invariant_sweep::<T>();
        let mut found = values.clone();
        K::ew().run(&mut found).unwrap();
        for (x, y) in values.iter().zip(found.iter()) {
            proptest::prop_assert!(
                invariant(*x, *y),
                "{}({x:?}) returned {y:?}, expected {expected}",
                K::name()
            );
        }
        Ok(())
    }

    pub fn test_element_wise<K: ElementWiseKer<T, ()>, T: LADatum, F: Fn(T) -> T>(
        values: &[T],
        reference: F,
    ) -> TestCaseResult {
        test_element_wise_params::<K, T, F, ()>(values, reference, ())
    }

    pub fn test_element_wise_params<
        K: ElementWiseKer<T, Params>,
        T: LADatum,
        F: Fn(T) -> T,
        Params,
    >(
        values: &[T],
        reference: F,
        params: Params,
    ) -> TestCaseResult
    where
        Params: Copy + Send + Sync + Debug + 'static + Default,
    {
        crate::setup_test_logger();
        let op = ElementWiseImpl::<K, T, Params>::new();
        let mut values = values.to_vec();
        while values.len() < K::nr() {
            values.push(T::zero());
        }
        let expected = values.iter().copied().map(reference).collect::<Vec<_>>();
        let mut found = values;
        op.run_with_params(&mut found, params).unwrap();
        tensor1(&found)
            .close_enough(&tensor1(&expected), true)
            .map_err(|e| TestCaseError::fail(e.root_cause().to_string()))?;
        Ok(())
    }
}
