use std::fmt::Debug;
use std::marker::PhantomData;

use tract_data::TractResult;

use crate::LADatum;

use super::element_wise_helper::map_slice_with_alignment;

macro_rules! ew_impl_wrap {
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
/// `ew_impl_wrap!`.
///
/// `CHUNK` must be a multiple of `nr`: the f32 kernel steps `nr` lanes with no
/// tail, and each chunk length passed to it is a multiple of `nr` only because
/// both `CHUNK` and every buffer length are.
///
/// The param arm converts the f16-side param into the f32 kernel's param via
/// `$pname => $pconv` (e.g. `f16, alpha => alpha.to_f32()`), computed once per call.
macro_rules! ew_impl_f16_via_f32 {
    ($func:ident, $nr:expr, $alignment_items:expr, $chunk:expr, $scratch_align:literal,
     $cvt_in:path, $cvt_out:path, $f32_kernel:ty) => {
        ew_impl_f16_via_f32!(@build $func, $nr, $alignment_items, $chunk, $scratch_align,
            $cvt_in, $cvt_out, $f32_kernel, (), _params, ());
    };
    ($func:ident, $nr:expr, $alignment_items:expr, $chunk:expr, $scratch_align:literal,
     $cvt_in:path, $cvt_out:path, $f32_kernel:ty, $params:ty, $pname:ident => $pconv:expr) => {
        ew_impl_f16_via_f32!(@build $func, $nr, $alignment_items, $chunk, $scratch_align,
            $cvt_in, $cvt_out, $f32_kernel, $params, $pname, $pconv);
    };
    (@build $func:ident, $nr:expr, $alignment_items:expr, $chunk:expr, $scratch_align:literal,
     $cvt_in:path, $cvt_out:path, $f32_kernel:ty, $params:ty, $pname:ident, $pconv:expr) => {
        ew_impl_wrap!(
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

macro_rules! ew_impl {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr) => {
        paste! {
            mod [<sys_ $func>] {
                #[allow(unused_imports)]
                use tract_data::prelude::f16;
                extern_kernel!(fn $func(ptr: *mut $ti, count: usize) -> ());
            }
            ew_impl_wrap!($ti, $func, $nr, $alignment_items, (),
                #[inline(never)]
                fn run(buf: &mut [$ti], _params: ()) {
                    unsafe { [<sys_ $func>]::$func(buf.as_mut_ptr(), buf.len()) }
                }
            );
        }
    };
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $params: ty) => {
        paste! {
            mod [<sys_ $func>] {
                #[allow(unused_imports)]
                use tract_data::prelude::f16;
                extern_kernel!(fn $func(ptr: *mut $ti, count: usize, params: $params) -> ());
            }
            ew_impl_wrap!($ti, $func, $nr, $alignment_items, $params,
                #[inline(never)]
                fn run(buf: &mut [$ti], params: $params) {
                    unsafe { [<sys_ $func>]::$func(buf.as_mut_ptr(), buf.len(), params) }
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
    use proptest::test_runner::{TestCaseError, TestCaseResult};
    use tract_data::internal::*;

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
