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
    use crate::{frame::element_wise::*, LADatum};
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
