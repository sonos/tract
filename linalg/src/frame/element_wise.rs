use std::fmt::Debug;
use std::marker::PhantomData;

use tract_data::TractResult;

use crate::LADatum;

use super::element_wise_helper::run_over_slice_with_alignment;

macro_rules! ew_impl {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $params: ty) => {
        paste! {
            mod [<sys_ $func>] {
                #[allow(unused_imports)]
                use tract_data::prelude::f16;
                extern_kernel!(fn $func(ptr: *mut $ti, count: usize, params: $params) -> ());
            }

            #[derive(Copy, Clone, Debug)]
            #[allow(non_camel_case_types)]
            pub struct $func;

            impl ElementWiseKer<$ti, $params> for $func {
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
                #[inline(always)]
                fn alignment_bytes() -> usize {
                    $alignment_items * std::mem::size_of::<$ti>()
                }
                #[inline(never)]
                fn run(buf: &mut [$ti], params: $params) {
                    unsafe { [<sys_ $func>]::$func(buf.as_mut_ptr(), buf.len(), params) }
                }
            }
        }
    };
}

pub trait ElementWise<T, Params = usize>: Send + Sync + Debug + dyn_clone::DynClone
where
    Params: Copy + Send + Sync + Debug + 'static,
    T: Copy + Debug + PartialEq + Send + Sync,
{
    fn run(&self, vec: &mut [T], params: Params) -> TractResult<()>;
}

dyn_clone::clone_trait_object!(<T, Params> ElementWise<T, Params> where T: Copy, Params: Copy);

#[derive(Debug, Clone, new)]
pub struct ElementWiseImpl<K, T, Params = usize>
where
    T: LADatum,
    Params: Copy + Send + Sync + Debug + 'static,
    K: ElementWiseKer<T, Params> + Clone,
{
    phantom: PhantomData<(K, T, Params)>,
}

impl<K, T, Params> ElementWise<T, Params> for ElementWiseImpl<K, T, Params>
where
    T: LADatum,
    Params: Copy + Send + Sync + Debug + 'static,
    K: ElementWiseKer<T, Params> + Clone,
{
    fn run(&self, vec: &mut [T], params: Params) -> TractResult<()> {
        run_over_slice_with_alignment(
            vec,
            |data| K::run(data, params),
            K::nr(),
            K::alignment_bytes(),
        )
    }
}

pub trait ElementWiseKer<T, Params = usize>:
    Send + Sync + Debug + dyn_clone::DynClone + Clone + 'static
where
    Params: Copy + Send + Sync + Debug + 'static,
    T: LADatum,
{
    fn name() -> &'static str;
    fn alignment_bytes() -> usize;
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

    pub fn test_element_wise<K: ElementWiseKer<T>, T: LADatum, F: Fn(T) -> T>(
        values: &[T],
        reference: F,
    ) -> TestCaseResult {
        crate::setup_test_logger();
        let op = ElementWiseImpl::<K, T>::new();
        let mut values = values.to_vec();
        while values.len() < K::nr() {
            values.push(T::zero());
        }
        let expected = values.iter().copied().map(reference).collect::<Vec<_>>();
        let mut found = values;
        op.run(&mut found, 0).unwrap();
        tensor1(&found)
            .close_enough(&tensor1(&expected), true)
            .map_err(|e| TestCaseError::fail(e.root_cause().to_string()))?;
        Ok(())
    }
}
