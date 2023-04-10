use std::fmt::Debug;
use std::marker::PhantomData;

use tract_data::TractResult;

use crate::LADatum;

use super::element_wise_helper::run_over_slice_with_alignment;

macro_rules! ew_impl {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr) => {
        paste! {
            mod [<sys_ $func>] {
                #[allow(unused_imports)]
                use tract_data::prelude::f16;
                extern_kernel!(fn $func(ptr: *mut $ti, count: usize) -> ());
            }

            #[derive(Copy, Clone, Debug)]
            #[allow(non_camel_case_types)]
            pub struct $func;

            impl ElementWiseKer<$ti> for $func {
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
                fn run(buf: &mut [$ti]) {
                    unsafe { [<sys_ $func>]::$func(buf.as_mut_ptr(), buf.len()) }
                }
            }
        }
    };
}

pub trait ElementWise<T>: Send + Sync + Debug + dyn_clone::DynClone
where
    T: Copy + Debug + PartialEq + Send + Sync,
{
    fn run(&self, vec: &mut [T]) -> TractResult<()>;
}

dyn_clone::clone_trait_object!(<T> ElementWise<T> where T: Copy);

#[derive(Debug, Clone, new)]
pub struct ElementWiseImpl<K, T>
where
    T: LADatum,
    K: ElementWiseKer<T> + Clone,
{
    phantom: PhantomData<(K, T)>,
}

impl<K, T> ElementWise<T> for ElementWiseImpl<K, T>
where
    T: LADatum,
    K: ElementWiseKer<T> + Clone,
{
    fn run(&self, vec: &mut [T]) -> TractResult<()> {
        run_over_slice_with_alignment(vec, K::run, K::nr(), K::alignment_bytes())
    }
}


pub trait ElementWiseKer<T>: Send + Sync + Debug + dyn_clone::DynClone + Clone + 'static
where
    T: LADatum,
{
    fn name() -> &'static str;
    fn alignment_bytes() -> usize;
    fn alignment_items() -> usize;
    fn nr() -> usize;
    fn run(vec: &mut [T]);
    fn ew() -> Box<dyn ElementWise<T>> {
        Box::new(ElementWiseImpl::<Self, T>::new())
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
        op.run(&mut found).unwrap();
        tensor1(&found)
            .close_enough(&tensor1(&expected), true)
            .map_err(|e| TestCaseError::fail(e.root_cause().to_string()))?;
        Ok(())
    }
}
