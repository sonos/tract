use std::fmt::Debug;
use std::marker::PhantomData;

use tract_data::TractResult;

use crate::LADatum;

use super::element_wise_helper::reduce_slice_with_alignment;

macro_rules! reduce_impl_wrap {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $params: ty, $neutral: expr, $run: item, $reduce_two: item) => {
        paste! {
            #[derive(Copy, Clone, Debug)]
            #[allow(non_camel_case_types)]
            pub struct $func;

            impl crate::frame::reduce::ReduceKer<$ti, $params> for $func {
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
                #[inline(always)]
                fn neutral() -> $ti {
                    $neutral
                }
                $run
                $reduce_two
            }
        }
    };
}

/*
macro_rules! reduce_impl {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr) => {
        paste! {
            mod [<sys_ $func>] {
                #[allow(unused_imports)]
                use tract_data::prelude::f16;
                extern_kernel!(fn $func(ptr: *mut $ti, count: usize) -> ());
            }
            reduce_impl_wrap!($ti, $func, $nr, $alignment_items, (),
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
*/

pub trait Reduce<T, Params = ()>: Send + Sync + Debug + dyn_clone::DynClone
where
    Params: Copy + Send + Sync + Debug + 'static + Default,
    T: Copy + Debug + PartialEq + Send + Sync,
{
    fn run(&self, vec: &[T]) -> TractResult<T> {
        self.run_with_params(vec, Params::default())
    }
    fn run_with_params(&self, vec: &[T], params: Params) -> TractResult<T>;
}

dyn_clone::clone_trait_object!(<T, Params> Reduce<T, Params> where T: Copy, Params: Copy);

#[derive(Debug, Clone, new)]
pub struct ReduceImpl<K, T, Params = ()>
where
    T: LADatum,
    Params: Copy + Send + Sync + Debug + 'static + Default,
    K: ReduceKer<T, Params> + Clone,
{
    phantom: PhantomData<(K, T, Params)>,
}

impl<K, T, Params> Reduce<T, Params> for ReduceImpl<K, T, Params>
where
    T: LADatum,
    Params: Copy + Send + Sync + Debug + 'static + Default,
    K: ReduceKer<T, Params> + Clone,
{
    fn run_with_params(&self, vec: &[T], params: Params) -> TractResult<T> {
        reduce_slice_with_alignment(
            vec,
            |data| K::run(data, params),
            K::nr(),
            K::alignment_bytes(),
            K::neutral(),
            K::reduce_two,
        )
    }
}

pub trait ReduceKer<T, Params = ()>:
    Send + Sync + Debug + dyn_clone::DynClone + Clone + 'static
where
    Params: Copy + Send + Sync + Debug + 'static + Default,
    T: LADatum,
{
    fn name() -> &'static str;
    fn alignment_bytes() -> usize;
    fn alignment_items() -> usize;
    fn nr() -> usize;
    fn neutral() -> T;
    fn reduce_two(a: T, b: T) -> T;
    fn run(vec: &[T], params: Params) -> T;
    fn red() -> Box<dyn Reduce<T, Params>> {
        Box::new(ReduceImpl::<Self, T, Params>::new())
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use proptest::test_runner::{TestCaseError, TestCaseResult};
    use tract_data::internal::*;

    pub fn test_reducer<K: ReduceKer<T, ()>, T: LADatum>(
        values: &[T],
        neutral: T,
        reference_reducer: impl Fn(T, T) -> T,
    ) -> TestCaseResult {
        test_reducer_params::<K, T, ()>(values, neutral, reference_reducer, ())
    }

    pub fn test_reducer_params<K: ReduceKer<T, Params>, T: LADatum, Params>(
        values: &[T],
        neutral: T,
        reference_reducer: impl Fn(T, T) -> T,
        params: Params,
    ) -> TestCaseResult
    where
        Params: Copy + Send + Sync + Debug + 'static + Default,
    {
        crate::setup_test_logger();
        let op = K::red();
        let expected = values.iter().fold(neutral, |acc, i| reference_reducer(acc, *i));
        let mut found = values;
        let red = op.run_with_params(&mut found, params).unwrap();
        tensor0(red)
            .close_enough(&tensor0(expected), true)
            .map_err(|e| TestCaseError::fail(e.root_cause().to_string()))?;
        Ok(())
    }
}
