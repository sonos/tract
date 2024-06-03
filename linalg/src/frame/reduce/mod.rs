pub mod max;
pub mod softmax;
pub mod sum;

use std::fmt::Debug;
use std::marker::PhantomData;

use tract_data::TractResult;

use crate::LADatum;

use super::element_wise_helper::{map_reduce_slice_with_alignment, reduce_slice_with_alignment};

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

pub trait Reduce<T, Params = ()>: Send + Sync + Debug + dyn_clone::DynClone
where
    Params: Copy + Send + Sync + Debug + 'static + Default,
    T: Copy + Debug + PartialEq + Send + Sync,
{
    fn name(&self) -> &'static str;
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
    fn name(&self) -> &'static str {
        K::name()
    }

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
    fn alignment_bytes() -> usize {
        Self::alignment_items() * T::datum_type().size_of()
    }
    fn alignment_items() -> usize;
    fn nr() -> usize;
    fn neutral() -> T;
    fn reduce_two(a: T, b: T) -> T;
    fn run(vec: &[T], params: Params) -> T;
    fn red() -> Box<dyn Reduce<T, Params>> {
        Box::new(ReduceImpl::<Self, T, Params>::new())
    }
}

#[allow(unused_macros)]
macro_rules! map_reduce_impl_wrap {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $params: ty, $map_neutral: expr, $reduce_neutral: expr, $run: item, $reduce_two: item) => {
        paste! {
            #[derive(Copy, Clone, Debug)]
            #[allow(non_camel_case_types)]
            pub struct $func;

            impl crate::frame::reduce::MapReduceKer<$ti, $params> for $func {
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
                fn map_neutral() -> $ti {
                    $map_neutral
                }
                #[inline(always)]
                fn reduce_neutral() -> $ti {
                    $reduce_neutral
                }
                $run
                $reduce_two
            }
        }
    };
}

pub trait MapReduce<T, Params = ()>: Send + Sync + Debug + dyn_clone::DynClone
where
    Params: Copy + Send + Sync + Debug + 'static + Default,
    T: Copy + Debug + PartialEq + Send + Sync,
{
    fn name(&self) -> &'static str;
    fn run(&self, vec: &mut [T]) -> TractResult<T> {
        self.run_with_params(vec, Params::default())
    }
    fn run_with_params(&self, vec: &mut [T], params: Params) -> TractResult<T>;
}

dyn_clone::clone_trait_object!(<T, Params> MapReduce<T, Params> where T: Copy, Params: Copy);

#[derive(Debug, Clone, new)]
pub struct MapReduceImpl<K, T, Params = ()>
where
    T: LADatum,
    Params: Copy + Send + Sync + Debug + 'static + Default,
    K: MapReduceKer<T, Params> + Clone,
{
    phantom: PhantomData<(K, T, Params)>,
}

impl<K, T, Params> MapReduce<T, Params> for MapReduceImpl<K, T, Params>
where
    T: LADatum,
    Params: Copy + Send + Sync + Debug + 'static + Default,
    K: MapReduceKer<T, Params> + Clone,
{
    fn name(&self) -> &'static str {
        K::name()
    }
    fn run_with_params(&self, vec: &mut [T], params: Params) -> TractResult<T> {
        map_reduce_slice_with_alignment(
            vec,
            |data| K::run(data, params),
            K::nr(),
            K::alignment_bytes(),
            K::map_neutral(),
            K::reduce_neutral(),
            K::reduce_two,
        )
    }
}

pub trait MapReduceKer<T, Params = ()>:
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
    fn map_neutral() -> T;
    fn reduce_neutral() -> T;
    fn reduce_two(a: T, b: T) -> T;
    fn run(vec: &mut [T], params: Params) -> T;
    fn red() -> Box<dyn MapReduce<T, Params>> {
        Box::new(MapReduceImpl::<Self, T, Params>::new())
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use proptest::test_runner::{TestCaseError, TestCaseResult};
    use tract_data::internal::*;
    use tract_data::itertools::Itertools;

    pub fn test_reduce<K: ReduceKer<T, ()>, T: LADatum>(
        values: &[T],
        neutral: T,
        reference_reduce: impl Fn(T, T) -> T,
    ) -> TestCaseResult {
        test_reduce_params::<K, T, ()>(values, neutral, reference_reduce, ())
    }

    pub fn test_reduce_params<K: ReduceKer<T, Params>, T: LADatum, Params>(
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
        let found = values;
        let red = op.run_with_params(found, params).unwrap();
        tensor0(red)
            .close_enough(&tensor0(expected), true)
            .map_err(|e| TestCaseError::fail(e.root_cause().to_string()))?;
        Ok(())
    }

    pub fn test_map_reduce<K: MapReduceKer<T, ()>, T: LADatum>(
        values: &[T],
        map_neutral: T,
        neutral: T,
        reference_map: impl Fn(T) -> T,
        reference_reduce: impl Fn(T, T) -> T,
    ) -> TestCaseResult {
        test_map_reduce_params::<K, T, ()>(
            values,
            map_neutral,
            neutral,
            reference_map,
            reference_reduce,
            (),
        )
    }

    pub fn test_map_reduce_params<K: MapReduceKer<T, Params>, T: LADatum, Params>(
        values: &[T],
        _neutral: T,
        map_neutral: T,
        reference_map: impl Fn(T) -> T,
        reference_reducer: impl Fn(T, T) -> T,
        params: Params,
    ) -> TestCaseResult
    where
        Params: Copy + Send + Sync + Debug + 'static + Default,
    {
        crate::setup_test_logger();
        let op = K::red();
        let mut found = values.to_vec();
        let expected_values = values.iter().copied().map(reference_map).collect_vec();
        let expected_reduced =
            expected_values.iter().fold(map_neutral, |acc, i| reference_reducer(acc, *i));
        let red = op.run_with_params(&mut found, params).unwrap();
        tensor1(&found)
            .close_enough(&tensor1(&expected_values), Approximation::SuperApproximate)
            .map_err(|e| TestCaseError::fail(e.root_cause().to_string()))?;
        tensor0(red)
            .close_enough(&tensor0(expected_reduced), Approximation::SuperApproximate)
            .map_err(|e| TestCaseError::fail(e.root_cause().to_string()))?;
        Ok(())
    }
}
