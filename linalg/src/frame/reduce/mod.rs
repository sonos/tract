pub mod max;
pub mod min;
pub mod softmax;
pub mod sum;

use std::fmt::Debug;
use std::marker::PhantomData;

use tract_data::TractResult;

use crate::LADatum;

use super::element_wise_helper::{map_reduce_slice_with_alignment, reduce_slice_with_alignment};

// A reduction kernel from a `run` body. A leading arch ident is for bodies that are inline
// arch asm or intrinsics, which will not even compile elsewhere: those builds get
// signature-matched panic stubs instead, so the kernel struct exists everywhere.
/// Declare a reduction routine: the kernel, its registry descriptor and its accuracy tests, from
/// one statement. `op` is what the kernel folds, which gives the descriptor's function, the tests'
/// reference, the identity to start from and how two answers combine; `isa` says which machines
/// may run it, and therefore which may test it.
macro_rules! routine_reduce_rust {
    (arm; $($rest:tt)*) => { routine_reduce_rust!(@ arm, target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => {
        routine_reduce_rust!(@ aarch64, target_arch = "aarch64"; $($rest)*);
    };
    (x86_64; $($rest:tt)*) => {
        routine_reduce_rust!(@ x86_64, target_arch = "x86_64"; $($rest)*);
    };
    (riscv64; $($rest:tt)*) => {
        routine_reduce_rust!(@ riscv64, target_arch = "riscv64"; $($rest)*);
    };
    (wasm32; $($rest:tt)*) => {
        routine_reduce_rust!(@ wasm32,
            all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*);
    };
    (generic; $($rest:tt)*) => { routine_reduce_rust!(@ generic, all(); $($rest)*); };

    // One arm per operation, each naming the identity it starts from and how two answers combine,
    // then handing the rest on. That is the only place those two facts are written.
    (@ $arch:ident, $built:meta; $ti:ident, $ker:ident, $nr:expr, $alignment_items:expr,
     $run:item, op(Max) $(, isa($($isa:ident),+))?) => {
        routine_reduce_rust!(@@ $arch, $built; $ti, $ker, $nr, $alignment_items, $run, Max,
            <$ti>::MIN, fn reduce_two(a: $ti, b: $ti) -> $ti { a.max(b) }
            $(, isa($($isa),+))?);
    };
    (@ $arch:ident, $built:meta; $ti:ident, $ker:ident, $nr:expr, $alignment_items:expr,
     $run:item, op(Min) $(, isa($($isa:ident),+))?) => {
        routine_reduce_rust!(@@ $arch, $built; $ti, $ker, $nr, $alignment_items, $run, Min,
            <$ti>::MAX, fn reduce_two(a: $ti, b: $ti) -> $ti { a.min(b) }
            $(, isa($($isa),+))?);
    };
    (@ $arch:ident, $built:meta; $ti:ident, $ker:ident, $nr:expr, $alignment_items:expr,
     $run:item, op(Sum) $(, isa($($isa:ident),+))?) => {
        routine_reduce_rust!(@@ $arch, $built; $ti, $ker, $nr, $alignment_items, $run, Sum,
            <$ti as num_traits::Zero>::zero(), fn reduce_two(a: $ti, b: $ti) -> $ti { a + b }
            $(, isa($($isa),+))?);
    };

    (@@ $arch:ident, $built:meta; $ti:ident, $ker:ident, $nr:expr, $alignment_items:expr,
     $run:item, $op:ident, $neutral:expr, $fold:item $(, isa($($isa:ident),+))?) => {
        reduce_kernel!(@ $built; $ti, $ker, $nr, $alignment_items, (), $neutral, $run, $fold);
        paste! {
            submit_routine!($arch; [<$ti:upper Reduce>], [<Reduce $op>], $ker $(, isa($($isa),+))?);
            #[cfg(test)]
            mod [<test_ $ker:snake>] {
                use super::*;
                crate::[<$op:snake _frame_tests>]!(
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

macro_rules! reduce_kernel {
    (arm; $($rest:tt)*) => { reduce_kernel!(@ target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { reduce_kernel!(@ target_arch = "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*) => { reduce_kernel!(@ target_arch = "x86_64"; $($rest)*); };
    (riscv64; $($rest:tt)*) => { reduce_kernel!(@ target_arch = "riscv64"; $($rest)*); };
    (wasm32; $($rest:tt)*) => { reduce_kernel!(@ all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*); };

    (@ $built:meta; $ti:ident, $func:ident, $nr:expr, $alignment_items:expr, $params:ty, $neutral:expr, $run:item, $reduce_two:item) => {
        #[cfg($built)]
        reduce_kernel!($ti, $func, $nr, $alignment_items, $params, $neutral, $run, $reduce_two);
        #[cfg(not($built))]
        reduce_kernel!($ti, $func, $nr, $alignment_items, $params, $neutral,
            fn run(_vec: &[$ti], _params: $params) -> $ti {
                panic!(concat!(stringify!($func), ": kernel not built for this target"))
            },
            fn reduce_two(_a: $ti, _b: $ti) -> $ti {
                panic!(concat!(stringify!($func), ": kernel not built for this target"))
            }
        );
    };

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
// A map-reduce kernel from a `run` body, arch ident as in `reduce_kernel!`.
/// Declare a map-reduction routine: the kernel, its registry descriptor and its accuracy tests,
/// from one statement. One arm per operation, naming the two identities it starts from and how two
/// answers combine, which is the only place those follow from the operation.
macro_rules! routine_map_reduce_rust {
    (arm; $($rest:tt)*) => { routine_map_reduce_rust!(@ arm, target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => {
        routine_map_reduce_rust!(@ aarch64, target_arch = "aarch64"; $($rest)*);
    };
    (x86_64; $($rest:tt)*) => {
        routine_map_reduce_rust!(@ x86_64, target_arch = "x86_64"; $($rest)*);
    };
    (riscv64; $($rest:tt)*) => {
        routine_map_reduce_rust!(@ riscv64, target_arch = "riscv64"; $($rest)*);
    };
    (wasm32; $($rest:tt)*) => {
        routine_map_reduce_rust!(@ wasm32,
            all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*);
    };
    (generic; $($rest:tt)*) => { routine_map_reduce_rust!(@ generic, all(); $($rest)*); };

    (@ $arch:ident, $built:meta; $ti:ident, $ker:ident, $nr:expr, $alignment_items:expr,
     $run:item, op(Softmax2) $(, isa($($isa:ident),+))?) => {
        map_reduce_kernel!(@ $built; $ti, $ker, $nr, $alignment_items, $ti,
            <$ti>::NEG_INFINITY, <$ti as num_traits::Zero>::zero(), $run,
            fn reduce_two(a: $ti, b: $ti) -> $ti { a + b });
        paste! {
            submit_routine!($arch; [<$ti:upper MapReduce>], Softmax2, $ker $(, isa($($isa),+))?);
            #[cfg(test)]
            mod [<test_ $ker:snake>] {
                use super::*;
                crate::softmax_l2_frame_tests!(
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

macro_rules! map_reduce_kernel {
    (arm; $($rest:tt)*) => { map_reduce_kernel!(@ target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { map_reduce_kernel!(@ target_arch = "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*) => { map_reduce_kernel!(@ target_arch = "x86_64"; $($rest)*); };
    (riscv64; $($rest:tt)*) => { map_reduce_kernel!(@ target_arch = "riscv64"; $($rest)*); };
    (wasm32; $($rest:tt)*) => { map_reduce_kernel!(@ all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*); };

    (@ $built:meta; $ti:ident, $func:ident, $nr:expr, $alignment_items:expr, $params:ty, $map_neutral:expr, $reduce_neutral:expr, $run:item, $reduce_two:item) => {
        #[cfg($built)]
        map_reduce_kernel!($ti, $func, $nr, $alignment_items, $params, $map_neutral, $reduce_neutral, $run, $reduce_two);
        #[cfg(not($built))]
        map_reduce_kernel!($ti, $func, $nr, $alignment_items, $params, $map_neutral, $reduce_neutral,
            fn run(_vec: &mut [$ti], _params: $params) -> $ti {
                panic!(concat!(stringify!($func), ": kernel not built for this target"))
            },
            fn reduce_two(_a: $ti, _b: $ti) -> $ti {
                panic!(concat!(stringify!($func), ": kernel not built for this target"))
            }
        );
    };

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
