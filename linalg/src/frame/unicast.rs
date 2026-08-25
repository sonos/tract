use std::fmt::Debug;
use std::marker::PhantomData;

use tract_data::TractResult;
use tract_data::internal::TensorView;

use crate::frame::element_wise_helper::TempBuffer;
use crate::{BinFn, LADatum};

// A unicast binary kernel from a `run` body. A leading arch ident is for bodies that are
// inline arch asm or intrinsics, which will not even compile elsewhere: those builds get a
// signature-matched panic stub instead, so the kernel struct exists everywhere.
/// Declare a unicast routine: the kernel, its registry descriptor and its accuracy tests, from one
/// statement. `op` is what the kernel computes, which is what the tests compare against, and `isa`
/// says which machines may run it, and therefore which may test it.
macro_rules! routine_unicast_rust {
    (arm; $($rest:tt)*) => { routine_unicast_rust!(@ arm, target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => {
        routine_unicast_rust!(@ aarch64, target_arch = "aarch64"; $($rest)*);
    };
    (x86_64; $($rest:tt)*) => {
        routine_unicast_rust!(@ x86_64, target_arch = "x86_64"; $($rest)*);
    };
    (riscv64; $($rest:tt)*) => {
        routine_unicast_rust!(@ riscv64, target_arch = "riscv64"; $($rest)*);
    };
    (wasm32; $($rest:tt)*) => {
        routine_unicast_rust!(@ wasm32,
            all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*);
    };
    (generic; $($rest:tt)*) => { routine_unicast_rust!(@ generic, all(); $($rest)*); };

    (@ $arch:ident, $built:meta; $ti:ident, $ker:ident, $nr:expr, $alignment_items:expr,
     $run:item, op($op:ident) $(, isa($($isa:ident),+))?) => {
        unicast_kernel!(@ $built; $ti, $ker, $nr, $alignment_items, $run);
        paste! {
            submit_routine!($arch; [<Bin $ti:upper>], BinUnicast($op), $ker $(, isa($($isa),+))?);
            #[cfg(test)]
            mod [<test_ $ker:snake>] {
                use super::*;
                unicast_frame_tests!(
                    cfg!($built)
                        && $crate::isa::IsaReq::ANY
                            $(.needing(&[$($crate::isa::Isa::$isa),+]))?
                            .satisfied_by($crate::isa::native()),
                    $ti,
                    $ker,
                    bin_reference!($op)
                );
            }
        }
    };
}

macro_rules! unicast_kernel {
    (arm; $($rest:tt)*) => { unicast_kernel!(@ target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { unicast_kernel!(@ target_arch = "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*) => { unicast_kernel!(@ target_arch = "x86_64"; $($rest)*); };
    (riscv64; $($rest:tt)*) => { unicast_kernel!(@ target_arch = "riscv64"; $($rest)*); };
    (wasm32; $($rest:tt)*) => { unicast_kernel!(@ all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*); };

    (@ $built:meta; $ti:ident, $func:ident, $nr:expr, $alignment_items:expr, $run:item) => {
        #[cfg($built)]
        unicast_kernel!($ti, $func, $nr, $alignment_items, $run);
        #[cfg(not($built))]
        unicast_kernel!($ti, $func, $nr, $alignment_items,
            fn run(_a: &mut [$ti], _b: &[$ti]) {
                panic!(concat!(stringify!($func), ": kernel not built for this target"))
            }
        );
    };

    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $run: item) => {
        paste! {
            #[derive(Copy, Clone, Debug)]
            #[allow(non_camel_case_types)]
            pub struct $func;

            impl crate::frame::unicast::UnicastKer<$ti> for $func {
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

pub trait Unicast<T>: Send + Sync + Debug + dyn_clone::DynClone
where
    T: Copy + Debug + PartialEq + Send + Sync,
{
    fn name(&self) -> &'static str;
    fn run(&self, a: &mut [T], b: &[T]) -> TractResult<()>;
}

dyn_clone::clone_trait_object!(<T> Unicast<T> where T: Copy);

#[derive(Debug, Clone, new)]
pub struct UnicastImpl<K, T>
where
    T: LADatum,
    K: UnicastKer<T> + Clone,
{
    phantom: PhantomData<(K, T)>,
}

impl<K, T> UnicastImpl<K, T>
where
    T: LADatum,
    K: UnicastKer<T> + Clone,
{
}
impl<K, T> Unicast<T> for UnicastImpl<K, T>
where
    T: LADatum,
    K: UnicastKer<T> + Clone,
{
    fn name(&self) -> &'static str {
        K::name()
    }
    fn run(&self, a: &mut [T], b: &[T]) -> TractResult<()> {
        unicast_with_alignment(a, b, |a, b| K::run(a, b), K::nr(), K::alignment_bytes())
    }
}

pub trait UnicastKer<T>: Send + Sync + Debug + dyn_clone::DynClone + Clone + 'static
where
    T: LADatum,
{
    fn name() -> &'static str;
    fn alignment_bytes() -> usize {
        Self::alignment_items() * T::datum_type().size_of()
    }
    fn alignment_items() -> usize;
    fn nr() -> usize;
    fn run(a: &mut [T], b: &[T]);
    fn bin() -> Box<BinFn> {
        Box::new(|a: &mut TensorView, b: &TensorView| {
            let a_slice = a.as_slice_mut()?;
            let b_slice = b.as_slice()?;
            UnicastImpl::<Self, T>::new().run(a_slice, b_slice)
        })
    }
}

std::thread_local! {
    static TMP: std::cell::RefCell<(TempBuffer, TempBuffer)> = std::cell::RefCell::new((TempBuffer::default(), TempBuffer::default()));
}

pub(crate) fn unicast_with_alignment<T>(
    a: &mut [T],
    b: &[T],
    f: impl Fn(&mut [T], &[T]),
    nr: usize,
    alignment_bytes: usize,
) -> TractResult<()>
where
    T: LADatum,
{
    if a.is_empty() {
        return Ok(());
    }
    unsafe {
        TMP.with(|buffers| {
            let mut buffers = buffers.borrow_mut();
            buffers.0.ensure(nr * T::datum_type().size_of(), alignment_bytes);
            buffers.1.ensure(nr * T::datum_type().size_of(), alignment_bytes);
            let tmp_a = std::slice::from_raw_parts_mut(buffers.0.buffer as *mut T, nr);
            let tmp_b = std::slice::from_raw_parts_mut(buffers.1.buffer as *mut T, nr);
            let mut compute_via_temp_buffer = |a: &mut [T], b: &[T]| {
                tmp_a[..a.len()].copy_from_slice(a);
                tmp_b[..b.len()].copy_from_slice(b);
                f(tmp_a, tmp_b);
                a.copy_from_slice(&tmp_a[..a.len()])
            };

            let mut num_element_processed = 0;
            let a_prefix_len = a.as_ptr().align_offset(alignment_bytes).min(a.len());
            let b_prefix_len = b.as_ptr().align_offset(alignment_bytes).min(b.len());
            assert!(
                a_prefix_len == b_prefix_len,
                "Both inputs should be of the same alignement, got {a_prefix_len:?}, {b_prefix_len:?}"
            );
            let mut applied_prefix_len = 0;
            if a_prefix_len > 0 {
                // Incomplete tile needs to be created to process unaligned data.
                let sub_a = &mut a[..a_prefix_len];
                let sub_b = &b[..a_prefix_len];
                compute_via_temp_buffer(sub_a, sub_b);
                num_element_processed += a_prefix_len;
                applied_prefix_len = a_prefix_len;
            }

            let num_complete_tiles = (a.len() - applied_prefix_len) / nr;
            if num_complete_tiles > 0 {
                // Process all tiles that are complete.
                let sub_a = &mut a[applied_prefix_len..][..(num_complete_tiles * nr)];
                let sub_b = &b[applied_prefix_len..][..(num_complete_tiles * nr)];
                f(sub_a, sub_b);
                num_element_processed += num_complete_tiles * nr;
            }

            if num_element_processed < a.len() {
                // Incomplete tile needs to be created to process remaining elements.
                compute_via_temp_buffer(
                    &mut a[num_element_processed..],
                    &b[num_element_processed..],
                );
            }
        })
    }
    Ok(())
}

#[cfg(test)]
#[macro_use]
pub mod test {
    use super::*;
    use crate::LADatum;
    use proptest::test_runner::{TestCaseError, TestCaseResult};
    use tract_data::internal::*;
    use tract_num_traits::{AsPrimitive, Float};

    pub fn test_unicast<K: UnicastKer<T>, T: LADatum>(
        a: &mut [T],
        b: &[T],
        reference: impl Fn(T, T) -> T,
    ) -> TestCaseResult {
        crate::setup_test_logger();
        let op = UnicastImpl::<K, T>::new();
        let expected = a.iter().zip(b.iter()).map(|(a, b)| (reference)(*a, *b)).collect::<Vec<_>>();
        op.run(a, b).unwrap();
        tensor1(a)
            .close_enough(&tensor1(&expected), true)
            .map_err(|e| TestCaseError::fail(e.root_cause().to_string()))?;
        Ok(())
    }

    pub fn test_unicast_t<K: UnicastKer<T>, T: LADatum + Float>(
        a: &[f32],
        b: &[f32],
        func: impl Fn(T, T) -> T,
    ) -> TestCaseResult
    where
        f32: AsPrimitive<T>,
    {
        crate::setup_test_logger();
        let vec_a: Vec<T> = a.iter().copied().map(|x| x.as_()).collect();
        // We allocate a tensor to ensure allocation is done with alignement
        let mut a = unsafe { Tensor::from_slice_align(vec_a.as_slice(), vector_size()).unwrap() };
        let vec_b: Vec<T> = b.iter().copied().map(|x| x.as_()).collect();
        // We allocate a tensor to ensure allocation is done with alignement
        let b = unsafe { Tensor::from_slice_align(vec_b.as_slice(), vector_size()).unwrap() };
        crate::frame::unicast::test::test_unicast::<K, _>(
            a.try_as_plain_mut().unwrap().as_slice_mut::<T>().unwrap(),
            b.try_as_plain().unwrap().as_slice::<T>().unwrap(),
            func,
        )
    }

    #[macro_export]
    macro_rules! unicast_frame_tests {
        ($cond:expr, $t: ty, $ker:ty, $func:expr) => {
            pastey::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<prop_ $ker:snake>](
                        (a, b) in proptest::strategy::Strategy::prop_flat_map(
                            0..100_usize,
                            |len| (vec![-25f32..25.0; len], vec![-25f32..25.0; len])
                        )
                    ) {
                        if $cond {
                            $crate::frame::unicast::test::test_unicast_t::<$ker, $t>(&*a, &*b, $func).unwrap()
                        }
                    }
                }

                #[test]
                fn [<empty_ $ker:snake>]() {
                    if $cond {
                        $crate::frame::unicast::test::test_unicast_t::<$ker, $t>(&[], &[], $func).unwrap()
                    }
                }
            }
        };
    }
}
