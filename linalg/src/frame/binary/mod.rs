pub mod mul;

use std::fmt::Debug;
use std::marker::PhantomData;

use tract_data::TractResult;

use crate::frame::element_wise_helper::TempBuffer;
use crate::LADatum;

macro_rules! binary_impl_wrap {
    ($ti: ident, $func: ident, $nr: expr, $alignment_items: expr, $run: item) => {
        paste! {
            #[derive(Copy, Clone, Debug)]
            #[allow(non_camel_case_types)]
            pub struct $func;

            impl crate::frame::binary::BinaryKer<$ti> for $func {
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

pub trait Binary<T>: Send + Sync + Debug + dyn_clone::DynClone
where
    T: Copy + Debug + PartialEq + Send + Sync,
{
    fn name(&self) -> &'static str;
    fn run(&self, a: &mut [T], b: &[T]) -> TractResult<()>;
}

dyn_clone::clone_trait_object!(<T> Binary<T> where T: Copy);

#[derive(Debug, Clone, new)]
pub struct BinaryImpl<K, T>
where
    T: LADatum,
    K: BinaryKer<T> + Clone,
{
    phantom: PhantomData<(K, T)>,
}

impl<K, T> Binary<T> for BinaryImpl<K, T>
where
    T: LADatum,
    K: BinaryKer<T> + Clone,
{
    fn name(&self) -> &'static str {
        K::name()
    }
    fn run(&self, a: &mut [T], b: &[T]) -> TractResult<()> {
        binary_with_alignment(a, b, |a, b| K::run(a, b), K::nr(), K::alignment_bytes())
    }
}

pub trait BinaryKer<T>: Send + Sync + Debug + dyn_clone::DynClone + Clone + 'static
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
}

std::thread_local! {
    static TMP: std::cell::RefCell<(TempBuffer, TempBuffer)> = std::cell::RefCell::new((TempBuffer::default(), TempBuffer::default()));
}

pub(crate) fn binary_with_alignment<T>(
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
            let prefix_len = a.as_ptr().align_offset(alignment_bytes).min(a.len());
            if prefix_len > 0 {
                compute_via_temp_buffer(&mut a[..prefix_len], &b[..prefix_len]);
            }
            let aligned_len = (a.len() - prefix_len) / nr * nr;
            if aligned_len > 0 {
                f(&mut a[prefix_len..][..aligned_len], &b[prefix_len..][..aligned_len]);
            }
            if prefix_len + aligned_len < a.len() {
                compute_via_temp_buffer(
                    &mut a[prefix_len + aligned_len..],
                    &b[prefix_len + aligned_len..],
                );
            }
        })
    }
    Ok(())
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::LADatum;
    use proptest::test_runner::{TestCaseError, TestCaseResult};
    use tract_data::internal::*;

    pub fn test_binary<K: BinaryKer<T>, T: LADatum>(
        a: &[T],
        b: &[T],
        reference: impl Fn(T, T) -> T,
    ) -> TestCaseResult {
        crate::setup_test_logger();
        let op = BinaryImpl::<K, T>::new();
        let expected = a.iter().zip(b.iter()).map(|(a, b)| (reference)(*a, *b)).collect::<Vec<_>>();
        let mut found = a.to_vec();
        op.run(&mut found, &b).unwrap();
        tensor1(&found)
            .close_enough(&tensor1(&expected), true)
            .map_err(|e| TestCaseError::fail(e.root_cause().to_string()))?;
        Ok(())
    }
}
