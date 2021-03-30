use std::fmt::Debug;
use std::marker::PhantomData;
use tract_data::anyhow;

use tract_data::prelude::Tensor;

pub trait ElementWise<T>: Send + Sync + Debug + dyn_clone::DynClone
where
    T: Copy + Debug + PartialEq + Send + Sync,
{
    fn run(&self, vec: &mut [T]) -> anyhow::Result<()>;
}

dyn_clone::clone_trait_object!(<T> ElementWise<T> where T: Copy);

#[derive(Debug, Clone, new)]
pub struct ElementWiseImpl<K, T>
where
    T: Copy + Debug + PartialEq + Send + Sync,
    K: ElementWiseKer<T> + Clone,
{
    phantom: PhantomData<(K, T)>,
}

impl<K, T> ElementWise<T> for ElementWiseImpl<K, T>
where
    T: crate::Datum + Copy + Debug + PartialEq + Send + Sync,
    K: ElementWiseKer<T> + Clone,
{
    fn run(&self, vec: &mut [T]) -> anyhow::Result<()> {
        if vec.len() == 0 {
            return Ok(());
        }
        unsafe {
            let mut tmp_buffer =
                Tensor::uninitialized_aligned::<T>(&[K::nr()], K::alignment_bytes()).unwrap();
            let mut tmp = tmp_buffer.as_slice_mut_unchecked::<T>();
            let mut compute_via_temp_buffer = |slice: &mut [T]| {
                tmp[..slice.len()].copy_from_slice(slice);
                K::run(&mut tmp);
                slice.copy_from_slice(&tmp[..slice.len()])
            };
            let prefix_len = vec.as_ptr().align_offset(K::alignment_bytes()).min(vec.len());
            if prefix_len > 0 {
                compute_via_temp_buffer(&mut vec[..prefix_len]);
            }
            let aligned_len = (vec.len() - prefix_len) / K::nr() * K::nr();
            if aligned_len > 0 {
                K::run(&mut vec[prefix_len..][..aligned_len]);
            }
            if prefix_len + aligned_len < vec.len() {
                compute_via_temp_buffer(&mut vec[prefix_len + aligned_len..]);
            }
        }
        Ok(())
    }
}

pub trait ElementWiseKer<T>: Send + Sync + Debug + dyn_clone::DynClone + Clone
where
    T: Copy + Debug + PartialEq + Send + Sync,
{
    fn name() -> &'static str;
    fn alignment_bytes() -> usize;
    fn alignment_items() -> usize;
    fn nr() -> usize;
    fn run(vec: &mut [T]);
}
