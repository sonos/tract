use crate::datum::Blob;
use crate::dim::TDim;
use crate::prelude::*;
use crate::tensor::IntoTensor;
use ndarray::*;

pub trait ArrayDatum: Sized {
    unsafe fn stack_tensors(
        axis: usize,
        tensors: &[impl std::borrow::Borrow<Tensor>],
    ) -> anyhow::Result<Tensor>;
    unsafe fn stack_views(axis: usize, views: &[ArrayViewD<Self>]) -> anyhow::Result<ArrayD<Self>>;
    unsafe fn uninitialized_array<S, D, Sh>(shape: Sh) -> ArrayBase<S, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned<Elem = Self>,
        D: Dimension;
}

macro_rules! impl_stack_views_by_copy(
    ($t: ty) => {
        impl ArrayDatum for $t {
            unsafe fn stack_tensors(axis: usize, tensors:&[impl std::borrow::Borrow<Tensor>]) -> anyhow::Result<Tensor> {
                let arrays = tensors.iter().map(|t| t.borrow().to_array_view_unchecked::<$t>()).collect::<TVec<_>>();
                Self::stack_views(axis, &arrays).map(|a| a.into_tensor())
            }

            unsafe fn stack_views(axis: usize, views:&[ArrayViewD<$t>]) -> anyhow::Result<ArrayD<$t>> {
                Ok(ndarray::stack(ndarray::Axis(axis), views)?)
            }
            unsafe fn uninitialized_array<S, D, Sh>(shape: Sh) -> ArrayBase<S, D> where
                Sh: ShapeBuilder<Dim = D>,
                S: DataOwned<Elem=Self>,
                D: Dimension {
                    ArrayBase::<S,D>::uninitialized(shape)
                }
        }
    };
);

macro_rules! impl_stack_views_by_clone(
    ($t: ty) => {
        impl ArrayDatum for $t {
            unsafe fn stack_tensors(axis: usize, tensors:&[impl std::borrow::Borrow<Tensor>]) -> anyhow::Result<Tensor> {
                let arrays = tensors.iter().map(|t| t.borrow().to_array_view::<$t>()).collect::<anyhow::Result<TVec<_>>>()?;
                let views = arrays.iter().map(|a| a.view()).collect::<TVec<_>>();
                Self::stack_views(axis, &views).map(|a| a.into_tensor())
            }

            unsafe fn stack_views(axis: usize, views:&[ArrayViewD<$t>]) -> anyhow::Result<ArrayD<$t>> {
                let mut shape = views[0].shape().to_vec();
                shape[axis] = views.iter().map(|v| v.shape()[axis]).sum();
                let mut array = ndarray::Array::default(&*shape);
                let mut offset = 0;
                for v in views {
                    let len = v.shape()[axis];
                    array.slice_axis_mut(Axis(axis), (offset..(offset + len)).into()).assign(&v);
                    offset += len;
                }
                Ok(array)
            }

            unsafe fn uninitialized_array<S, D, Sh>(shape: Sh) -> ArrayBase<S, D> where
                Sh: ShapeBuilder<Dim = D>,
                S: DataOwned<Elem=Self>,
                D: Dimension {
                    ArrayBase::<S,D>::default(shape)
                }
        }
    };
);

impl_stack_views_by_copy!(i8);
impl_stack_views_by_copy!(i16);
impl_stack_views_by_copy!(i32);
impl_stack_views_by_copy!(i64);

impl_stack_views_by_clone!(Blob);
impl_stack_views_by_clone!(String);
impl_stack_views_by_clone!(TDim);
