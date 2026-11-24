use super::Tensor;
use crate::datum::Datum;
use ndarray::*;
use std::sync::Arc;

pub fn arr4<A, const N: usize, const M: usize, const T: usize>(xs: &[[[[A; T]; M]; N]]) -> Array4<A>
where
    A: Clone,
{
    let flat = xs.as_flattened().as_flattened().as_flattened().to_vec();
    Array4::from_shape_vec((xs.len(), N, M, T), flat).unwrap()
}

pub fn tensor0<A: Datum>(x: A) -> Tensor {
    unsafe {
        let mut tensor = Tensor::uninitialized::<A>(&[]).unwrap();
        std::ptr::write(tensor.as_slice_mut_unchecked::<A>().as_mut_ptr(), x);
        tensor
    }
}

pub fn tensor1<A: Datum>(xs: &[A]) -> Tensor {
    Tensor::from(arr1(xs))
}

pub fn tensor2<A: Datum, const N: usize>(xs: &[[A; N]]) -> Tensor {
    Tensor::from(arr2(xs))
}

pub fn tensor3<A: Datum, const N: usize, const M: usize>(xs: &[[[A; M]; N]]) -> Tensor {
    Tensor::from(arr3(xs))
}

pub fn tensor4<A: Datum, const N: usize, const M: usize, const T: usize>(
    xs: &[[[[A; T]; M]; N]],
) -> Tensor {
    Tensor::from(arr4(xs))
}

pub fn rctensor0<A: Datum>(x: A) -> Arc<Tensor> {
    Arc::new(Tensor::from(arr0(x)))
}

pub fn rctensor1<A: Datum>(xs: &[A]) -> Arc<Tensor> {
    Arc::new(Tensor::from(arr1(xs)))
}

pub fn rctensor2<A: Datum, const N: usize>(xs: &[[A; N]]) -> Arc<Tensor> {
    Arc::new(Tensor::from(arr2(xs)))
}

pub fn rctensor3<A: Datum, const N: usize, const M: usize>(xs: &[[[A; M]; N]]) -> Arc<Tensor> {
    Arc::new(Tensor::from(arr3(xs)))
}

pub fn rctensor4<A: Datum, const N: usize, const M: usize, const T: usize>(
    xs: &[[[[A; T]; M]; N]],
) -> Arc<Tensor> {
    Arc::new(Tensor::from(arr4(xs)))
}
