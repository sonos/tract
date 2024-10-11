use super::Tensor;
use crate::datum::Datum;
use ndarray::*;
use std::sync::Arc;

pub fn arr4<A, const N: usize, const M: usize, const T: usize>(xs: &[[[[A; T]; M]; N]]) -> Array4<A>
where
    A: Clone,
{
    use ndarray::*;
    let xs = xs.to_vec();
    let dim = Ix4(xs.len(), N, M, T);
    let len = xs.len();
    let cap = xs.capacity();
    let expand_len = len * N * M * T;
    let ptr = Box::into_raw(xs.into_boxed_slice());
    unsafe {
        let v = if ::std::mem::size_of::<A>() == 0 {
            Vec::from_raw_parts(ptr as *mut A, expand_len, expand_len)
        } else if N == 0 || M == 0 || T == 0 {
            Vec::new()
        } else {
            let expand_cap = cap * N * M * T;
            Vec::from_raw_parts(ptr as *mut A, expand_len, expand_cap)
        };
        ArrayBase::from_shape_vec_unchecked(dim, v)
    }
}

pub fn tensor0<A: Datum>(x: A) -> Tensor {
    unsafe {
        let mut tensor = Tensor::uninitialized::<A>(&[]).unwrap();
        tensor.as_slice_mut_unchecked::<A>()[0] = x;
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
