
use super::Tensor;
use crate::datum::Datum;
use std::sync::Arc;
use ndarray::*;

pub fn arr4<A, V, U, T>(xs: &[V]) -> Array4<A>
where
    V: FixedInitializer<Elem = U> + Clone,
    U: FixedInitializer<Elem = T> + Clone,
    T: FixedInitializer<Elem = A> + Clone,
    A: Clone,
{
    use ndarray::*;
    let mut xs = xs.to_vec();
    let dim = Ix4(xs.len(), V::len(), U::len(), T::len());
    let ptr = xs.as_mut_ptr();
    let len = xs.len();
    let cap = xs.capacity();
    let expand_len = len * V::len() * U::len() * T::len();
    ::std::mem::forget(xs);
    unsafe {
        let v = if ::std::mem::size_of::<A>() == 0 {
            Vec::from_raw_parts(ptr as *mut A, expand_len, expand_len)
        } else if V::len() == 0 || U::len() == 0 || T::len() == 0 {
            Vec::new()
        } else {
            let expand_cap = cap * V::len() * U::len() * T::len();
            Vec::from_raw_parts(ptr as *mut A, expand_len, expand_cap)
        };
        ArrayBase::from_shape_vec_unchecked(dim, v)
    }
}

pub fn tensor0<A: Datum>(x: A) -> Tensor {
    Tensor::from(arr0(x))
}

pub fn tensor1<A: Datum>(xs: &[A]) -> Tensor {
    Tensor::from(arr1(xs))
}

pub fn tensor2<A: Datum, T>(xs: &[T]) -> Tensor
where
    T: FixedInitializer<Elem = A> + Clone,
{
    Tensor::from(arr2(xs))
}

pub fn tensor3<A: Datum, T, U>(xs: &[U]) -> Tensor
where
    U: FixedInitializer<Elem = T> + Clone,
    T: FixedInitializer<Elem = A> + Clone,
{
    Tensor::from(arr3(xs))
}

pub fn tensor4<A: Datum, T, U, V>(xs: &[V]) -> Tensor
where
    V: FixedInitializer<Elem = U> + Clone,
    U: FixedInitializer<Elem = T> + Clone,
    T: FixedInitializer<Elem = A> + Clone,
{
    Tensor::from(arr4(xs))
}

pub fn rctensor0<A: Datum>(x: A) -> Arc<Tensor> {
    Arc::new(Tensor::from(arr0(x)))
}

pub fn rctensor1<A: Datum>(xs: &[A]) -> Arc<Tensor> {
    Arc::new(Tensor::from(arr1(xs)))
}

pub fn rctensor2<A: Datum, T>(xs: &[T]) -> Arc<Tensor>
where
    T: FixedInitializer<Elem = A> + Clone,
{
    Arc::new(Tensor::from(arr2(xs)))
}

pub fn rctensor3<A: Datum, T, U>(xs: &[U]) -> Arc<Tensor>
where
    U: FixedInitializer<Elem = T> + Clone,
    T: FixedInitializer<Elem = A> + Clone,
{
    Arc::new(Tensor::from(arr3(xs)))
}

pub fn rctensor4<A: Datum, T, U, V>(xs: &[V]) -> Arc<Tensor>
where
    V: FixedInitializer<Elem = U> + Clone,
    U: FixedInitializer<Elem = T> + Clone,
    T: FixedInitializer<Elem = A> + Clone,
{
    Arc::new(Tensor::from(arr4(xs)))
}
