/// Generate ndarray interop for [`tract::Tensor`][crate::Tensor] using the caller
/// crate's own `ndarray` version.
///
/// `tract` itself has no public `ndarray` dependency: the tensor interface deals
/// only in shapes, slices, bytes and primitive datums. If your application wants
/// the ergonomics of `ndarray`, invoke this macro once at the root of your crate.
/// The macro expands in your crate's scope, so the `ndarray::*` types referenced
/// in the generated code resolve against *your* `ndarray` dependency — freeing
/// you from whichever version `tract` happens to use internally.
///
/// Two traits are generated:
///
/// - `Tract`, with method `fn tract(self) -> anyhow::Result<tract::Tensor>`,
///   implemented for every `ndarray::ArrayBase<S, D>` with a `Datum` element
///   type.
/// - `Ndarray`, with methods `ndarray::<T>()`, `ndarray0::<T>()` …
///   `ndarray6::<T>()`, implemented for `tract::Tensor`. `ndarrayN` returns a
///   rank-`N` `ArrayView`; `ndarray` returns the dynamic-rank `ArrayViewD`.
///
/// # Example
///
/// ```ignore
/// tract::impl_ndarray_interop!();
///
/// use ndarray::Array4;
///
/// let input: tract::Tensor = Array4::<f32>::zeros((1, 3, 224, 224)).tract()?;
/// let outputs = model.run([input])?;
/// let view = outputs[0].ndarray::<f32>()?;          // ArrayViewD<f32>
/// let v4   = outputs[0].ndarray4::<f32>()?;         // ArrayView4<f32>
/// ```
///
/// Your crate must depend on `ndarray` directly for this macro to compile.
#[macro_export]
macro_rules! impl_ndarray_interop {
    () => {
        trait Tract {
            fn tract(self) -> $crate::__ndarray_interop::anyhow::Result<$crate::Tensor>;
        }

        impl<T, S, D> Tract for ::ndarray::ArrayBase<S, D>
        where
            T: $crate::__ndarray_interop::Datum + Clone + 'static,
            S: ::ndarray::RawData<Elem = T> + ::ndarray::Data,
            D: ::ndarray::Dimension,
        {
            fn tract(self) -> $crate::__ndarray_interop::anyhow::Result<$crate::Tensor> {
                use $crate::__ndarray_interop::TensorInterface as _;
                if let Some(slice) = self.as_slice_memory_order()
                    && (0..self.ndim()).all(|ix| {
                        self.strides().get(ix + 1).is_none_or(|next| *next <= self.strides()[ix])
                    })
                {
                    $crate::Tensor::from_slice(self.shape(), slice)
                } else {
                    let slice: ::std::vec::Vec<_> = self.iter().cloned().collect();
                    $crate::Tensor::from_slice(self.shape(), &slice)
                }
            }
        }

        trait Ndarray {
            fn ndarray<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayViewD<'_, T>>;
            fn ndarray0<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayView0<'_, T>>;
            fn ndarray1<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayView1<'_, T>>;
            fn ndarray2<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayView2<'_, T>>;
            fn ndarray3<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayView3<'_, T>>;
            fn ndarray4<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayView4<'_, T>>;
            fn ndarray5<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayView5<'_, T>>;
            fn ndarray6<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayView6<'_, T>>;
        }

        impl Ndarray for $crate::Tensor {
            fn ndarray<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayViewD<'_, T>> {
                use $crate::__ndarray_interop::TensorInterface as _;
                let (shape, data) = self.as_shape_and_slice::<T>()?;
                Ok(unsafe { ::ndarray::ArrayViewD::from_shape_ptr(shape, data.as_ptr()) })
            }
            fn ndarray0<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayView0<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray1<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayView1<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray2<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayView2<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray3<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayView3<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray4<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayView4<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray5<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayView5<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray6<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<::ndarray::ArrayView6<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
        }
    };
}

#[doc(hidden)]
pub mod __ndarray_interop {
    pub use ::anyhow;
    pub use ::tract_api::{Datum, TensorInterface};
}
