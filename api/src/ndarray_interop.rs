/// Expand ndarray interop code for a concrete tensor type.
///
/// This is the template shared by `tract` and `tract-proxy`. End users
/// should call the zero-argument `impl_ndarray_interop!()` re-exported by
/// the wrapper crate they depend on (`tract::impl_ndarray_interop!()` or
/// `tract_proxy::impl_ndarray_interop!()`); both delegate to this template.
///
/// Parameters:
/// - `$tensor` — path to the concrete tensor type (e.g. `tract::Tensor`).
/// - `$datum` — path to the `Datum` trait as seen from the caller's crate.
/// - `$ti` — path to the `TensorInterface` trait as seen from the caller's
///   crate.
///
/// The macro body is expanded in the caller's crate, so references to
/// `::ndarray::*` and `::anyhow::Result` resolve against the caller's own
/// dependencies. See the docs of the zero-argument wrappers for usage.
#[macro_export]
#[doc(hidden)]
macro_rules! impl_ndarray_interop_with {
    ($tensor:path, $datum:path, $ti:path $(,)?) => {
        trait Tract {
            fn tract(self) -> ::anyhow::Result<$tensor>;
        }

        impl<T, S, D> Tract for ::ndarray::ArrayBase<S, D>
        where
            T: $datum + Clone + 'static,
            S: ::ndarray::RawData<Elem = T> + ::ndarray::Data,
            D: ::ndarray::Dimension,
        {
            fn tract(self) -> ::anyhow::Result<$tensor> {
                use $ti as _;
                if let Some(slice) = self.as_slice_memory_order()
                    && (0..self.ndim()).all(|ix| {
                        self.strides().get(ix + 1).is_none_or(|next| *next <= self.strides()[ix])
                    })
                {
                    <$tensor>::from_slice(self.shape(), slice)
                } else {
                    let slice: ::std::vec::Vec<_> = self.iter().cloned().collect();
                    <$tensor>::from_slice(self.shape(), &slice)
                }
            }
        }

        trait Ndarray {
            fn ndarray<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayViewD<'_, T>>;
            fn ndarray0<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayView0<'_, T>>;
            fn ndarray1<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayView1<'_, T>>;
            fn ndarray2<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayView2<'_, T>>;
            fn ndarray3<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayView3<'_, T>>;
            fn ndarray4<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayView4<'_, T>>;
            fn ndarray5<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayView5<'_, T>>;
            fn ndarray6<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayView6<'_, T>>;
        }

        impl Ndarray for $tensor {
            fn ndarray<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayViewD<'_, T>> {
                use $ti as _;
                let (shape, data) = self.as_shape_and_slice::<T>()?;
                Ok(unsafe { ::ndarray::ArrayViewD::from_shape_ptr(shape, data.as_ptr()) })
            }
            fn ndarray0<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayView0<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray1<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayView1<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray2<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayView2<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray3<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayView3<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray4<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayView4<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray5<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayView5<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray6<T: $datum>(&self) -> ::anyhow::Result<::ndarray::ArrayView6<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
        }
    };
}
