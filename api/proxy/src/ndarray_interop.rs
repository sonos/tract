/// Generate ndarray interop for [`tract_proxy::Tensor`][crate::Tensor] using the
/// caller crate's own `ndarray` version.
///
/// `tract-proxy` itself has no public `ndarray` dependency: the tensor interface
/// deals only in shapes, slices, bytes and primitive datums. If your
/// application wants the ergonomics of `ndarray`, invoke this macro once at the
/// root of your crate. The macro expands in your crate's scope, so the
/// `ndarray::*` types referenced in the generated code resolve against *your*
/// `ndarray` dependency.
///
/// The generated surface mirrors `tract::impl_ndarray_interop!` exactly: a
/// `Tract` trait with `fn tract(self) -> anyhow::Result<tract_proxy::Tensor>`
/// for `ndarray::ArrayBase`, and an `Ndarray` trait with `ndarray::<T>()` /
/// `ndarray0..ndarray6::<T>()` on `tract_proxy::Tensor`.
///
/// # Invocation
///
/// Zero-argument form uses the `ndarray` crate from your crate's
/// dependencies:
///
/// ```ignore
/// tract_proxy::impl_ndarray_interop!();
/// ```
///
/// Explicit form takes the ndarray root as a path — useful if your
/// `Cargo.toml` renames the crate or pins multiple versions side by side:
///
/// ```ignore
/// tract_proxy::impl_ndarray_interop!(ndarray_017);
/// ```
#[macro_export]
macro_rules! impl_ndarray_interop {
    () => {
        $crate::impl_ndarray_interop!(ndarray);
    };
    ($($nd:ident)::+) => {
        trait Tract {
            fn tract(self) -> $crate::__ndarray_interop::anyhow::Result<$crate::Tensor>;
        }

        impl<T, S, D> Tract for $($nd)::+::ArrayBase<S, D>
        where
            T: $crate::__ndarray_interop::Datum + Clone + 'static,
            S: $($nd)::+::RawData<Elem = T> + $($nd)::+::Data,
            D: $($nd)::+::Dimension,
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
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayViewD<'_, T>>;
            fn ndarray0<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayView0<'_, T>>;
            fn ndarray1<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayView1<'_, T>>;
            fn ndarray2<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayView2<'_, T>>;
            fn ndarray3<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayView3<'_, T>>;
            fn ndarray4<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayView4<'_, T>>;
            fn ndarray5<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayView5<'_, T>>;
            fn ndarray6<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayView6<'_, T>>;
        }

        impl Ndarray for $crate::Tensor {
            fn ndarray<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayViewD<'_, T>> {
                use $crate::__ndarray_interop::TensorInterface as _;
                let (shape, data) = self.as_shape_and_slice::<T>()?;
                Ok(unsafe { $($nd)::+::ArrayViewD::from_shape_ptr(shape, data.as_ptr()) })
            }
            fn ndarray0<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayView0<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray1<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayView1<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray2<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayView2<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray3<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayView3<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray4<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayView4<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray5<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayView5<'_, T>> {
                Ok(self.ndarray::<T>()?.into_dimensionality()?)
            }
            fn ndarray6<T: $crate::__ndarray_interop::Datum>(
                &self,
            ) -> $crate::__ndarray_interop::anyhow::Result<$($nd)::+::ArrayView6<'_, T>> {
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
