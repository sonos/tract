#[macro_export]
macro_rules! as_inference_fact_impl {
    ($IM:ident, $IF: ident) => {
        impl AsFact<$IM, $IF> for $IF {
            fn as_fact(&self, _model: &$IM) -> Result<boow::Bow<$IF>> {
                Ok(boow::Bow::Borrowed(self))
            }
        }

        impl AsFact<$IM, $IF> for &str {
            fn as_fact(&self, model: &$IM) -> Result<boow::Bow<$IF>> {
                Ok(boow::Bow::Owned($IF::new(model, self)?))
            }
        }

        impl AsFact<$IM, $IF> for () {
            fn as_fact(&self, model: &$IM) -> Result<boow::Bow<$IF>> {
                Ok(boow::Bow::Owned($IF::new(model, "")?))
            }
        }

        impl AsFact<$IM, $IF> for Option<&str> {
            fn as_fact(&self, model: &$IM) -> Result<boow::Bow<$IF>> {
                if let Some(it) = self {
                    Ok(boow::Bow::Owned($IF::new(model, it)?))
                } else {
                    Ok(boow::Bow::Owned($IF::new(model, "")?))
                }
            }
        }
    };
}

#[macro_export]
macro_rules! as_fact_impl {
    ($M:ident, $F: ident) => {
        impl AsFact<$M, $F> for $F {
            fn as_fact(&self, _model: &$M) -> Result<boow::Bow<$F>> {
                Ok(boow::Bow::Borrowed(self))
            }
        }

        impl AsFact<$M, $F> for &str {
            fn as_fact(&self, model: &$M) -> Result<boow::Bow<$F>> {
                Ok(boow::Bow::Owned($F::new(model, self)?))
            }
        }
    };
}

#[macro_export]
macro_rules! tensor_from_to_ndarray {
    () => {
        impl<T, S, D> TryFrom<ndarray::ArrayBase<S, D>> for Tensor
        where
            T: $crate::Datum + Clone + 'static,
            S: RawData<Elem = T> + Data,
            D: Dimension,
        {
            type Error = anyhow::Error;
            fn try_from(view: ndarray::ArrayBase<S, D>) -> Result<Tensor> {
                if let Some(slice) = view.as_slice_memory_order()
                    && (0..view.ndim()).all(|ix| {
                        view.strides().get(ix + 1).is_none_or(|next| *next <= view.strides()[ix])
                    })
                {
                    Tensor::from_slice(view.shape(), slice)
                } else {
                    let slice: Vec<_> = view.iter().cloned().collect();
                    Tensor::from_slice(view.shape(), &slice)
                }
            }
        }

        impl<'a, T: $crate::Datum> TryFrom<&'a Tensor> for ndarray::ArrayViewD<'a, T> {
            type Error = anyhow::Error;
            fn try_from(value: &'a Tensor) -> Result<Self, Self::Error> {
                value.view()
            }
        }
    };
}
