macro_rules! as_inference_fact_impl {
    ($IM:ident, $IF: ident) => {
        impl AsFact<$IM, $IF> for $IF {
            fn as_fact(&self, _model: &mut $IM) -> Result<boow::Bow<$IF>> {
                Ok(boow::Bow::Borrowed(self))
            }
        }

        impl AsFact<$IM, $IF> for &str {
            fn as_fact(&self, model: &mut $IM) -> Result<boow::Bow<$IF>> {
                Ok(boow::Bow::Owned($IF::new(model, self)?))
            }
        }

        impl AsFact<$IM, $IF> for () {
            fn as_fact(&self, model: &mut $IM) -> Result<boow::Bow<$IF>> {
                Ok(boow::Bow::Owned($IF::new(model, "")?))
            }
        }

        impl AsFact<$IM, $IF> for Option<&str> {
            fn as_fact(&self, model: &mut $IM) -> Result<boow::Bow<$IF>> {
                if let Some(it) = self {
                    Ok(boow::Bow::Owned($IF::new(model, it)?))
                } else {
                    Ok(boow::Bow::Owned($IF::new(model, "")?))
                }
            }
        }
    };
}

macro_rules! as_fact_impl {
    ($M:ident, $F: ident) => {
        impl AsFact<$M, $F> for $F {
            fn as_fact(&self, _model: &mut $M) -> Result<boow::Bow<$F>> {
                Ok(boow::Bow::Borrowed(self))
            }
        }

        impl<S: AsRef<str>> AsFact<$M, $F> for S {
            fn as_fact(&self, model: &mut $M) -> Result<boow::Bow<$F>> {
                Ok(boow::Bow::Owned($F::new(model, self.as_ref())?))
            }
        }
    };
}
