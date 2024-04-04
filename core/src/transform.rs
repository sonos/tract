use crate::internal::*;
#[cfg(feature="blas")]
use crate::ops::einsum::as_blas::AsBlas;
use std::borrow::Cow;
use std::fmt::Debug;

use tract_data::TractResult;

use crate::floats::FloatPrecisionTranslator;
use crate::ops::nn::{Softmax, SoftmaxExp, TypedModel};

pub fn get_transform(name: &str) -> Option<Box<dyn ModelTransform>> {
    match name {
        #[cfg(feature="blas")]
        "as-blas" => Some(Box::<AsBlas>::default()),
        "f32-to-f16" => Some(Box::<FloatPrecisionTranslator<f32, f16>>::default()),
        "f16-to-f32" => Some(Box::<FloatPrecisionTranslator<f16, f32>>::default()),
        "softmax-fast-compact" => Some(Box::new(SoftmaxFastCompact)),
        _ => None,
    }
}

pub trait ModelTransform: Debug {
    fn name(&self) -> Cow<str>;
    fn transform(&self, model: &mut TypedModel) -> TractResult<()>;
    fn transform_into(&self, model: &TypedModel) -> TractResult<TypedModel> {
        let mut model = model.clone();
        self.transform(&mut model)?;
        Ok(model)
    }
}

#[derive(Debug)]
struct SoftmaxFastCompact;

impl ModelTransform for SoftmaxFastCompact {
    fn name(&self) -> Cow<str> {
        "softmax-fast-compact".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        for node in &mut model.nodes {
            if let Some(softmax) = node.op_as_mut::<Softmax>() {
                softmax.exp = SoftmaxExp::FastCompact;
            }
        }
        Ok(())
    }
}
