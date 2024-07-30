use crate::internal::*;
#[cfg(feature = "blas")]
use crate::ops::einsum::as_blas::AsBlas;
use crate::ops::matmul::de_block_quant::BlockQuantTransform;
use num_traits::Float;
use std::borrow::Cow;
use std::fmt::Debug;

use tract_data::TractResult;

use crate::floats::FloatPrecisionTranslator;
use crate::ops::nn::{Softmax, SoftmaxExp, TypedModel};

pub fn get_transform(name: &str) -> Option<Box<dyn ModelTransform>> {
    match name {
        #[cfg(feature = "blas")]
        "as-blas" => Some(Box::<AsBlas>::default()),
        name if name.starts_with("f32-to-f16") => {
            build_float_translator::<f32, f16>(name.strip_prefix("f32-to-f16"))
        }
        name if name.starts_with("f16-to-f32") => {
            build_float_translator::<f16, f32>(name.strip_prefix("f16-to-f32"))
        }
        "softmax-fast-compact" => Some(Box::new(SoftmaxFastCompact)),
        "block-quant" => Some(Box::new(BlockQuantTransform)),
        _ => None,
    }
}

/// Build Float precision translator given a filter_predicate. If the filter_predicate is none or empty, all nodes will
/// be translated during the transformation.
///
/// filter_predicate format:
/// - `==node-name/layer,node-name-layer.1`: Only node which has a name that contains `node-name/layer` or `node-name-layer.1`
/// - `!=node-name/layer,node-name-layer.1`: Only node which has a name that doesn't contain `node-name/layer` or `node-name-layer.1`
pub fn build_float_translator<T1: Datum + Float, T2: Datum + Float>(
    filter_predicate: Option<&str>,
) -> Option<Box<dyn ModelTransform>> {
    let Some(filter_predicate) = filter_predicate.filter(|f| !f.is_empty()) else {
        return Some(Box::<FloatPrecisionTranslator<T1, T2>>::default());
    };

    if let Some(node_name_patterns) = filter_predicate.strip_prefix("!=") {
        let patterns =
            node_name_patterns.split(',').map(|it| it.trim().to_string()).collect::<Vec<_>>();
        Some(Box::new(FloatPrecisionTranslator::<T1, T2>::with_filter(move |node| {
            !patterns.iter().any(|p| node.name.contains(p))
        })))
    } else if let Some(node_name_patterns) = filter_predicate.strip_prefix("==") {
        let patterns =
            node_name_patterns.split(',').map(|it| it.trim().to_string()).collect::<Vec<_>>();
        Some(Box::new(FloatPrecisionTranslator::<T1, T2>::with_filter(move |node| {
            patterns.iter().any(|p| node.name.contains(p))
        })))
    } else {
        None
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
