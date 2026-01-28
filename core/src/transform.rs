use crate::internal::*;
#[cfg(feature = "blas")]
use crate::ops::einsum::as_blas::AsBlas;
use crate::ops::matmul::de_block_quant::BlockQuantTransform;
use num_traits::Float;
use std::fmt::Debug;

use tract_data::TractResult;

use crate::floats::FloatPrecisionTranslator;
use crate::ops::nn::{Softmax, SoftmaxExp, SoftmaxKind, TypedModel};

#[macro_export]
macro_rules! rule_if {
    ($cond:expr) => {
        if !$cond {
            return Ok(None);
        }
    };
}

#[macro_export]
macro_rules! rule_if_let {
    ($pat:pat = $expr:expr) => {
        let $pat = $expr else {
            return Ok(None);
        };
    };
}

#[macro_export]
macro_rules! rule_if_some {
    ($pat:pat = $expr:expr) => {
        let Some($pat) = $expr else {
            return Ok(None);
        };
    };
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
    fn name(&self) -> StaticName;
    fn transform(&self, model: &mut TypedModel) -> TractResult<()>;
    fn transform_into(&self, mut model: TypedModel) -> TractResult<TypedModel> {
        self.transform(&mut model)?;
        Ok(model)
    }
}

#[derive(Debug)]
struct SoftmaxFastCompact;

impl ModelTransform for SoftmaxFastCompact {
    fn name(&self) -> StaticName {
        "softmax-fast-compact".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        for node in &mut model.nodes {
            if let Some(softmax) = node.op_as_mut::<Softmax>() {
                if let SoftmaxKind::Softmax(kind) = &mut softmax.kind {
                    *kind = SoftmaxExp::FastCompact
                }
            }
        }
        Ok(())
    }
}

#[allow(clippy::type_complexity)]
pub struct ModelTransformFactory {
    pub name: &'static str,
    pub builder: fn(spec: &str) -> TractResult<Option<Box<dyn ModelTransform>>>,
}

inventory::collect!(ModelTransformFactory);

#[macro_export]
macro_rules! register_simple_model_transform {
    ($name: expr, $type: expr) => {
        $crate::internal::inventory::submit! {
            $crate::transform::ModelTransformFactory {
                name: $name,
                builder: |_| Ok(Some(Box::new($type)))
            }
        }
    };
}

pub fn get_transform(spec: &str) -> TractResult<Option<Box<dyn ModelTransform>>> {
    for factory in inventory::iter::<ModelTransformFactory>() {
        if spec.starts_with(factory.name) {
            return (factory.builder)(spec);
        }
    }
    Ok(None)
}

register_simple_model_transform!("softmax-fast-compact", SoftmaxFastCompact);
#[cfg(feature = "blas")]
register_simple_model_transform!("as-blas", AsBlas);
register_simple_model_transform!("block-quant", BlockQuantTransform);

inventory::submit! {
    ModelTransformFactory {
        name: "f32-to-f16",
        builder: |spec| Ok(build_float_translator::<f32,f16>(spec.strip_prefix("f32-to-f16")))
    }
}

inventory::submit! {
    ModelTransformFactory {
        name: "f16-to-f32",
        builder: |spec| Ok(build_float_translator::<f16,f32>(spec.strip_prefix("f16-to-f32")))
    }
}
