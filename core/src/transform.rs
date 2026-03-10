use std::borrow::Cow;

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
) -> Box<dyn ModelTransform> {
    let Some(filter_predicate) = filter_predicate.filter(|f| !f.is_empty()) else {
        return Box::<FloatPrecisionTranslator<T1, T2>>::default();
    };

    if let Some(node_name_patterns) = filter_predicate.strip_prefix("!=") {
        let patterns =
            node_name_patterns.split(',').map(|it| it.trim().to_string()).collect::<Vec<_>>();
        Box::new(FloatPrecisionTranslator::<T1, T2>::with_filter(move |node| {
            !patterns.iter().any(|p| node.name.contains(p))
        }))
    } else if let Some(node_name_patterns) = filter_predicate.strip_prefix("==") {
        let patterns =
            node_name_patterns.split(',').map(|it| it.trim().to_string()).collect::<Vec<_>>();
        Box::new(FloatPrecisionTranslator::<T1, T2>::with_filter(move |node| {
            patterns.iter().any(|p| node.name.contains(p))
        }))
    } else {
        Box::<FloatPrecisionTranslator<T1, T2>>::default()
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
        "softmax_fast_compact".into()
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

/// Config for float precision transforms (f32_to_f16, f16_to_f32).
#[derive(Debug, Default, serde::Deserialize)]
pub struct FloatTranslatorConfig {
    pub filter: Option<String>,
}

pub struct ModelTransformFactory {
    pub name: &'static str,
    /// Build with default config (no params).
    pub build_default: fn() -> TractResult<Box<dyn ModelTransform>>,
    /// Build from a type-erased deserializer.
    pub build: fn(&mut dyn erased_serde::Deserializer) -> TractResult<Box<dyn ModelTransform>>,
}

inventory::collect!(ModelTransformFactory);

#[macro_export]
macro_rules! register_simple_model_transform {
    ($name: expr, $type: expr) => {
        $crate::internal::inventory::submit! {
            $crate::transform::ModelTransformFactory {
                name: $name,
                build_default: || Ok(Box::new($type)),
                build: |_de| Ok(Box::new($type)),
            }
        }
    };
}

#[macro_export]
macro_rules! register_model_transform {
    ($name:expr, $config:ty, $builder:expr) => {
        $crate::internal::inventory::submit! {
            $crate::transform::ModelTransformFactory {
                name: $name,
                build_default: || {
                    let config = <$config>::default();
                    let builder: fn($config) -> $crate::prelude::TractResult<Box<dyn $crate::transform::ModelTransform>> = $builder;
                    builder(config)
                },
                build: |de: &mut dyn erased_serde::Deserializer| {
                    let config: $config = erased_serde::deserialize(de)
                        .map_err(|e| $crate::internal::anyhow!("deserializing transform config: {e}"))?;
                    let builder: fn($config) -> $crate::prelude::TractResult<Box<dyn $crate::transform::ModelTransform>> = $builder;
                    builder(config)
                },
            }
        }
    };
}

/// Split a transform spec like `"f32_to_f16(filter: \"!=layer.norm\")"` into name and params.
pub fn split_spec(spec: &str) -> (Cow<'_, str>, &str) {
    if let Some(pos) = spec.find('(') {
        (Cow::Borrowed(&spec[..pos]), &spec[pos..])
    } else if spec.contains('-') {
        // Backward compat: simple name with no params, convert kebab→snake
        (Cow::Owned(spec.replace('-', "_")), "")
    } else {
        (Cow::Borrowed(spec), "")
    }
}

/// Look up a transform by name, using default config.
pub fn get_transform(name: &str) -> TractResult<Option<Box<dyn ModelTransform>>> {
    let (name, _) = split_spec(name);
    for factory in inventory::iter::<ModelTransformFactory>() {
        if factory.name == &*name {
            return Ok(Some((factory.build_default)()?));
        }
    }
    Ok(None)
}

/// Look up a transform by name, deserializing config from the given deserializer.
pub fn get_transform_with_params(
    name: &str,
    de: &mut dyn erased_serde::Deserializer,
) -> TractResult<Option<Box<dyn ModelTransform>>> {
    for factory in inventory::iter::<ModelTransformFactory>() {
        if factory.name == name {
            return Ok(Some((factory.build)(de)?));
        }
    }
    Ok(None)
}

#[derive(Debug, Default, serde::Deserialize)]
pub struct ConcretizeSymbolsConfig {
    pub values: std::collections::HashMap<String, i64>,
}

#[derive(Debug)]
struct ConcretizeSymbolsTransform(ConcretizeSymbolsConfig);

impl ModelTransform for ConcretizeSymbolsTransform {
    fn name(&self) -> StaticName {
        "concretize_symbols".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let mut table = SymbolValues::default();
        for (k, v) in &self.0.values {
            table = table.with(&model.symbols.sym(k), *v);
        }
        *model = model.concretize_dims(&table)?;
        Ok(())
    }
}

register_model_transform!("concretize_symbols", ConcretizeSymbolsConfig, |config| Ok(Box::new(
    ConcretizeSymbolsTransform(config)
)));

register_simple_model_transform!("softmax_fast_compact", SoftmaxFastCompact);
#[cfg(feature = "blas")]
register_simple_model_transform!("as_blas", AsBlas);
register_simple_model_transform!("block_quant", BlockQuantTransform);

inventory::submit! {
    ModelTransformFactory {
        name: "f32_to_f16",
        build_default: || Ok(build_float_translator::<f32, f16>(None)),
        build: |de| {
            let config: FloatTranslatorConfig = erased_serde::deserialize(de)
                .map_err(|e| anyhow::anyhow!("deserializing f32_to_f16 config: {e}"))?;
            Ok(build_float_translator::<f32, f16>(config.filter.as_deref()))
        },
    }
}

inventory::submit! {
    ModelTransformFactory {
        name: "f16_to_f32",
        build_default: || Ok(build_float_translator::<f16, f32>(None)),
        build: |de| {
            let config: FloatTranslatorConfig = erased_serde::deserialize(de)
                .map_err(|e| anyhow::anyhow!("deserializing f16_to_f32 config: {e}"))?;
            Ok(build_float_translator::<f16, f32>(config.filter.as_deref()))
        },
    }
}
