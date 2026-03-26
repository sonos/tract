use std::borrow::Cow;

use crate::internal::*;
#[cfg(feature = "blas")]
use crate::ops::einsum::as_blas::AsBlas;
use crate::ops::matmul::de_block_quant::BlockQuantTransform;
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

/// Structured include/exclude filter for node names.
///
/// If `include` is `None`, all nodes are candidates; if `Some`, only nodes matching
/// at least one pattern are included. `exclude` then removes from that set.
#[derive(Debug, Clone, Default)]
pub struct NodeFilter {
    pub include: Option<Vec<String>>,
    pub exclude: Option<Vec<String>>,
}

impl NodeFilter {
    /// Returns `true` if the given node name passes the filter.
    pub fn matches(&self, name: &str) -> bool {
        let dominated = match &self.include {
            Some(patterns) => patterns.iter().any(|p| name.contains(p)),
            None => true,
        };
        if !dominated {
            return false;
        }
        match &self.exclude {
            Some(patterns) => !patterns.iter().any(|p| name.contains(p)),
            None => true,
        }
    }

    /// Returns `true` when neither include nor exclude is set.
    pub fn is_pass_through(&self) -> bool {
        self.include.is_none() && self.exclude.is_none()
    }
}

/// Parse a legacy filter string (`"!=..."` / `"==..."`) into a `NodeFilter`.
pub fn parse_legacy_filter(filter: Option<&str>) -> TractResult<NodeFilter> {
    let Some(filter) = filter.filter(|f| !f.is_empty()) else {
        return Ok(NodeFilter::default());
    };
    if let Some(patterns) = filter.strip_prefix("!=") {
        let patterns = patterns.split(',').map(|it| it.trim().to_string()).collect();
        Ok(NodeFilter { exclude: Some(patterns), ..Default::default() })
    } else if let Some(patterns) = filter.strip_prefix("==") {
        let patterns = patterns.split(',').map(|it| it.trim().to_string()).collect();
        Ok(NodeFilter { include: Some(patterns), ..Default::default() })
    } else {
        Ok(NodeFilter::default())
    }
}

/// Build Float precision translator given a `NodeFilter`. If the filter is pass-through,
/// all nodes will be translated during the transformation.
pub fn build_float_translator(
    from_dt: DatumType,
    to_dt: DatumType,
    filter: NodeFilter,
) -> Box<dyn ModelTransform> {
    if filter.is_pass_through() {
        return Box::new(FloatPrecisionTranslator::new(from_dt, to_dt));
    }
    Box::new(FloatPrecisionTranslator::with_filter(from_dt, to_dt, move |node| {
        filter.matches(&node.name)
    }))
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
            if let Some(softmax) = node.op_as_mut::<Softmax>()
                && let SoftmaxKind::Softmax(kind) = &mut softmax.kind
            {
                *kind = SoftmaxExp::FastCompact
            }
        }
        Ok(())
    }
}

/// Config for float precision transforms (f32_to_f16, f16_to_f32).
#[derive(Debug, Default, serde::Deserialize)]
pub struct FloatTranslatorConfig {
    /// Legacy filter string (`"!=..."` / `"==..."`).
    #[serde(default)]
    pub filter: Option<String>,
    /// Include patterns — only nodes matching at least one pattern are translated.
    #[serde(default)]
    pub include: Option<Vec<String>>,
    /// Exclude patterns — matching nodes are excluded from translation.
    #[serde(default)]
    pub exclude: Option<Vec<String>>,
}

impl FloatTranslatorConfig {
    pub fn into_node_filter(self) -> TractResult<NodeFilter> {
        if self.include.is_some() || self.exclude.is_some() {
            Ok(NodeFilter { include: self.include, exclude: self.exclude })
        } else {
            parse_legacy_filter(self.filter.as_deref())
        }
    }
}

/// Config for the `float_precision` transform.
#[derive(Debug, serde::Deserialize)]
pub struct FloatPrecisionConfig {
    pub from: String,
    pub to: String,
    /// Include patterns — only nodes matching at least one pattern are translated.
    #[serde(default)]
    pub include: Option<Vec<String>>,
    /// Exclude patterns — matching nodes are excluded from translation.
    #[serde(default)]
    pub exclude: Option<Vec<String>>,
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

/// Config for the `substitute_input_with_shape_of` transform.
#[derive(Debug, serde::Deserialize)]
pub struct SubstituteInputWithShapeOfConfig {
    pub input_to_replace: String,
    pub source_input: String,
    pub axis: usize,
}

#[derive(Debug)]
struct SubstituteInputWithShapeOfTransform(SubstituteInputWithShapeOfConfig);

impl ModelTransform for SubstituteInputWithShapeOfTransform {
    fn name(&self) -> StaticName {
        "substitute_input_with_shape_of".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let cfg = &self.0;

        let length_outlet = model
            .inputs
            .iter()
            .find(|&&o| model.node(o.node).name == cfg.input_to_replace)
            .copied()
            .ok_or_else(|| anyhow!("input '{}' not found in model inputs", cfg.input_to_replace))?;

        let signal_outlet = model
            .inputs
            .iter()
            .find(|&&o| model.node(o.node).name == cfg.source_input)
            .copied()
            .ok_or_else(|| anyhow!("input '{}' not found in model inputs", cfg.source_input))?;

        let dim = model.outlet_fact(signal_outlet)?.shape[cfg.axis].clone();
        let shape_const = tensor1(&[dim]);

        let mut patch = TypedModelPatch::default();
        let const_wire = patch.add_const("length_from_shape", shape_const)?;
        let cast_wire = patch.wire_node(
            "length_cast_i64",
            crate::ops::cast::cast(DatumType::I64),
            &[const_wire],
        )?[0];
        patch.shunt_outside(model, length_outlet, cast_wire)?;
        patch.apply(model)?;

        model.inputs.retain(|&o| o != length_outlet);
        // Shunting an input doesn't update downstream facts; refresh them in-place.
        model.refresh_output_facts()?;
        Ok(())
    }
}

inventory::submit! {
    ModelTransformFactory {
        name: "substitute_input_with_shape_of",
        build_default: || {
            anyhow::bail!(
                "substitute_input_with_shape_of requires 'input_to_replace', 'source_input', and 'axis' parameters"
            )
        },
        build: |de| {
            let config: SubstituteInputWithShapeOfConfig = erased_serde::deserialize(de)
                .map_err(|e| anyhow::anyhow!("deserializing substitute_input_with_shape_of config: {e}"))?;
            Ok(Box::new(SubstituteInputWithShapeOfTransform(config)))
        },
    }
}

inventory::submit! {
    ModelTransformFactory {
        name: "f32_to_f16",
        build_default: || Ok(build_float_translator(DatumType::F32, DatumType::F16, NodeFilter::default())),
        build: |de| {
            let config: FloatTranslatorConfig = erased_serde::deserialize(de)
                .map_err(|e| anyhow::anyhow!("deserializing f32_to_f16 config: {e}"))?;
            Ok(build_float_translator(DatumType::F32, DatumType::F16, config.into_node_filter()?))
        },
    }
}

inventory::submit! {
    ModelTransformFactory {
        name: "f16_to_f32",
        build_default: || Ok(build_float_translator(DatumType::F16, DatumType::F32, NodeFilter::default())),
        build: |de| {
            let config: FloatTranslatorConfig = erased_serde::deserialize(de)
                .map_err(|e| anyhow::anyhow!("deserializing f16_to_f32 config: {e}"))?;
            Ok(build_float_translator(DatumType::F16, DatumType::F32, config.into_node_filter()?))
        },
    }
}

inventory::submit! {
    ModelTransformFactory {
        name: "float_precision",
        build_default: || {
            anyhow::bail!("float_precision transform requires 'from' and 'to' parameters")
        },
        build: |de| {
            let config: FloatPrecisionConfig = erased_serde::deserialize(de)
                .map_err(|e| anyhow::anyhow!("deserializing float_precision config: {e}"))?;
            let from_dt: DatumType = config.from.parse()
                .map_err(|e| anyhow::anyhow!("parsing 'from' datum type: {e}"))?;
            let to_dt: DatumType = config.to.parse()
                .map_err(|e| anyhow::anyhow!("parsing 'to' datum type: {e}"))?;
            let filter = NodeFilter { include: config.include, exclude: config.exclude };
            Ok(build_float_translator(from_dt, to_dt, filter))
        },
    }
}
