use std::borrow::Cow;

use crate::internal::*;
use crate::ops::matmul::de_block_quant::BlockQuantTransform;
use std::fmt::Debug;

use tract_data::TractResult;

use crate::floats::FloatPrecisionTranslator;
use crate::ops::nn::TypedModel;

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

/// Per-symbol substitution: either a concrete integer or a TDim
/// expression string parsed against the model's symbol scope.
#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
pub enum SymbolValueSpec {
    Int(i64),
    Expr(String),
}

#[derive(Debug, Default, serde::Deserialize)]
pub struct SetSymbolsConfig {
    pub values: std::collections::HashMap<String, SymbolValueSpec>,
}

#[derive(Debug)]
struct SetSymbolsTransform(SetSymbolsConfig);

impl ModelTransform for SetSymbolsTransform {
    fn name(&self) -> StaticName {
        "set_symbols".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let mut subs = std::collections::HashMap::new();
        for (k, spec) in &self.0.values {
            let sym = model.symbols.sym(k);
            let dim = match spec {
                SymbolValueSpec::Int(v) => TDim::Val(*v),
                SymbolValueSpec::Expr(s) => model
                    .symbols
                    .parse_tdim(s)
                    .with_context(|| format!("Parsing TDim expression {s:?} for symbol {k}"))?,
            };
            subs.insert(sym, dim);
        }
        *model = model.set_symbols(&subs)?;
        Ok(())
    }
}

register_model_transform!("set_symbols", SetSymbolsConfig, |config| Ok(Box::new(
    SetSymbolsTransform(config)
)));

/// Ad-hoc fix-up for NNEF artifacts exported before Scan grew the
/// `external_state` flag (issue #2157). Sets `external_state = true` on every
/// Scan, asserting that the caller plumbs initial state in and reads final
/// state out each call. Apply only when the loaded model is known to use
/// external state management, e.g. the parakeet decoder. Cheaper than
/// re-exporting cached NNEF.
///
/// This does *not* touch the sequence dimension. Inlining the Scan body via
/// `declutter_single_loop` additionally requires `iters == 1`, which is the
/// caller's per-call contract — concretize it explicitly (e.g. `--set
/// TARGETS__TIME=1`), separately from this flag.
#[derive(Debug)]
struct ForceScanExternalState;

impl ModelTransform for ForceScanExternalState {
    fn name(&self) -> StaticName {
        "force_scan_external_state".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        use crate::ops::scan::Scan;
        for node in &mut model.nodes {
            if let Some(scan) = node.op_as_mut::<Scan>() {
                scan.external_state = true;
            }
        }
        Ok(())
    }
}

register_simple_model_transform!("force_scan_external_state", ForceScanExternalState);

/// Hands a single-iteration Scan's recurrent state to the caller, so
/// `declutter_single_loop` can inline the body.
///
/// A Scan whose state is seeded by a constant carries that state inside tract
/// across calls, which `declutter_single_loop` refuses to inline because
/// inlining would rewire the body's state input back to the seed. This rewires
/// the state input to a fresh model input and publishes the scan output — which
/// is the final state when the loop runs once — as a model output, then asserts
/// `external_state`. The caller must thread the added state pair: pass the
/// previous value in, feed the returned value back on the next call.
///
/// Only Scans that run exactly one iteration, with no warm-up (`skip == 0`),
/// and are seeded by a constant are touched. A state already fed by a `Source`
/// belongs to the caller; past one iteration the scan output is the whole
/// sequence rather than a final value; and a Scan still inside its warm-up
/// window suppresses body execution in a way an inlined body cannot reproduce.
/// Model inputs and outputs are appended in Scan node order.
///
/// Output is bit-identical to the untransformed model, but the state round-trip
/// through model I/O is not free: it pays off only where the Scan scaffolding it
/// removes costs more than the state copy it adds. The profile that benefits is
/// a pulse-1 streaming graph whose recurrent state is small relative to the
/// per-frame work — DeepFilterNet3's constant-seeded GRUs are the canonical
/// case. Models with large recurrent state, or whose remaining Scans are
/// already cheap, can regress. Measure before adopting.
#[derive(Debug)]
struct ExportScanState;

impl ModelTransform for ExportScanState {
    fn name(&self) -> StaticName {
        "export_scan_state".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        use crate::ops::konst::Const;
        use crate::ops::scan::{InputMapping, Scan};

        let scans: Vec<usize> =
            model.nodes().iter().filter(|n| n.op_is::<Scan>()).map(|n| n.id).collect();

        for id in scans {
            let eligible = {
                let inputs = model.node_input_facts(id)?;
                let scan = model.node(id).op_as::<Scan>().unwrap();
                scan.skip == 0 && scan.iteration_count(&inputs).map(|i| i.is_one()).unwrap_or(false)
            };
            if !eligible {
                continue;
            }

            let scan = model.node(id).op_as::<Scan>().unwrap();
            let Some(state_output) =
                scan.output_mapping.iter().find(|om| om.state).and_then(|om| om.scan.map(|s| s.0))
            else {
                continue;
            };
            let seeded: Vec<usize> = scan
                .input_mapping
                .iter()
                .enumerate()
                .filter(|(_, im)| matches!(im, InputMapping::State))
                .map(|(slot, _)| slot)
                .filter(|&slot| model.node(model.node(id).inputs[slot].node).op_is::<Const>())
                .collect();
            if seeded.is_empty() {
                continue;
            }

            for slot in seeded {
                let fact = model.outlet_fact(model.node(id).inputs[slot])?.clone().without_value();
                let name = format!("{}.state_in", model.node(id).name);
                let source = model.add_source(name, fact)?;
                model.add_edge(source, InletId::new(id, slot))?;
            }
            model.outputs.push(OutletId::new(id, state_output));
            model.node_mut(id).op_as_mut::<Scan>().unwrap().external_state = true;
        }
        Ok(())
    }
}

register_simple_model_transform!("export_scan_state", ExportScanState);

register_simple_model_transform!("block_quant", BlockQuantTransform);

#[derive(Debug, serde::Deserialize, Default)]
pub struct SelectOutputsConfig {
    pub outputs: Vec<String>,
}

#[derive(Debug)]
struct SelectOutputsTransform(SelectOutputsConfig);

impl ModelTransform for SelectOutputsTransform {
    fn name(&self) -> StaticName {
        "select_outputs".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        model.select_outputs_by_name(self.0.outputs.iter())
    }
}

register_model_transform!("select_outputs", SelectOutputsConfig, |config| Ok(Box::new(
    SelectOutputsTransform(config)
)));

#[derive(Debug, serde::Deserialize, Default)]
pub struct SelectInputsConfig {
    pub inputs: Vec<String>,
}

#[derive(Debug)]
struct SelectInputsTransform(SelectInputsConfig);

impl ModelTransform for SelectInputsTransform {
    fn name(&self) -> StaticName {
        "select_inputs".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        model.select_inputs_by_name(self.0.inputs.iter())
    }
}

register_model_transform!("select_inputs", SelectInputsConfig, |config| Ok(Box::new(
    SelectInputsTransform(config)
)));

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
