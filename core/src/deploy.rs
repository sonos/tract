//! The `deploy` runtime: prepares each model the way a deployment config says,
//! outside the host. The host only asks for the `deploy` runtime by name.
//!
//! The config is TOML, read from the file named by `TRACT_CONFIG`, then from
//! environment variables `TRACT_<MODEL>__<KEY>`, where `<MODEL>` is a
//! lower-case model name upper-cased and `<KEY>` is `RUNTIME` or a knob name
//! without its `TRACT_` prefix. It is loaded once, when the runtime is first
//! used.
//!
//! A global `[knobs]` table sets process-wide knob values; every other table is
//! a model section, keyed by the model's name (its `tract.name`, which NNEF
//! carries as the graph id and the `set_property` transform sets), with an
//! optional `runtime` (default `gpu-or-cpu`), `transforms` (applied in order,
//! each a transform name or a table carrying `name` and the transform's
//! parameters) and a `knobs` table scoped to that model.
//!
//! ```toml
//! [knobs]
//! TRACT_LLC_BYTES = "8M"
//!
//! [encoder]
//! runtime = "cuda"
//! transforms = [{ name = "batchify", symbol = "BATCH" }, "batchify_data_free"]
//!
//! [encoder.knobs]
//! TRACT_AUTOBATCH_LANES = 64
//! TRACT_TURN_LINGER_US = 0
//! ```
//!
//! Preparing a model applies its section's transforms, prepares it on the
//! section's runtime and autobatches it if `TRACT_AUTOBATCH_LANES` is set, all
//! under the section's knob scope. An autobatched model whose batch axis the
//! transforms added keeps the contract it was handed over with: its facts and
//! properties are the unbatched ones, and a stream feeds and gets back tensors
//! without the axis. A model's knobs beat the plain
//! `TRACT_<KNOB>` environment variable, which beats the global `[knobs]` table.
//! A model without a section, unnamed models included, is prepared on
//! `gpu-or-cpu` with no transform.
//!
//! Global knobs are installed when the config is loaded: machine-fact knobs
//! linalg caches for the process are only reached by a config loaded before
//! anything reads them.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use figment2::Figment;
use figment2::providers::Format;
use figment2::providers::Serialized;
use figment2::providers::Toml;
use figment2::value::Value;
use serde::Deserialize;
use serde::Serialize;
use tract_data::knobs::KnobScope;
use tract_data::knobs::set_config_global;

use crate::internal::*;
use crate::lanes::LanedRunnable;
use crate::lanes::TRACT_AUTOBATCH_LANES;
use crate::transform::ModelTransform;
use crate::transform::get_transform;
use crate::transform::get_transform_with_params;

const DEFAULT_RUNTIME: &str = "gpu-or-cpu";
const DEPLOY: &str = "deploy";

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SectionSpec {
    runtime: Option<String>,
    #[serde(default)]
    transforms: Vec<Value>,
    #[serde(default)]
    knobs: BTreeMap<String, Value>,
}

#[derive(Debug, Default)]
struct Section {
    runtime: Option<String>,
    transforms: Vec<Value>,
    scope: KnobScope,
}

#[derive(Debug, Default)]
struct DeployConfig {
    sections: BTreeMap<String, Section>,
}

impl DeployConfig {
    fn from_env() -> TractResult<DeployConfig> {
        let mut figment = Figment::new();
        if let Ok(path) = std::env::var("TRACT_CONFIG") {
            figment = figment.merge(Toml::file(path).search(false).required(true));
        }
        figment = figment.merge(Serialized::defaults(scoped_env()?));
        let root: BTreeMap<String, Value> = figment.extract()?;
        let mut config = DeployConfig::default();
        for (key, value) in root {
            if key == "knobs" {
                let knobs: BTreeMap<String, Value> = value.deserialize()?;
                for (name, value) in knobs {
                    set_config_global(&name, &knob_value(&name, &value)?)?;
                }
            } else {
                let spec: SectionSpec =
                    value.deserialize().with_context(|| format!("In section {key}"))?;
                let knobs = spec
                    .knobs
                    .iter()
                    .map(|(name, value)| Ok((name.as_str(), knob_value(name, value)?)))
                    .collect::<TractResult<Vec<_>>>()?;
                let scope = KnobScope::new(knobs.iter().map(|(n, v)| (*n, v.as_str())))
                    .with_context(|| format!("In section {key}"))?;
                config.sections.insert(
                    key,
                    Section { runtime: spec.runtime, transforms: spec.transforms, scope },
                );
            }
        }
        Ok(config)
    }

    fn prepare(
        &self,
        mut model: TypedModel,
        options: &RunOptions,
    ) -> TractResult<Box<dyn Runnable>> {
        let default = Section::default();
        let section = model.name().and_then(|n| self.sections.get(n)).unwrap_or(&default);
        let runtime = section.runtime.as_deref().unwrap_or(DEFAULT_RUNTIME);
        ensure!(runtime != DEPLOY, "A deploy config section can not use the deploy runtime");
        section.scope.enter(|| {
            let lanes = TRACT_AUTOBATCH_LANES.get();
            let original = lanes.map(|_| model.clone());
            for spec in &section.transforms {
                build_transform(spec)
                    .and_then(|t| t.transform(&mut model))
                    .and_then(|_| model.declutter())
                    .with_context(|| format!("Applying transform {spec:?}"))?;
            }
            let rt =
                runtime_for_name(runtime)?.with_context(|| format!("Unknown runtime {runtime}"))?;
            let runnable = rt.prepare_with_options(model, options)?;
            match lanes.zip(original) {
                Some((lanes, original)) => {
                    Ok(Box::new(LanedRunnable::wrap_batchified(runnable.into(), lanes, &original)?)
                        as _)
                }
                None => Ok(runnable),
            }
        })
    }
}

/// The runtime registered as `deploy`, see the module documentation.
#[derive(Debug)]
pub struct DeployRuntime {
    config: OnceLock<Result<DeployConfig, String>>,
}

impl DeployRuntime {
    fn config(&self) -> TractResult<&DeployConfig> {
        self.config
            .get_or_init(|| DeployConfig::from_env().map_err(|e| format!("{e:?}")))
            .as_ref()
            .map_err(|e| format_err!("Loading the deploy config: {e}"))
    }
}

impl Runtime for DeployRuntime {
    fn name(&self) -> StaticName {
        DEPLOY.into()
    }

    fn check(&self) -> TractResult<()> {
        self.config().map(|_| ())
    }

    fn prepare_with_options(
        &self,
        model: TypedModel,
        options: &RunOptions,
    ) -> TractResult<Box<dyn Runnable>> {
        self.config()?.prepare(model, options)
    }
}

register_runtime!(DeployRuntime = DeployRuntime { config: OnceLock::new() });

fn build_transform(spec: &Value) -> TractResult<Box<dyn ModelTransform>> {
    if let Some(name) = spec.as_str() {
        return get_transform(name)?.with_context(|| format!("No transform named {name}"));
    }
    let Some(dict) = spec.as_dict() else {
        bail!("A transform is a name or a table with a name, got {spec:?}")
    };
    let name =
        dict.get("name").and_then(|n| n.as_str()).context("Transform table without a name")?;
    let mut params = dict.clone();
    params.remove("name");
    let params = Value::from(params);
    let mut erased = <dyn erased_serde::Deserializer>::erase(&params);
    get_transform_with_params(name, &mut erased)?
        .with_context(|| format!("No transform named {name}"))
}

fn knob_value(name: &str, value: &Value) -> TractResult<String> {
    match value {
        Value::String(_, s) => Ok(s.clone()),
        Value::Bool(_, b) => Ok(b.to_string()),
        Value::Num(_, n) => match n.to_i128() {
            Some(i) => Ok(i.to_string()),
            None => Ok(n.to_f64().context("Unrepresentable number")?.to_string()),
        },
        _ => bail!("Knob {name} takes a string, a boolean or a number, got {value:?}"),
    }
}

#[derive(Serialize, Default)]
struct EnvSection {
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime: Option<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    knobs: BTreeMap<String, String>,
}

/// `TRACT_<MODEL>__RUNTIME` and `TRACT_<MODEL>__<KNOB>`, as a config layer.
fn scoped_env() -> TractResult<BTreeMap<String, EnvSection>> {
    let mut sections = BTreeMap::<String, EnvSection>::new();
    for (key, value) in std::env::vars() {
        let Some((model, rest)) = key.strip_prefix("TRACT_").and_then(|k| k.split_once("__"))
        else {
            continue;
        };
        let section = sections.entry(model.to_ascii_lowercase()).or_default();
        match rest {
            "RUNTIME" => section.runtime = Some(value),
            "TRANSFORMS" => {
                bail!("{key}: transforms are set in a config file, not in the environment")
            }
            knob => {
                section.knobs.insert(format!("TRACT_{knob}"), value);
            }
        }
    }
    Ok(sections)
}
