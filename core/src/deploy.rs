//! Deployment config: how each model a host loads is transformed and prepared,
//! and the knobs it runs with, kept outside the host.
//!
//! A config is TOML. A global `[knobs]` table sets process-wide knob values;
//! every other table is a model section, keyed by the model's name (its
//! `tract.name`, which NNEF carries as the graph id), with an optional
//! `runtime` (default `gpu-or-cpu`), `transforms` (applied in order, each a
//! transform name or a table carrying `name` and the transform's parameters)
//! and a `knobs` table scoped to that model.
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
//! Sources, later ones winning per key (arrays are replaced, never
//! concatenated): the bundle config the host hands over, the file named by
//! `TRACT_CONFIG`, then environment variables `TRACT_<MODEL>__<KEY>`, where
//! `<MODEL>` is the model name, which must be lower-case, upper-cased, and `<KEY>` is `RUNTIME` or a knob
//! name without its `TRACT_` prefix. A model's knobs beat the plain
//! `TRACT_<KNOB>` environment variable, which beats the global `[knobs]` table.
//!
//! Global knobs are installed when the config is loaded, so load it before
//! anything reads the machine-fact knobs linalg caches for the process.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

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

/// A loaded deployment config. A model it has no section for, unnamed models
/// included, is prepared on `gpu-or-cpu` with no transform and global knobs
/// only.
#[derive(Debug, Default)]
pub struct DeployConfig {
    sections: BTreeMap<String, Section>,
}

impl DeployConfig {
    /// Load the bundle config at `bundle`, then `TRACT_CONFIG` and the
    /// environment over it.
    pub fn load(bundle: impl AsRef<Path>) -> TractResult<DeployConfig> {
        Self::from_figment(Figment::from(Toml::file(bundle.as_ref()).search(false).required(true)))
    }

    /// Like [`load`](Self::load), with the bundle config given as TOML text.
    pub fn load_str(bundle: &str) -> TractResult<DeployConfig> {
        Self::from_figment(Figment::from(Toml::string(bundle)))
    }

    /// `TRACT_CONFIG` and the environment alone, for a host without a bundle
    /// config.
    pub fn from_env() -> TractResult<DeployConfig> {
        Self::from_figment(Figment::new())
    }

    fn from_figment(mut figment: Figment) -> TractResult<DeployConfig> {
        if let Ok(ops) = std::env::var("TRACT_CONFIG") {
            figment = figment.merge(Toml::file(ops).search(false).required(true));
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

    fn section(&self, model: &TypedModel) -> Option<&Section> {
        self.sections.get(model.name()?)
    }

    /// Apply the transforms of `model`'s section, in order, each followed by a
    /// declutter, under the section's knob scope.
    pub fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let Some(section) = self.section(model) else { return Ok(()) };
        section.scope.enter(|| {
            for spec in &section.transforms {
                build_transform(spec)
                    .and_then(|t| t.transform(model))
                    .and_then(|_| model.declutter())
                    .with_context(|| format!("Applying transform {spec:?}"))?;
            }
            Ok(())
        })
    }

    /// Prepare `model` on its section's runtime, then wrapped to serve
    /// `TRACT_AUTOBATCH_LANES` sessions if that knob is set, under the
    /// section's knob scope.
    pub fn prepare(&self, model: TypedModel) -> TractResult<Arc<dyn Runnable>> {
        self.prepare_with_options(model, &RunOptions::default())
    }

    /// [`prepare`](Self::prepare), with `options` for the runtime.
    pub fn prepare_with_options(
        &self,
        model: TypedModel,
        options: &RunOptions,
    ) -> TractResult<Arc<dyn Runnable>> {
        let default = Section::default();
        let section = self.section(&model).unwrap_or(&default);
        let runtime = section.runtime.as_deref().unwrap_or(DEFAULT_RUNTIME);
        section.scope.enter(|| {
            let rt =
                runtime_for_name(runtime)?.with_context(|| format!("Unknown runtime {runtime}"))?;
            let runnable: Arc<dyn Runnable> = rt.prepare_with_options(model, options)?.into();
            match TRACT_AUTOBATCH_LANES.get() {
                Some(lanes) => Ok(Arc::new(LanedRunnable::wrap(runnable, lanes)?) as _),
                None => Ok(runnable),
            }
        })
    }
}

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
