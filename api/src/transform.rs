use std::collections::HashMap;

use crate::DatumType;

/// A transform specification accepted by `ModelInterface::transform`.
///
/// Can be constructed from:
/// - A raw string (plain name like `"f32_to_f16"` or a JSON object string)
/// - A typed config struct like [`ConcretizeSymbols`] or [`Pulse`]
#[derive(Debug, Clone)]
pub enum TransformSpec {
    /// Raw string: plain name ("f32_to_f16") or JSON object.
    Raw(String),
    /// Typed: transform name + JSON params (without "name" key).
    Typed { name: &'static str, params_json: String },
}

impl TransformSpec {
    /// Produce the string the transform registry expects.
    pub fn to_transform_string(&self) -> String {
        match self {
            TransformSpec::Raw(s) => s.clone(),
            TransformSpec::Typed { name, params_json } => {
                if params_json == "{}" {
                    name.to_string()
                } else {
                    let mut obj: serde_json::Map<String, serde_json::Value> =
                        serde_json::from_str(params_json)
                            .expect("params_json must be a valid JSON object");
                    obj.insert("name".into(), serde_json::Value::String(name.to_string()));
                    serde_json::to_string(&obj).expect("serialization cannot fail")
                }
            }
        }
    }
}

impl From<&str> for TransformSpec {
    fn from(s: &str) -> Self {
        TransformSpec::Raw(s.to_string())
    }
}

impl From<String> for TransformSpec {
    fn from(s: String) -> Self {
        TransformSpec::Raw(s)
    }
}

impl From<&String> for TransformSpec {
    fn from(s: &String) -> Self {
        TransformSpec::Raw(s.clone())
    }
}

/// Typed config for the `concretize_symbols` transform.
///
/// Replaces symbolic dimensions with concrete integer values.
///
/// # Example
/// ```ignore
/// model.transform(ConcretizeSymbols::new().value("B", 1))?;
/// ```
#[derive(Debug, Clone, Default)]
pub struct ConcretizeSymbols {
    values: HashMap<String, i64>,
}

impl ConcretizeSymbols {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a symbol to a concrete value.
    pub fn value(mut self, symbol: impl Into<String>, val: i64) -> Self {
        self.values.insert(symbol.into(), val);
        self
    }
}

impl From<ConcretizeSymbols> for TransformSpec {
    fn from(config: ConcretizeSymbols) -> Self {
        let params = serde_json::json!({ "values": config.values });
        TransformSpec::Typed {
            name: "concretize_symbols",
            params_json: serde_json::to_string(&params).expect("serialization cannot fail"),
        }
    }
}

/// Typed config for the `pulse` transform.
///
/// Converts a model to a pulsed (streaming) model.
///
/// # Example
/// ```ignore
/// model.transform(Pulse::new("5").symbol("B"))?;
/// ```
#[derive(Debug, Clone)]
pub struct Pulse {
    pulse: String,
    symbol: Option<String>,
}

impl Pulse {
    /// Create a new Pulse config with the given pulse dimension.
    pub fn new(pulse: impl Into<String>) -> Self {
        Self { pulse: pulse.into(), symbol: None }
    }

    /// Set the symbol to pulse over (defaults to "S" if not set).
    pub fn symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }
}

impl From<Pulse> for TransformSpec {
    fn from(config: Pulse) -> Self {
        let mut params = serde_json::Map::new();
        params.insert("pulse".into(), serde_json::Value::String(config.pulse));
        if let Some(symbol) = config.symbol {
            params.insert("symbol".into(), serde_json::Value::String(symbol));
        }
        TransformSpec::Typed {
            name: "pulse",
            params_json: serde_json::to_string(&params).expect("serialization cannot fail"),
        }
    }
}

/// Typed config for the `float_precision` transform.
///
/// Changes the float precision of a model (e.g. F32 to F16).
///
/// # Example
/// ```ignore
/// use tract_api::DatumType;
/// model.transform(FloatPrecision::new(DatumType::TRACT_DATUM_TYPE_F32, DatumType::TRACT_DATUM_TYPE_F16))?;
/// ```
#[derive(Debug, Clone)]
pub struct FloatPrecision {
    from: DatumType,
    to: DatumType,
    include: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
}

impl FloatPrecision {
    pub fn new(from: DatumType, to: DatumType) -> Self {
        Self { from, to, include: None, exclude: None }
    }

    /// Set include patterns — only nodes matching at least one pattern are translated.
    pub fn include(mut self, patterns: Vec<String>) -> Self {
        self.include = Some(patterns);
        self
    }

    /// Set exclude patterns — matching nodes are excluded from translation.
    pub fn exclude(mut self, patterns: Vec<String>) -> Self {
        self.exclude = Some(patterns);
        self
    }
}

fn datum_type_to_str(dt: DatumType) -> &'static str {
    use DatumType::*;
    match dt {
        TRACT_DATUM_TYPE_F16 => "f16",
        TRACT_DATUM_TYPE_F32 => "f32",
        TRACT_DATUM_TYPE_F64 => "f64",
        _ => panic!("FloatPrecision only supports float datum types (F16, F32, F64)"),
    }
}

impl From<FloatPrecision> for TransformSpec {
    fn from(config: FloatPrecision) -> Self {
        let mut params = serde_json::Map::new();
        params.insert(
            "from".into(),
            serde_json::Value::String(datum_type_to_str(config.from).to_string()),
        );
        params.insert(
            "to".into(),
            serde_json::Value::String(datum_type_to_str(config.to).to_string()),
        );
        if let Some(include) = config.include {
            let arr: Vec<serde_json::Value> =
                include.into_iter().map(serde_json::Value::String).collect();
            params.insert("include".into(), serde_json::Value::Array(arr));
        }
        if let Some(exclude) = config.exclude {
            let arr: Vec<serde_json::Value> =
                exclude.into_iter().map(serde_json::Value::String).collect();
            params.insert("exclude".into(), serde_json::Value::Array(arr));
        }
        TransformSpec::Typed {
            name: "float_precision",
            params_json: serde_json::to_string(&params).expect("serialization cannot fail"),
        }
    }
}
