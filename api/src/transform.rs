use std::collections::HashMap;

use crate::DatumType;

/// A serialized transform specification passed to `ModelInterface::transform`.
///
/// Wraps the string representation expected by the transform registry.
/// Constructed from raw strings or typed config structs implementing [`TransformConfig`].
#[derive(Debug, Clone)]
pub struct TransformSpec(String);

impl TransformSpec {
    /// Produce the string the transform registry expects.
    pub fn to_transform_string(&self) -> String {
        self.0.clone()
    }
}

impl From<&str> for TransformSpec {
    fn from(s: &str) -> Self {
        TransformSpec(s.to_string())
    }
}

impl From<String> for TransformSpec {
    fn from(s: String) -> Self {
        TransformSpec(s)
    }
}

impl From<&String> for TransformSpec {
    fn from(s: &String) -> Self {
        TransformSpec(s.clone())
    }
}

/// Trait for typed transform configurations.
///
/// Implementors derive [`serde::Serialize`] and provide a transform [`name()`](TransformConfig::name).
/// The default [`to_transform_string()`](TransformConfig::to_transform_string) serializes the
/// struct as a JSON object and injects the `"name"` key.
pub trait TransformConfig: serde::Serialize {
    /// The transform registry name (e.g. `"pulse"`, `"float_precision"`).
    fn name(&self) -> &'static str;

    /// Produce the string the transform registry expects.
    ///
    /// The default implementation serializes `self` to a JSON object and inserts `"name"`.
    fn to_transform_string(&self) -> String {
        let mut obj: serde_json::Map<String, serde_json::Value> = serde_json::to_value(self)
            .expect("TransformConfig serialization cannot fail")
            .as_object()
            .expect("TransformConfig must serialize to a JSON object")
            .clone();
        obj.insert("name".into(), serde_json::Value::String(self.name().to_string()));
        serde_json::to_string(&obj).expect("serialization cannot fail")
    }
}

/// Implements [`TransformConfig`] and `From<$ty> for TransformSpec`.
macro_rules! transform_config {
    ($ty:ty, $name:expr) => {
        impl TransformConfig for $ty {
            fn name(&self) -> &'static str {
                $name
            }
        }

        impl From<$ty> for TransformSpec {
            fn from(config: $ty) -> Self {
                TransformSpec(config.to_transform_string())
            }
        }
    };
}

/// Typed config for the `concretize_symbols` transform.
///
/// Replaces symbolic dimensions with concrete integer values.
///
/// # Example
/// ```ignore
/// model.transform(ConcretizeSymbols::new().value("B", 1))?;
/// ```
#[derive(Debug, Clone, Default, serde::Serialize)]
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

transform_config!(ConcretizeSymbols, "concretize_symbols");

/// Typed config for the `pulse` transform.
///
/// Converts a model to a pulsed (streaming) model.
///
/// # Example
/// ```ignore
/// model.transform(Pulse::new("5").symbol("B"))?;
/// ```
#[derive(Debug, Clone, serde::Serialize)]
pub struct Pulse {
    pulse: String,
    #[serde(skip_serializing_if = "Option::is_none")]
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

transform_config!(Pulse, "pulse");

/// Typed config for the `float_precision` transform.
///
/// Changes the float precision of a model (e.g. F32 to F16).
///
/// # Example
/// ```ignore
/// use tract_api::DatumType;
/// model.transform(FloatPrecision::new(DatumType::TRACT_DATUM_TYPE_F32, DatumType::TRACT_DATUM_TYPE_F16))?;
/// ```
#[derive(Debug, Clone, serde::Serialize)]
pub struct FloatPrecision {
    from: String,
    to: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exclude: Option<Vec<String>>,
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

impl FloatPrecision {
    pub fn new(from: DatumType, to: DatumType) -> Self {
        Self {
            from: datum_type_to_str(from).to_string(),
            to: datum_type_to_str(to).to_string(),
            include: None,
            exclude: None,
        }
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

transform_config!(FloatPrecision, "float_precision");
