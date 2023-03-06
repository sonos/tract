use anyhow::Result;
use nom::branch::alt;
use nom::character::complete::{char, digit1};
use nom::combinator::{all_consuming, opt};
use nom::combinator::{map, map_res};
use nom::error::{ErrorKind, ParseError};
use nom::multi::separated_list1;
use nom::sequence::delimited;
use nom::sequence::tuple;
use nom::AsChar;
use nom::IResult;
use nom::InputTakeAtPosition;
use std::path::Path;
use tract_nnef::internal::*;

/// Loader for JSON resources inside a NNEF archive
#[derive(Debug, Clone, PartialEq)]
pub struct JsonLoader;

impl ResourceLoader for JsonLoader {
    fn name(&self) -> Cow<str> {
        "JsonLoader".into()
    }

    fn try_load(
        &self,
        path: &Path,
        reader: &mut dyn std::io::Read,
        _framework: &tract_nnef::framework::Nnef,
    ) -> TractResult<Option<(String, Arc<dyn Resource>)>> {
        if path.extension().map(|e| e == "json").unwrap_or(false) {
            let value = serde_json::from_reader(reader)
                .with_context(|| anyhow!("Error while parsing JSON"))?;
            Ok(Some((
                tract_nnef::resource::resource_path_to_id(path)?,
                Arc::new(JsonResource(value)),
            )))
        } else {
            Ok(None)
        }
    }
}

/// JSON resource than can be queried while loading a NNEF graph.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonResource(pub serde_json::Value);

impl Resource for JsonResource {
    fn get(&self, key: &str) -> TractResult<Value> {
        let value = JsonPath::parse(key)
            .with_context(|| anyhow!("Error while parsing JSON path: {:?}", key))?
            .search(&self.0)
            .with_context(|| anyhow!("Error while acessing JSON using given path: {:?}", key))?;

        convert_value(value)
            .with_context(|| anyhow!("Error while converting JSON value to NNEF value"))
    }
}

pub fn convert_value(value: &serde_json::Value) -> TractResult<Value> {
    match value {
        serde_json::Value::Bool(b) => Ok(Value::Bool(*b)),
        serde_json::Value::Number(v) => {
            if let Some(v) = v.as_i64() {
                Ok(Value::Dim(TDim::Val(v)))
            } else {
                let v = v.as_f64().ok_or_else(|| {
                    anyhow!("Json number {} could not be cast to floating value", v)
                })?;
                Ok(Value::Scalar(v as f32))
            }
        }
        serde_json::Value::String(s) => Ok(Value::String(s.clone())),
        serde_json::Value::Null => bail!("JSON null value cannot be converted to NNEF value"),
        serde_json::Value::Object(_) => bail!("JSON object cannot be converted to NNEF value"),
        serde_json::Value::Array(values) => {
            let t_values = values.iter().map(convert_value).collect::<Result<Vec<Value>>>()?;
            Ok(Value::Array(t_values))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonComponent {
    Root,
    Field(String),
    Index(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonPath {
    pub components: Vec<JsonComponent>,
}

impl JsonPath {
    pub fn new(components: Vec<JsonComponent>) -> Self {
        Self { components }
    }

    pub fn parse(s: &str) -> Result<Self> {
        let (_, components) = all_consuming(parse_components)(s)
            .map_err(|e| anyhow!("Error while parsing JSON path: {:?}", e))?;

        ensure!(
            components.first() == Some(&JsonComponent::Root),
            "Json path must start with the root symbol '$'. None found in {}",
            s
        );

        Ok(Self::new(components))
    }

    pub fn search<'a>(&self, json: &'a serde_json::Value) -> Result<&'a serde_json::Value> {
        let mut components_iter = self.components.iter();
        ensure!(
            components_iter.next() == Some(&JsonComponent::Root),
            "JSON path must start with root key '$'"
        );

        let value = components_iter
            .try_fold(json, |json, component| match component {
                JsonComponent::Index(idx) => Ok(&json[idx]),
                JsonComponent::Field(field) => Ok(&json[field]),
                JsonComponent::Root => bail!("Unexpected '$'(root) symbol in json path"),
            })
            .with_context(|| anyhow!("Error while accessing JSON with path: {:?}", self))?;
        Ok(value)
    }
}

pub fn json_key<T, E: ParseError<T>>(input: T) -> IResult<T, T, E>
where
    T: InputTakeAtPosition,
    <T as InputTakeAtPosition>::Item: AsChar,
{
    input.split_at_position1_complete(
        |item| {
            let c = item.as_char();
            !(c.is_alphanum() || ['-', '_', '+', '='].contains(&c))
        },
        ErrorKind::Fail,
    )
}

fn parse_components(i: &str) -> IResult<&str, Vec<JsonComponent>> {
    map(
        separated_list1(
            char('.'),
            map(
                tuple((
                    alt((
                        map(char('$'), |_| JsonComponent::Root),
                        map(json_key, |f: &str| JsonComponent::Field(f.to_string())),
                    )),
                    opt(map_res(delimited(char('['), digit1, char(']')), |s: &str| {
                        s.parse().map(JsonComponent::Index)
                    })),
                )),
                |(c, idx)| vec![Some(c), idx].into_iter().flatten().collect::<Vec<_>>(),
            ),
        ),
        |components| components.into_iter().flatten().collect::<Vec<_>>(),
    )(i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use serde_json::json;

    #[test]
    fn test_json_key() -> Result<()> {
        let example: serde_json::Value = json!({
            "name": "John Doe",
            "age": 43usize,
            "phones": [
                "+44 1234567",
                "+44 2345678"
            ],
            "others-info": {
                "address": ["Sonos"],
                "indexes": [1, 2, 3, 4, 5, 6],
                "weights": [[1, 2], [3, 4], [5, 6]]
            }
        });
        let resource = JsonResource(example);
        assert_eq!(resource.get("$.name")?, Value::String("John Doe".into()));
        assert_eq!(resource.get("$.age")?, Value::Dim(TDim::Val(43)));
        assert_eq!(resource.get("$.phones[0]")?, Value::String("+44 1234567".into()));
        assert_eq!(resource.get("$.others-info.address[0]")?, Value::String("Sonos".into()));
        assert_eq!(
            resource.get("$.others-info.indexes")?,
            Value::Array(vec![
                Value::Dim(TDim::Val(1)),
                Value::Dim(TDim::Val(2)),
                Value::Dim(TDim::Val(3)),
                Value::Dim(TDim::Val(4)),
                Value::Dim(TDim::Val(5)),
                Value::Dim(TDim::Val(6))
            ])
        );
        assert_eq!(
            resource.get("$.others-info.weights")?,
            Value::Array(vec![
                Value::Array(vec![Value::Dim(TDim::Val(1)), Value::Dim(TDim::Val(2)),]),
                Value::Array(vec![Value::Dim(TDim::Val(3)), Value::Dim(TDim::Val(4)),]),
                Value::Array(vec![Value::Dim(TDim::Val(5)), Value::Dim(TDim::Val(6))]),
            ])
        );
        Ok(())
    }
}
