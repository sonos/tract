use std::path::Path;
use tract_nnef::internal::*;

pub struct JsonLoader;

impl ResourceLoader for JsonLoader {
    fn name(&self) -> Cow<str> {
        "JsonLoader".into()
    }

    fn try_load(
        &self,
        path: &Path,
        reader: &mut dyn std::io::Read,
    ) -> TractResult<Option<(String, Arc<dyn Resource>)>> {
        if path.extension().map(|e| e == "json").unwrap_or(false) {
            let value = serde_json::from_reader(reader)
                .with_context(|| anyhow!("Error while parsing JSON"))?;
            Ok(Some((tract_nnef::resource::resource_path_to_id(path)?, Arc::new(JsonResource(value)))))
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct JsonResource(serde_json::Value);

impl Resource for JsonResource {
    fn get(&self, key: &str) -> TractResult<Value> {
        let v = self.0.get(key)
            .with_context(|| anyhow!("No value found for key {:?}", key))?
            .as_i64()
            .with_context(|| anyhow!("Value at key {:?} is not an integer", key))?;
        Ok(Value::Scalar(v as f32))
    }
}

#[test]
fn load_model_with_json_resource() -> TractResult<()> {
    let model = tract_nnef::nnef()
            .with_tract_core()
            .with_tract_resource()
            .with_resource_loader(JsonLoader)
            .model_for_path("tests/nnef_with_json")?;

    assert_eq!(model.input_fact(0)?.shape.as_concrete().unwrap(), &vec![2, 10]);
    assert_eq!(model.output_fact(0)?.shape.as_concrete().unwrap(), &vec![2, 10]);
    Ok(())
}