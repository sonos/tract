use crate::internal::*;

pub(crate) fn process_file(
    input: &str,
    resources: &HashMap<String, Arc<dyn Resource>>,
) -> TractResult<String> {
    let env = minijinja::Environment::new();
    let tmpl = env.template_from_str(input)?;
    let mut globals = serde_json::Map::new();
    for (k, v) in resources {
        if let Some(value) = v.to_template_value() {
            globals.insert(k.clone(), value);
        }
    }
    Ok(tmpl.render(serde_json::Value::Object(globals))?)
}
