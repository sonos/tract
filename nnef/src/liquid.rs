use crate::internal::*;

pub(crate) fn process_file(
    input: &str,
    resources: &HashMap<String, Arc<dyn Resource>>,
) -> TractResult<String> {
    let parser = liquid::ParserBuilder::with_stdlib().build()?;
    let tmpl = parser.parse(input)?;
    let mut globals = liquid::object!({});
    for (k, v) in resources {
        if let Some(value) = v.to_liquid_value() {
            globals.insert(k.into(), value);
        }
    }
    Ok(tmpl.render(&globals)?)
}
