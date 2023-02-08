use tract_nnef::internal::*;
use tract_nnef_resources::internal::JsonLoader;

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