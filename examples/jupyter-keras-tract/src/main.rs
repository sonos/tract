use rand::*;
use tract_tensorflow::prelude::*;

fn main() -> TractResult<()> {
    let model = tensorflow()
        .model_for_path("./my_model.pb")? // load the model
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec![10, 100]))? // specify input type and shape
        .into_optimized()? // optimize graph
        .into_runnable()?; // make the model runnable and fix its inputs and outputs

    // Generate some input data for the model
    let mut rng = thread_rng();
    let vals: Vec<_> = (0..1000).map(|_| rng.gen::<f32>()).collect();
    let input = tract_ndarray::arr1(&vals).into_shape((10, 100)).unwrap();

    // Input the generated data into the model
    let result = model.run(tvec![input.into()]).unwrap();
    let to_show = result[0].to_array_view::<f32>()?;
    println!("result: {:?}", to_show);
    Ok(())
}
