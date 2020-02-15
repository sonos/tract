use rand::*;
use tract_core::ndarray;
use tract_core::prelude::*;

fn main() -> TractResult<()> {
    // load the model
    let mut model = tract_tensorflow::tensorflow().model_for_path("./my_model.pb")?;

    // specify input type and shape
    model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec![10, 100]))?;
    let model = model.into_optimized()?;

    // Generate some input data for the model
    let mut rng = thread_rng();
    let vals: Vec<_> = (0..1000).map(|_| rng.gen::<f32>()).collect();
    let input = ndarray::arr1(&vals).into_shape((10, 100)).unwrap();

    // Make an execution plan for the model
    let plan = SimplePlan::new(&model).unwrap();

    // Input the generated data into the model
    let result = plan.run(tvec![input.into()]).unwrap();
    let to_show = result[0].to_array_view::<f32>()?;
    println!("result: {:?}", to_show);
    Ok(())
}
