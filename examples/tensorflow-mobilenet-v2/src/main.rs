use tract_core::ndarray;
use tract_core::prelude::*;
use tract_core::infer::*;

fn main() -> TractResult<()> {
    // load the model
    let mut model =
        tract_tensorflow::tensorflow().model_for_path("mobilenet_v2_1.4_224_frozen.pb")?;

    // specify input type and shape
    model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 224, 224, 3)))?;

    // optimize the model and get an execution plan
    let model = model.into_optimized()?;
    let plan = SimplePlan::new(&model)?;

    // open image, resize it and make a Tensor out of it
    let image = image::open("grace_hopper.jpg").unwrap().to_rgb();
    let resized = image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    let image: Tensor = ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    })
    .into();

    // run the plan on the input
    let result = plan.run(tvec!(image))?;

    // find and display the max value with its index
    let best = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .zip(1..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("result: {:?}", best);
    Ok(())
}
