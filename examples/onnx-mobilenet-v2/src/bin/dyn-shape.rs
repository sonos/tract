use tract_onnx::prelude::*;
use tract_onnx::tract_core::dims;

fn main() -> TractResult<()> {
    // load the model
    let model = tract_onnx::onnx().model_for_path("mobilenetv2-7.onnx")?;

    // Create a symbol for the batch dimension and define a replacement input_fact
    let batch = model.sym("N");
    let input_fact = f32::fact(dims!(batch, 3, 224, 224));

    let model = model.with_input_fact(0, input_fact.into())?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;

    // open image, resize it and make a Tensor out of it
    let image = image::open("grace_hopper.jpg").unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);

    for n in [1, 3, 5] {
        println!("batch of {n} images:");

        // we put the saùe image n times to simulate a batch of size pictures
        let images: Tensor =
            tract_ndarray::Array4::from_shape_fn((n, 3, 224, 224), |(_, c, y, x)| {
                let mean = [0.485, 0.456, 0.406][c];
                let std = [0.229, 0.224, 0.225][c];
                (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
            })
            .into();
        // run the model on the input
        let results = model.run(tvec!(images.into()))?;

        // loop over the batch
        for image in results[0].to_array_view::<f32>()?.outer_iter() {
            // find and display the max value with its index
            let best = image.iter().zip(2..).max_by(|a, b| a.0.partial_cmp(b.0).unwrap());
            println!("  result: {best:?}");
        }
    }
    Ok(())
}
