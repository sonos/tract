use anyhow::Result;
use tract::prelude::*;

tract::impl_ndarray_interop!();

fn main() -> Result<()> {
    // load the model
    let mut model = tract::onnx()?.load("mobilenetv2-7.onnx")?;

    // Create a symbolic batch dimension
    model.set_input_fact(0, "N,3,224,224,f32")?;

    let model = model.into_model()?.into_runnable()?;

    // open image, resize it and make a Tensor out of it
    let image = image::open("grace_hopper.jpg").unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);

    for n in [1, 3, 5] {
        println!("batch of {n} images:");

        // we put the same image n times to simulate a batch of size pictures
        let images = ndarray::Array4::from_shape_fn((n, 3, 224, 224), |(_, c, y, x)| {
            let mean = [0.485, 0.456, 0.406][c];
            let std = [0.229, 0.224, 0.225][c];
            (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
        });
        // run the model on the input
        let results = model.run([images.tract()?])?;

        // loop over the batch
        for image in results[0].ndarray::<f32>()?.outer_iter() {
            // find and display the max value with its index
            let best = image.iter().zip(2..).max_by(|a, b| a.0.partial_cmp(b.0).unwrap());
            println!("  result: {best:?}");
        }
    }
    Ok(())
}
