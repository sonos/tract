use tract_ndarray::Array;
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let model = tract_onnx::onnx()
        .model_for_path("resnet.onnx")? // load the model
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 224, 224)))? // specify input type and shape
        .into_optimized()? // optimize the model
        .into_runnable()?; // make the model runnable and fix its inputs and outputs

    // Imagenet mean and standard deviation
    let mean = Array::from_shape_vec((1, 3, 1, 1), vec![0.485, 0.456, 0.406])?;
    let std = Array::from_shape_vec((1, 3, 1, 1), vec![0.229, 0.224, 0.225])?;

    let img = image::open("elephants.jpg").unwrap().to_rgb();
    let resized = image::imageops::resize(&img, 224, 224, ::image::imageops::FilterType::Triangle);
    let image: Tensor =
        ((tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
            resized[(x as _, y as _)][c] as f32 / 255.0
        }) - mean)
            / std)
            .into();

    let result = model.run(tvec!(image))?;

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
