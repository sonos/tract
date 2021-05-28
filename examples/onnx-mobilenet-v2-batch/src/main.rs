use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let batch = Symbol::new('N');
    let model = tract_onnx::onnx()
        // load the model
        .model_for_path("mobilenetv2-7.onnx")?
        // specify input type and shape
        .with_input_fact(
            0,
            InferenceFact::dt_shape(
                DatumType::F32,
                &[batch.to_dim(), 3usize.into(), 224usize.into(), 224usize.into()],
                ),
                )?
        // this model hardcodes a "1" as batch output shape, erase the output shape
        // to let tract perform inference and find "N"
        .with_output_fact(0, InferenceFact::default())?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;

    // open image, resize it and make a Tensor out of it
    let mut images = vec![];
    for img in &["grace_hopper.jpg", "cat.jpg"] {
        let image = image::open(img).unwrap().to_rgb8();
        let resized =
            image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);
        images.push(resized);
    }
    let n = images.len();
    let input = tract_ndarray::Array4::from_shape_fn((n, 3, 224, 224), |(n, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (images[n][(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    });

    // run the model on the input
    let result = model.run(tvec!(input.into_tensor()))?;

    // result is an array of shape [batch_size, category_count]
    let best: tract_ndarray::ArrayView2<f32> =
        result[0].to_array_view::<f32>()?.into_dimensionality()?;
    for (ix, b) in best.outer_iter().enumerate() {
        // find the best category
        let best = b.iter().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        println!("batch index:{} best score {} for category {:?}", ix, best.1, best.0 + 2);
    }
    Ok(())
}
