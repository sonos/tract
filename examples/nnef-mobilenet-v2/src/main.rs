use tract_nnef::prelude::*;

fn main() -> TractResult<()> {
    let model = tract_nnef::open_model("mobilenet_v2_1.0.onnx.nnef.tgz")?
        .into_typed_model()
        .map_err(|pair| pair.1)?
        .declutter()?
        .optimize()?
        .into_runnable()?;

    // open image, resize it and make a Tensor out of it
    let image = image::open("grace_hopper.jpg").unwrap().to_rgb();
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    })
    .into();

    // run the model on the input
    let result = model.run(tvec!(image))?;

    // find and display the max value with its index
    let best = result[0]
        .as_slice::<f32>()?
        .iter()
        .zip(2..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("result: {:?}", best);
    Ok(())
}
