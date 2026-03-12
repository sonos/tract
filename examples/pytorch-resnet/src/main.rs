use anyhow::Result;
use tract_ndarray::Array;
use tract_rs::prelude::*;

fn main() -> Result<()> {
    let mut model = tract_rs::onnx()?.load("resnet.onnx")?;
    model.set_input_fact(0, "1,3,224,224,f32")?;
    let model = model.into_tract()?.into_runnable()?;

    // Imagenet mean and standard deviation
    let mean = Array::from_shape_vec((1, 3, 1, 1), vec![0.485, 0.456, 0.406])?;
    let std = Array::from_shape_vec((1, 3, 1, 1), vec![0.229, 0.224, 0.225])?;

    let img = image::open("elephants.jpg").unwrap().to_rgb8();
    let resized = image::imageops::resize(&img, 224, 224, ::image::imageops::FilterType::Triangle);
    let input = (tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    }) - mean)
        / std;

    let result = model.run([input])?;

    // find and display the max value with its index
    let best =
        result[0].as_slice::<f32>()?.iter().zip(1..).max_by(|a, b| a.0.partial_cmp(b.0).unwrap());
    println!("result: {best:?}");
    Ok(())
}
