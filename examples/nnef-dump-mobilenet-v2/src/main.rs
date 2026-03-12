use anyhow::Result;
use tract::prelude::*;

fn main() -> Result<()> {
    let model = tract::nnef()?.with_tract_core()?.load("mobilenet.nnef.tgz")?.into_runnable()?;

    // open image, resize it and make a Tensor out of it
    let image = image::open("grace_hopper.jpg").unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    let input = tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    });

    // run the model on the input
    let result = model.run([input])?;

    // find and display the max value with its index
    let best =
        result[0].as_slice::<f32>()?.iter().zip(1..).max_by(|a, b| a.0.partial_cmp(b.0).unwrap());
    println!("result: {best:?}");
    Ok(())
}
