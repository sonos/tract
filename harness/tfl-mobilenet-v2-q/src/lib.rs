use std::{fs, path};

use tract_tflite::internal::*;

fn download() {
    use std::sync::Once;
    static START: Once = std::sync::Once::new();

    START.call_once(|| do_download().unwrap());
}

fn do_download() -> TractResult<()> {
    let run = std::process::Command::new("./download.sh").status().unwrap();
    if !run.success() {
        bail!("Failed to download model files")
    }
    Ok(())
}

fn cachedir() -> path::PathBuf {
    std::env::var("CACHEDIR").ok().unwrap_or_else(|| "../../.cached".to_string()).into()
}

pub fn load_labels() -> Vec<String> {
    fs::read_to_string(imagenet_slim_labels()).unwrap().lines().map(|s| s.into()).collect()
}
pub fn imagenet_slim_labels() -> path::PathBuf {
    download();
    cachedir().join("imagenet_slim_labels.txt")
}

pub fn grace_hopper() -> path::PathBuf {
    download();
    cachedir().join("grace_hopper.jpg")
}

pub fn input_dt() -> DatumType {
    i8::datum_type().with_zp_scale(-1, 0.007843138)
}

pub fn load_image<P: AsRef<path::Path>>(p: P) -> Tensor {
    let image = image::open(&p).unwrap().to_rgb8();
    let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);
    let mut tensor: Tensor =
        tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
            (resized[(x as _, y as _)][c] as i32 - 128) as i8
        })
        .into_dyn()
        .into();
    unsafe { tensor.set_datum_type(input_dt()) };
    tensor
}

#[cfg(test)]
mod tests {
    extern crate dinghy_test;
    use tract_tflite::prelude::*;

    use super::*;

    fn mobilenet_v2() -> path::PathBuf {
        download();
        cachedir().join("mobilenetv2_ptq_single_img.tflite")
    }

    fn run<F, O>(runnable: SimplePlan<F, O, Graph<F, O>>) -> TractResult<()>
    where
        F: Fact + Hash + Clone + 'static,
        O: std::fmt::Debug + std::fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    {
        let input = load_image(grace_hopper());
        let outputs = runnable.run(tvec![input.into()])?;
        let label_id = outputs[0].as_slice::<i8>()?.iter().enumerate().max().unwrap().0;
        let labels = load_labels();
        let label = &labels[label_id];
        assert_eq!(label, "military uniform");
        Ok(())
    }

    #[test]
    #[ignore]
    fn plain() -> TractResult<()> {
        let tfd = tract_tflite::tflite().model_for_path(mobilenet_v2())?.into_runnable()?;
        run(tfd)
    }

    #[test]
    #[ignore]
    fn declutter() -> TractResult<()> {
        let tfd = tract_tflite::tflite()
            .model_for_path(mobilenet_v2())?
            .with_input_fact(0, input_dt().fact([1, 224, 224, 3]))?
            .into_decluttered()?
            .into_runnable()?;
        run(tfd)
    }

    #[test]
    #[ignore]
    fn optimized() -> TractResult<()> {
        let tfd = tract_tflite::tflite()
            .model_for_path(mobilenet_v2())?
            .with_input_fact(0, input_dt().fact([1, 224, 224, 3]))?
            .into_optimized()?
            .into_runnable()?;
        run(tfd)
    }
}
