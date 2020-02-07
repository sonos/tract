#[cfg(features = "conform")]
extern crate conform;
extern crate image;
extern crate tract_core;
extern crate tract_tensorflow;

use std::{fs, path};

use tract_core::ndarray;
use tract_core::prelude::*;

fn download() {
    use std::sync::Once;
    static START: Once = std::sync::Once::new();

    START.call_once(|| do_download().unwrap());
}

fn do_download() -> TractResult<()> {
    let run = ::std::process::Command::new("./download.sh").status().unwrap();
    if !run.success() {
        Err("Failed to download model files")?
    }
    Ok(())
}

fn cachedir() -> path::PathBuf {
    ::std::env::var("CACHEDIR").ok().unwrap_or("../../.cached".to_string()).into()
}

pub fn load_labels() -> Vec<String> {
    fs::read_to_string(imagenet_slim_labels()).unwrap().lines().map(|s| s.into()).collect()
}

#[allow(dead_code)]
fn mobilenet_v2() -> path::PathBuf {
    download();
    cachedir().join("mobilenet_v2_1.4_224_frozen.pb")
}

pub fn imagenet_slim_labels() -> path::PathBuf {
    download();
    cachedir().join("imagenet_slim_labels.txt")
}

pub fn grace_hopper() -> path::PathBuf {
    download();
    cachedir().join("grace_hopper.jpg")
}

pub fn load_image<P: AsRef<path::Path>>(p: P) -> Tensor {
    let image = image::open(&p).unwrap().to_rgb();
    let resized = image::imageops::resize(&image, 224, 224, image::FilterType::Triangle);
    let image = ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    })
    .into_dyn()
    .into();
    image
}

#[cfg(test)]
mod tests {
    extern crate dinghy_test;
    use tract_core::ndarray::*;
    use tract_core::prelude::*;
    use tract_core::infer::*;

    use super::*;

    pub fn argmax(input: ArrayViewD<f32>) -> usize {
        input
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(0u32.cmp(&1)))
            .unwrap()
            .0
    }

    #[test]
    fn plain() {
        let tfd = tract_tensorflow::tensorflow().model_for_path(mobilenet_v2()).unwrap();
        let plan = SimplePlan::new(&tfd).unwrap();
        let input = load_image(grace_hopper());
        let outputs = plan.run(tvec![input]).unwrap();
        let labels = load_labels();
        let label_id = argmax(outputs[0].to_array_view::<f32>().unwrap());
        let label = &labels[label_id];
        assert_eq!(label, "military uniform");
    }

    #[test]
    fn optimized() {
        let mut tfd = tract_tensorflow::tensorflow().model_for_path(mobilenet_v2()).unwrap();
        tfd.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), &[1, 224, 224, 3]))
            .unwrap();
        let tfd = tfd.into_optimized().unwrap();
        let plan = SimplePlan::new(&tfd).unwrap();
        let input = load_image(grace_hopper());
        let outputs = plan.run(tvec![input]).unwrap();
        let labels = load_labels();
        let label_id = argmax(outputs[0].to_array_view::<f32>().unwrap());
        let label = &labels[label_id];
        assert_eq!(label, "military uniform");
    }
}
