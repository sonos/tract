extern crate image;
extern crate tract_tensorflow;

use std::{fs, path};

use tract_tensorflow::prelude::*;
use tract_tensorflow::tract_core::internal::*;

fn download() {
    use std::sync::Once;
    static START: Once = std::sync::Once::new();

    START.call_once(|| do_download().unwrap());
}

fn do_download() -> TractResult<()> {
    let run = ::std::process::Command::new("./download.sh").status().unwrap();
    if !run.success() {
        bail!("Failed to download model files")
    }
    Ok(())
}

fn cachedir() -> path::PathBuf {
    ::std::env::var("CACHEDIR").ok().unwrap_or_else(|| "../../.cached".to_string()).into()
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

pub fn load_image<P: AsRef<path::Path>>(p: P) -> Tensor {
    let image = image::open(&p).unwrap().to_rgb8();
    let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);
    tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    })
    .into_dyn()
    .into()
}

#[cfg(test)]
mod tests {
    extern crate dinghy_test;
    use tract_tensorflow::prelude::*;

    use super::*;

    fn mobilenet_v2() -> path::PathBuf {
        download();
        cachedir().join("mobilenet_v2_1.4_224_frozen.pb")
    }

    fn run<F, O>(runnable: SimplePlan<F, O, Graph<F, O>>) -> TractResult<()>
    where
        F: Fact + Hash + Clone + 'static,
        O: std::fmt::Debug
            + std::fmt::Display
            + AsRef<dyn Op>
            + AsMut<dyn Op>
            + Clone
            + 'static
    {
        let input = load_image(grace_hopper());
        let outputs = runnable.run(tvec![input.into()])?;
        let label_id = outputs[0]
            .as_slice::<f32>()?
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let labels = load_labels();
        let label = &labels[label_id];
        assert_eq!(label, "military uniform");
        Ok(())
    }

    #[test]
    fn plain() -> TractResult<()> {
        let tfd = tract_tensorflow::tensorflow().model_for_path(mobilenet_v2())?.into_runnable()?;
        run(tfd)
    }

    #[test]
    fn declutter() -> TractResult<()> {
        let tfd = tract_tensorflow::tensorflow()
            .model_for_path(mobilenet_v2())?
            .with_input_fact(0, f32::fact([1, 224, 224, 3]).into())?
            .into_typed()?
            .into_decluttered()?
            .into_runnable()?;
        run(tfd)
    }

    #[test]
    fn optimized() -> TractResult<()> {
        let tfd = tract_tensorflow::tensorflow()
            .model_for_path(mobilenet_v2())?
            .with_input_fact(0, f32::fact([1, 224, 224, 3]).into())?
            .into_optimized()?
            .into_runnable()?;
        run(tfd)
    }
}
