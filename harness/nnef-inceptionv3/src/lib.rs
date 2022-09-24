use std::{fs, io, path};

use tract_core::prelude::*;

fn download() {
    use std::sync::Once;
    static START: Once = Once::new();

    START.call_once(|| do_download().unwrap());
}

fn do_download() -> TractResult<()> {
    let run = ::std::process::Command::new("./download.sh").status().unwrap();
    if !run.success() {
        tract_core::internal::bail!("Failed to download inception model files")
    }
    Ok(())
}

pub fn load_labels() -> Vec<String> {
    use std::io::BufRead;
    io::BufReader::new(fs::File::open(imagenet_slim_labels()).unwrap())
        .lines()
        .collect::<::std::io::Result<Vec<String>>>()
        .unwrap()
}

fn inception_v3_2016_08_28() -> path::PathBuf {
    ::std::env::var("CACHEDIR").ok().unwrap_or_else(|| "../../.cached".to_string()).into()
}

pub fn inception_v3_tgz() -> path::PathBuf {
    download();
    inception_v3_2016_08_28().join("inception_v3.tfpb.nnef.tgz")
}

pub fn imagenet_slim_labels() -> path::PathBuf {
    download();
    inception_v3_2016_08_28().join("imagenet_slim_labels.txt")
}

pub fn load_image<P: AsRef<path::Path>>(p: P) -> Tensor {
    let image = image::open(&p).unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&image, 299, 299, ::image::imageops::FilterType::Triangle);
    tract_ndarray::Array4::from_shape_fn((1, 3, 299, 299), |(_, c, y, x)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    })
    .into_dyn()
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dinghy_test::test_project_path;
    use std::path;

    const HOPPER: &str = "grace_hopper.jpg";
    pub fn hopper() -> path::PathBuf {
        test_project_path().join(HOPPER)
    }

    #[allow(dead_code)]
    pub fn setup_test_logger() {
        env_logger::Builder::from_default_env().filter_level(log::LevelFilter::Trace).init();
    }

    #[test]
    fn grace_hopper_is_a_military_uniform() -> TractResult<()> {
        download();
        // setup_test_logger();
        let nnef = tract_nnef::nnef();
        let mut tar = flate2::read::GzDecoder::new(fs::File::open(inception_v3_tgz())?);
        let model = nnef.model_for_read(&mut tar)?.into_optimized()?.into_runnable()?;
        let input = load_image(hopper());
        let outputs = model.run(tvec![input.into()]).unwrap();
        let labels = load_labels();
        let label_id = outputs[0]
            .to_array_view::<f32>()
            .unwrap()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(0u32.cmp(&1)))
            .unwrap()
            .0;
        let label = &labels[label_id];
        assert_eq!(label, "military uniform");
        Ok(())
    }
}
