extern crate image;
extern crate tract_tensorflow;

use std::{fs, io, path};

use tract_tensorflow::prelude::*;
use tract_tensorflow::tract_core::internal::*;

fn download() {
    use std::sync::Once;
    static START: Once = Once::new();

    START.call_once(|| do_download().unwrap());
}

fn do_download() -> TractResult<()> {
    let run = ::std::process::Command::new("./download.sh").status().unwrap();
    if !run.success() {
        bail!("Failed to download inception model files")
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

pub fn inception_v3_2016_08_28_frozen() -> path::PathBuf {
    download();
    inception_v3_2016_08_28().join("inception_v3_2016_08_28_frozen.pb")
}

pub fn imagenet_slim_labels() -> path::PathBuf {
    download();
    inception_v3_2016_08_28().join("imagenet_slim_labels.txt")
}

pub fn load_image<P: AsRef<path::Path>>(p: P) -> TValue {
    let image = image::open(&p).unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&image, 299, 299, ::image::imageops::FilterType::Triangle);
    tract_ndarray::Array4::from_shape_fn((1, 299, 299, 3), |(_, y, x, c)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    })
    .into_dyn()
    .into_tvalue()
}

#[cfg(test)]
mod tests {
    extern crate dinghy_test;
    use tract_tensorflow::prelude::*;

    use self::dinghy_test::test_project_path;
    use super::*;
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
    fn grace_hopper_is_a_military_uniform() {
        download();
        // setup_test_logger();
        println!("{:?}", inception_v3_2016_08_28_frozen());
        let tfd = tensorflow().model_for_path(inception_v3_2016_08_28_frozen()).unwrap();
        let plan = SimplePlan::new(&tfd).unwrap();
        let input = load_image(hopper());
        let outputs = plan.run(tvec![input]).unwrap();
        let labels = load_labels();
        let label_id = outputs[0]
            .to_array_view::<f32>()
            .unwrap()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let label = &labels[label_id];
        assert_eq!(label, "military uniform");
    }
}
