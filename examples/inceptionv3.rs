extern crate flate2;
extern crate image;
extern crate itertools;
extern crate ndarray;
extern crate reqwest;
extern crate tar;
extern crate tfdeploy;

use std::{fs, io, path};

use tfdeploy::errors::*;

use std::error::Error;

pub const INCEPTION_V3: &str = "examples/data/inception-v3-2016_08_28/inception_v3_2016_08_28_frozen.pb";
pub const HOPPER: &str = "examples/grace_hopper.jpg";

pub fn download() -> Result<()> {
    if fs::metadata(INCEPTION_V3).is_ok() {
        return Ok(());
    }
    let dir = "examples/data/inception-v3-2016_08_28";
    fs::create_dir_all(dir)?;
    let url = "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz";
    let resp = reqwest::get(url).map_err(|e| {
        format!("reqwest error: {}", e.description())
    })?;
    if resp.status() != reqwest::StatusCode::Ok {
        Err("Could not download inception v3")?
    }
    let mut archive = ::tar::Archive::new(::flate2::read::GzDecoder::new(resp)?);
    archive.unpack(dir)?;
    Ok(())
}

pub fn load_labels() -> Vec<String> {
    use std::io::BufRead;
    io::BufReader::new(
        fs::File::open(
            "examples/data/inception-v3-2016_08_28/imagenet_slim_labels.txt",
        ).unwrap(),
    ).lines()
        .collect::<::std::io::Result<Vec<String>>>()
        .unwrap()
}

pub fn load_image<P: AsRef<path::Path>>(p: P) -> ::tfdeploy::Matrix {
    let image = ::image::open(&p).unwrap().to_rgb();
    let resized = ::image::imageops::resize(&image, 299, 299, ::image::FilterType::Triangle);
    let image: ::tfdeploy::Matrix = ::ndarray::Array4::from_shape_fn(
        (1, 299, 299, 3),
        |(_, y, x, c)| resized[(x as _, y as _)][c] as f32 / 255.0,
    ).into_dyn()
        .into();
    image
}

#[allow(dead_code)]
fn main() {
    download().unwrap();
    let mut tfd = ::tfdeploy::for_path(INCEPTION_V3).unwrap();
    let input = load_image(HOPPER);
    let output = tfd.run(vec![("input", input)], "InceptionV3/Predictions/Reshape_1")
        .unwrap();
    let labels = load_labels();
    for (ix, output) in output[0].as_f32s().unwrap().iter().enumerate() {
        if *output > 0.4 {
            println!("{:0.05} {}", output, labels[ix]);
        }
    }
}
