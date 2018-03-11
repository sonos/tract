extern crate dinghy_test;
extern crate flate2;
extern crate image;
extern crate itertools;
extern crate ndarray;
extern crate mio_httpc;
extern crate tar;
extern crate tfdeploy;

use std::{fs, io, path};
use mio_httpc::SyncCall;
use dinghy_test::test_project_path;

use tfdeploy::errors::*;

pub const HOPPER: &str = "examples/grace_hopper.jpg";

fn download() {
    use std::sync::{Once, ONCE_INIT};
    static START: Once = ONCE_INIT;

    START.call_once(|| {
        do_download().unwrap()
    });
}

fn do_download() -> Result<()> {
    let dir = inception_v3_2016_08_28();
    let dir_partial = dir.clone().with_extension("partial");
    if fs::metadata(&dir).is_ok() {
        return Ok(());
    }
    println!("Downloading inception_v3 model...");
    if fs::metadata(&dir_partial).is_ok() {
        fs::remove_dir_all(&dir_partial).unwrap();
    }
    fs::create_dir_all(&dir_partial)?;
    let url = "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz";
    let (status, _hdrs, body) = SyncCall::new().timeout_ms(5000).get(url).map_err(|e| {
        format!("request error: {:?}", e)
    })?;
    if status != 200 {
        Err("Could not download inception v3")?
    }
    let mut archive = ::tar::Archive::new(::flate2::read::GzDecoder::new(&body[..]));
    archive.unpack(&dir_partial)?;
    fs::rename(dir_partial, dir)?;
    Ok(())
}

pub fn load_labels() -> Vec<String> {
    use std::io::BufRead;
    io::BufReader::new(
        fs::File::open(imagenet_slim_labels()).unwrap()
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

fn inception_v3_2016_08_28() -> path::PathBuf {
    ::std::env::temp_dir().join("inception-v3-2016_08_28")
}

pub fn inception_v3_2016_08_28_frozen() -> path::PathBuf {
    download();
    inception_v3_2016_08_28().join("inception_v3_2016_08_28_frozen.pb")
}

pub fn imagenet_slim_labels() -> path::PathBuf {
    download();
    inception_v3_2016_08_28().join("imagenet_slim_labels.txt")
}

#[allow(dead_code)]
fn main() {
    download();
    let tfd = ::tfdeploy::for_path(inception_v3_2016_08_28_frozen()).unwrap();
    let input_id = tfd.node_id_by_name("input").unwrap();
    let output_id = tfd.node_id_by_name("InceptionV3/Predictions/Reshape_1").unwrap();
    let input = load_image(test_project_path().join(HOPPER));
    let output = tfd.run(vec![(input_id, input)], output_id).unwrap();
    let labels = load_labels();
    for (ix, output) in output[0].as_f32s().unwrap().iter().enumerate() {
        if *output > 0.4 {
            println!("{:0.05} {}", output, labels[ix]);
        }
    }
}
