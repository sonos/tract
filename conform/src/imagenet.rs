use std::{fs, io, path};

use reqwest;

use tfdeploy::errors::*;

use std::error::Error;

const INCEPTION_V3: &str = "data/inception-v3-2016_08_28/inception_v3_2016_08_28_frozen.pb";

fn download() -> Result<()> {
    let dir = "data/inception-v3-2016_08_28";
    if path::PathBuf::from(dir).join("cropped_panda.jpg").exists() {
        return Ok(());
    }
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

fn categs() -> Vec<(String, String)> {
    use std::io::BufRead;
    use itertools::Itertools;
    let buf = io::BufReader::new(fs::File::open("imagenet_labels_1000.txt").unwrap());
    buf.lines()
        .map(|line| {
            let line = line.unwrap();
            let mut tokens = line.split(" ");
            (
                tokens.next().unwrap().to_string(),
                tokens.join(" ").to_string(),
            )
        })
        .collect()
}

fn load_image<P: AsRef<path::Path>>(p: P) -> ::tfdeploy::Matrix {
    let image = ::image::open(&p).unwrap().to_rgb();
    let resized = ::image::imageops::resize(&image, 299, 299, ::image::FilterType::Triangle);
    let image: ::tfdeploy::Matrix = ::ndarray::Array4::from_shape_fn(
        (1, 299, 299, 3),
        |(_, y, x, c)| resized[(x as _, y as _)][c] as f32 / 255.0,
    ).into_dyn()
        .into();
    image
}

#[test]
fn test_tf() {
    let mut tf = ::tfdeploy::tf::for_path(INCEPTION_V3).unwrap();
    let input = load_image("data/inception-v3-2016_08_28/grace_hopper.jpg");
    let mut output = tf.run(vec![("input", input)], "InceptionV3/Predictions/Reshape_1")
        .unwrap();
    let categs = categs();
    for (ix, c) in output.remove(0).take_f32s().unwrap().iter().enumerate() {
        if *c >= 0.01 {
            println!("{}: {} {}", ix, c, categs[ix - 1].1);
        }
    }
    panic!();
}

#[test]
fn test_compare_all() {
    download().unwrap();

    ::compare_all(
        INCEPTION_V3,
        vec![
            (
                "input",
                load_image("data/inception-v3-2016_08_28/grace_hopper.jpg")
            ),
        ],
        "InceptionV3/Predictions/Reshape_1",
    ).unwrap();
}
