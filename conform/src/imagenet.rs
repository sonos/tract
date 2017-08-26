use std::{fs, path};

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
    let resp = reqwest::get(url).map_err(|e| format!("reqwest error: {}", e.description()))?;
    if resp.status() != reqwest::StatusCode::Ok {
        Err("Could not download inception v3")?
    }
    let mut archive = ::tar::Archive::new(::flate2::read::GzDecoder::new(resp)?);
    archive.unpack(dir)?;
    Ok(())
}



#[test]
fn test() {
    download().unwrap();
    /*
    let tf = ::tf::build("data/inception-2015-12-05/classify_image_graph_def.pb").unwrap();
    let mut tfd = ::tfd::build("data/inception-2015-12-05/classify_image_graph_def.pb").unwrap();
    let mut g = ::tfdeploy::GraphAnalyser::from_file("data/inception-2015-12-05/classify_image_graph_def.pb").unwrap();
    let result = g.eval("DecodeJpeg");
    println!("result: {:?}", result);
    */

    let mut image_buffer = vec![];
    use std::io::Read;
    ::std::fs::File::open("data/inception-v3-2016_08_28/cropped_panda.jpg")
        .unwrap()
        .read_to_end(&mut image_buffer)
        .unwrap();
    let input = ::tfdeploy::ops::image::decode_one(&*image_buffer).unwrap();

    let input = ::ndarray::Array4::from_shape_fn(
        (1, input.shape()[0], input.shape()[1], 3),
        |(_, x, y, c)| input[(x, y, c)] as f32,
    ).into_dyn();

    ::compare_all(
        INCEPTION_V3,
        vec![("input", input.into())],
        "InceptionV3/Predictions/Reshape_1",
    ).unwrap();
}
