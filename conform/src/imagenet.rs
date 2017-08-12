use std::{fs, path};

use reqwest;

use errors::*;

fn download() -> Result<()> {
    let dir = "data/inception-2015-12-05";
    if path::PathBuf::from(dir).join("cropped_panda.jpg").exists() {
        return Ok(());
    }
    fs::create_dir_all(dir)?;
    let url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz";
    let resp = reqwest::get(url)?;
    if resp.status() != reqwest::StatusCode::Ok {
        Err("Could not download inception")?
    }
    let mut archive = ::tar::Archive::new(::flate2::read::GzDecoder::new(resp)?);
    archive.unpack(dir)?;
    Ok(())
}



#[test]
fn test() {
    download().unwrap();
    let tf = ::tf::build("data/inception-2015-12-05/classify_image_graph_def.pb").unwrap();
    let mut tfd = ::tfd::build("data/inception-2015-12-05/classify_image_graph_def.pb").unwrap();
    /*
    let mut g = ::tfdeploy::GraphAnalyser::from_file("data/inception-2015-12-05/classify_image_graph_def.pb").unwrap();
    let result = g.eval("DecodeJpeg");
    println!("result: {:?}", result);
    */
    ::compare(
        "data/inception-2015-12-05/classify_image_graph_def.pb",
        vec![],
        "DecodeJpeg",
    ).unwrap();
}
