mod yolo_face;
use tract_onnx::prelude::*;
use tract_ndarray::s;
use yolo_face::{load_yolo_model, YoloFace};
use clap::Parser;
use anyhow::{Error, Result};
use image::DynamicImage;


#[derive(Parser)]
struct CliArgs {
    #[arg(long)]
    input_image: String,
    
    #[arg(long)]
    weights: String,
}


fn main() -> Result<(), Error>{
    let test_image = image::open("/home/hbdesk/Downloads/cwigl2.jpeg")?;
    let yolo_model: YoloFace = load_yolo_model("/home/hbdesk/face_api_opencv/face_api/models/yolov8n-face.onnx", (640,640));
    let test_bbox = yolo_model.get_faces_bbox(&test_image)?; 
    println!("TEST BBOX: {:?}", test_bbox);
    Ok(())
}
