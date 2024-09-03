mod yolo_face;
mod arc_face;
use arc_face::{load_arcface_model, cosine_similarity};
use yolo_face::{load_yolo_model, YoloFace, sort_conf_bbox};
use clap::Parser;
use anyhow::{Error, Result};

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
    let arcface_model = load_arcface_model("/home/hbdesk/face_api_opencv/face_api/models/arcfaceresnet100-8.onnx", 112);
    
    let mut test_bbox= yolo_model.get_faces_bbox(&test_image, 0.5, 0.5)?; 
    let sorted_bbox = sort_conf_bbox(&mut test_bbox);
    
    let crop_face = sorted_bbox[0].crop_bbox(&test_image)?;
    let embedding = arcface_model.get_face_embedding(&crop_face)?;
    let similarity = cosine_similarity(&embedding, &embedding);
    println!("SIMILARITY {:?}", similarity);
    Ok(())
}

