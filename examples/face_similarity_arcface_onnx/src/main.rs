mod yolo_face;
mod arc_face;
use arc_face::{load_arcface_model, ArcFace};
use yolo_face::{load_yolo_model, YoloFace, sort_conf_bbox};
use clap::Parser;
use anyhow::{Error, Result};
use semanticsimilarity_rs::{cosine_similarity, euclidean_distance};

#[derive(Parser)]
struct CliArgs {
    #[arg(long)]
    face1: String,

    #[arg(long)]
    face2: String,
}

fn main() -> Result<(), Error>{
    let args = CliArgs::parse();
    println!("face1 {:#?}, \n face2 {:?}", &args.face1, &args.face2);
    let face1 = image::open(&args.face1)?;
    let face2 = image::open(&args.face2)?;
    
    let yolo_model: YoloFace = load_yolo_model("/home/hbdesk/face_api_opencv/face_api/models/yolov8n-face.onnx", (640,640));
    let arcface_model: ArcFace = load_arcface_model("/home/hbdesk/face_api_opencv/face_api/models/arcfaceresnet100-8.onnx", 112);
    
    let mut face1_bbox = yolo_model.get_faces_bbox(&face1, 0.5,0.5)?;
    let mut face2_bbox = yolo_model.get_faces_bbox(&face2, 0.5, 0.5)?;


    let f1_sorted_bbox = sort_conf_bbox(&mut face1_bbox);
    let f2_sorted_bbox = sort_conf_bbox(&mut face2_bbox);
    
    println!("face1 {:?}, \n face2 {:?}", f1_sorted_bbox, f2_sorted_bbox);
    
    let f1_crop = f1_sorted_bbox[0].crop_bbox(&face1)?;
    let f2_crop = f2_sorted_bbox[0].crop_bbox(&face2)?;

    let f1_embed = arcface_model.get_face_embedding(&f1_crop)?;
    let f2_embed = arcface_model.get_face_embedding(&f2_crop)?;

    println!("F1 {:?}, \n\n\n\n F2 {:?}", f1_embed, f2_embed);
    let f1_f64: Vec<f64> = {
        f1_embed.into_raw_vec().iter().map(|x| x.to_owned() as f64).collect()
    };
    let f2_f64: Vec<f64> = {
        f2_embed.into_raw_vec().iter().map(|x| x.to_owned() as f64).collect()
    };
    let similarity = cosine_similarity(&f1_f64, &f2_f64,false);
    println!("SIMILARITY {:#?}", similarity);
    Ok(())
}

