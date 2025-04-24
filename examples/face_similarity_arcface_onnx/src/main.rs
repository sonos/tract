mod arc_face;
mod yolo_face;
use anyhow::{Error, Result};
use arc_face::{cosine_similarity, load_arcface_model, ArcFace};
use clap::Parser;
use yolo_face::{load_yolo_model, sort_conf_bbox, YoloFace};

#[derive(Parser)]
struct CliArgs {
    #[arg(long)]
    face1: String,

    #[arg(long)]
    face2: String,
}

fn main() -> Result<(), Error> {
    let args = CliArgs::parse();
    println!("face1 {:#?},\nface2 {:?}", &args.face1, &args.face2);
    let face1 = image::open(&args.face1)?;
    let face2 = image::open(&args.face2)?;

    let yolo_model: YoloFace = load_yolo_model("yolov8n-face.onnx", (640, 640));
    let arcface_model: ArcFace = load_arcface_model("arcfaceresnet100-8.onnx", 112);

    let mut face1_bbox = yolo_model.get_faces_bbox(&face1, 0.5, 0.5)?;
    let mut face2_bbox = yolo_model.get_faces_bbox(&face2, 0.5, 0.5)?;

    let f1_sorted_bbox = sort_conf_bbox(&mut face1_bbox);
    let f2_sorted_bbox = sort_conf_bbox(&mut face2_bbox);

    let f1_crop = f1_sorted_bbox[0].crop_bbox(&face1)?;
    let f2_crop = f2_sorted_bbox[0].crop_bbox(&face2)?;

    let f1_embed = arcface_model.get_face_embedding(&f1_crop)?;
    let f2_embed = arcface_model.get_face_embedding(&f2_crop)?;

    let similarity = cosine_similarity(&f1_embed, &f2_embed);
    println!("SIMILARITY {similarity:#?}");
    Ok(())
}
