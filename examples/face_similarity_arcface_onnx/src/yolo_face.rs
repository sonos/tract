use anyhow::{Error, Result};
use image::DynamicImage;
use std::cmp::Ordering;
use std::cmp::PartialOrd;
use tract_core::plan::SimplePlan;
use tract_ndarray::s;
use tract_onnx::prelude::*;

#[allow(clippy::type_complexity)]
pub struct YoloFace {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    width: i32,
    height: i32,
}

pub fn sort_conf_bbox(input_bbox: &mut [Bbox]) -> Vec<Bbox> {
    input_bbox.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    input_bbox.to_vec()
}

impl YoloFace {
    pub fn get_faces_bbox(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<Bbox>, Error> {
        // assuming that model has input shape of a square
        let preprocess_image = preprocess_yoloface_square(input_image, self.width as f32);

        // run forward pass and then convert result to f32
        let forward = self.model.run(tvec![preprocess_image.to_owned().into()])?;
        let results = forward[0].to_array_view::<f32>()?.view().t().into_owned();

        // process results
        let mut bbox_vec: Vec<Bbox> = vec![];
        for i in 0..results.len_of(tract_ndarray::Axis(0)) {
            let row = results.slice(s![i, .., ..]);
            let confidence = row[[4, 0]];

            if confidence >= confidence_threshold {
                let x = row[[0, 0]];
                let y = row[[1, 0]];
                let w = row[[2, 0]];
                let h = row[[3, 0]];
                let x1 = x - w / 2.0;
                let y1 = y - h / 2.0;
                let x2 = x + w / 2.0;
                let y2 = y + h / 2.0;
                let bbox = Bbox::new(x1, y1, x2, y2, confidence).apply_image_scale(
                    input_image,
                    self.width as f32,
                    self.height as f32,
                );
                bbox_vec.push(bbox);
            }
        }
        Ok(non_maximum_suppression(bbox_vec, iou_threshold))
    }
}

#[derive(Debug, Clone)]
pub struct Bbox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
}

impl Bbox {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, confidence: f32) -> Bbox {
        Bbox { x1, y1, x2, y2, confidence }
    }
    pub fn apply_image_scale(
        &mut self,
        original_image: &DynamicImage,
        x_scale: f32,
        y_scale: f32,
    ) -> Bbox {
        let normalized_x1 = self.x1 / x_scale;
        let normalized_x2 = self.x2 / x_scale;
        let normalized_y1 = self.y1 / y_scale;
        let normalized_y2 = self.y2 / y_scale;

        let cart_x1 = original_image.width() as f32 * normalized_x1;
        let cart_x2 = original_image.width() as f32 * normalized_x2;
        let cart_y1 = original_image.height() as f32 * normalized_y1;
        let cart_y2 = original_image.height() as f32 * normalized_y2;

        Bbox { x1: cart_x1, y1: cart_y1, x2: cart_x2, y2: cart_y2, confidence: self.confidence }
    }

    pub fn crop_bbox(&self, original_image: &DynamicImage) -> Result<DynamicImage, Error> {
        let bbox_width = (self.x2 - self.x1) as u32;
        let bbox_height = (self.y2 - self.y1) as u32;
        Ok(original_image.to_owned().crop_imm(
            self.x1 as u32,
            self.y1 as u32,
            bbox_width,
            bbox_height,
        ))
    }
}

/// loads model, panic on failure.
pub fn load_yolo_model(model_path: &str, input_size: (i32, i32)) -> YoloFace {
    let load_model = tract_onnx::onnx()
        .model_for_path(model_path)
        .unwrap()
        .with_input_fact(0, f32::fact([1, 3, input_size.0, input_size.1]).into())
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();
    YoloFace { model: load_model, width: input_size.0, height: input_size.1 }
}

fn non_maximum_suppression(mut boxes: Vec<Bbox>, iou_threshold: f32) -> Vec<Bbox> {
    boxes.sort_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(Ordering::Equal));
    let mut keep = Vec::new();
    while !boxes.is_empty() {
        let current = boxes.remove(0);
        keep.push(current.clone());
        boxes.retain(|box_| calculate_iou(&current, box_) <= iou_threshold);
    }
    keep
}

fn calculate_iou(box1: &Bbox, box2: &Bbox) -> f32 {
    let x1 = box1.x1.max(box2.x1);
    let y1 = box1.y1.max(box2.y1);
    let x2 = box1.x2.min(box2.x2);
    let y2 = box1.y2.min(box2.y2);

    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    let area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    let union = area1 + area2 - intersection;
    intersection / union
}

/// scales the image to target dims with black padding.
fn preprocess_yoloface_square(input_image: &DynamicImage, target_size: f32) -> Tensor {
    let width = input_image.width();
    let height = input_image.height();
    let scale = target_size / (width.max(height) as f32);
    let new_width = (width as f32 * scale) as u32;
    let new_height = (height as f32 * scale) as u32;
    let resized = image::imageops::resize(
        &input_image.to_rgb8(),
        new_width,
        new_height,
        image::imageops::FilterType::Triangle,
    );
    let mut padded = image::RgbImage::new(target_size as u32, target_size as u32);
    image::imageops::replace(
        &mut padded,
        &resized,
        (target_size as u32 - new_width) as i64 / 2,
        (target_size as u32 - new_height) as i64 / 2,
    );
    let image: Tensor = tract_ndarray::Array4::from_shape_fn(
        (1, 3, target_size as usize, target_size as usize),
        |(_, c, y, x)| padded.get_pixel(x as u32, y as u32)[c] as f32 / 255.0,
    )
    .into();
    image
}
