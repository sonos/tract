use anyhow::{Error, Result};
use image::{DynamicImage, imageops};
use ndarray::{Array1, Array3, Array4, Axis};
use tract::prelude::*;

use crate::{Ndarray, Tract};

pub struct ArcFace {
    model: Runnable,
}

impl ArcFace {
    /// Returns an ndarray of length 512 representing a face.
    /// Please crop the face before using!
    pub fn get_face_embedding(&self, input_image: &DynamicImage) -> Result<Array1<f32>> {
        let preprocess_image = preprocess_arcface(input_image, 112)?;
        let forward = self.model.run([preprocess_image.tract()?])?;
        println!("FORWARD {forward:?}");
        let results = forward[0].ndarray::<f32>()?.to_shape(512)?.to_owned();
        Ok(results)
    }
}

pub fn load_arcface_model(model_path: &str, input_size: i32) -> ArcFace {
    let mut model = tract::onnx().unwrap().load(model_path).unwrap();
    let spec = format!("1,3,{input_size},{input_size},f32");
    model.set_input_fact(0, spec.as_str()).unwrap();
    let model = model.into_model().unwrap().into_runnable().unwrap();
    ArcFace { model }
}

fn image_to_ndarray(img: &DynamicImage) -> Array3<f32> {
    let height = img.height();
    let width = img.width();
    let img_buffer = img.to_rgb8();
    Array3::from_shape_vec((height as usize, width as usize, 3), img_buffer.into_raw())
        .expect("cannot convert image to ndarray")
        .mapv(|x| x as f32)
}

pub fn preprocess_arcface(
    input_image: &DynamicImage,
    target_size: u32,
) -> Result<Array4<f32>, Error> {
    let resize = input_image.resize_exact(target_size, target_size, imageops::FilterType::Triangle);
    let ndarray_img = image_to_ndarray(&resize);
    // (H, W, 3) → (3, H, W) → (1, 3, H, W)
    let permuted = ndarray_img.permuted_axes((2, 0, 1));
    Ok(permuted.insert_axis(Axis(0)))
}

pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dotprod = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    println!("dotprod {dotprod:?} norm_a {norm_a:?} norm_b {norm_b:?}");
    dotprod / (norm_a * norm_b)
}
