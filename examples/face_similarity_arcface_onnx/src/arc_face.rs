use anyhow::{Error, Result};
use image::{imageops, DynamicImage};
use tract_core::plan::SimplePlan;
use tract_ndarray::{Array1, Array3};
use tract_ndarray::{ArrayBase, OwnedRepr};
use tract_onnx::prelude::*;

#[allow(clippy::type_complexity)]
pub struct ArcFace {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl ArcFace {
    /// returns a ndarray of length 512 representing a face
    /// please crop face before using !
    pub fn get_face_embedding(
        &self,
        input_image: &DynamicImage,
    ) -> Result<ArrayBase<OwnedRepr<f32>, tract_core::ndarray::Dim<[usize; 1]>>, Error> {
        let preprocess_image = preprocess_arcface(input_image, 112)?;
        let forward = self.model.run(tvec![preprocess_image.to_owned().into()])?;
        println!("FORWARD {forward:?}");
        let results = forward[0].to_array_view::<f32>()?.to_shape(512)?.to_owned();
        Ok(results)
    }
}

pub fn load_arcface_model(model_path: &str, input_size: i32) -> ArcFace {
    let load_model = tract_onnx::onnx()
        .model_for_path(model_path)
        .unwrap()
        .with_input_fact(0, f32::fact([1, 3, input_size, input_size]).into())
        .unwrap()
        .incorporate()
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();
    ArcFace { model: load_model }
}

fn image_to_tract_tensor(img: &DynamicImage) -> Array3<f32> {
    let height = img.height();
    let width = img.width();
    let img_buffer = img.to_rgb8();
    Array3::from_shape_vec((height as usize, width as usize, 3), img_buffer.into_raw())
        .expect("cannot convert image to ndarray")
        .mapv(|x| x as f32)
}

pub fn preprocess_arcface(input_image: &DynamicImage, target_size: u32) -> Result<Tensor, Error> {
    let resize = input_image.resize_exact(target_size, target_size, imageops::FilterType::Triangle);
    let ndarray_img = image_to_tract_tensor(&resize);
    // ndarray_img *= 1.0 / 127.5;
    // ndarray_img -= 127.5;
    let mut _final: Tensor = ndarray_img.permuted_axes((2, 0, 1)).into();
    _final.insert_axis(0).unwrap();
    Ok(_final)
}

pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dotprod = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    println!("dotprod {dotprod:?} norm_a {norm_a:?} norm_b {norm_b:?}");
    dotprod / (norm_a * norm_b)
}
