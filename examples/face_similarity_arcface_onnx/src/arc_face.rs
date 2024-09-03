use image::{imageops, DynamicImage};
use tract_core::plan::SimplePlan;
use tract_onnx::prelude::*;
use anyhow::{Result, Error};
use tract_ndarray::{ArrayBase, OwnedRepr};
use tract_ndarray::Array1;

pub struct ArcFace {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}


impl ArcFace {
    // returns a vec size of [1, 512] representing a face
    pub fn get_face_embedding(&self, input_image: &DynamicImage) -> Result<ArrayBase<OwnedRepr<f32>, tract_core::ndarray::Dim<[usize; 1]>>, Error> {
        let preprocess_image = preprocess_arcface(input_image, 112);
        let forward = self.model.run(tvec![preprocess_image.to_owned().into()])?; 
        let results =  forward[0].to_array_view::<f32>()?.to_shape((512))?.to_owned();
        Ok(results)
    }
}

pub fn load_arcface_model(model_path: &str, input_size: i32) -> ArcFace {
    let load_model = tract_onnx::onnx().model_for_path(model_path).unwrap()
        .with_input_fact(0, f32::fact([1,3,input_size, input_size]).into()).unwrap()
        .into_optimized().unwrap()
        .into_runnable().unwrap();
    ArcFace {
        model: load_model
    }
}


pub fn preprocess_arcface(
    input_image: &DynamicImage,
    target_size: u32 
) -> Tensor {
    let resized = image::imageops::resize(&input_image.to_rgb8(), target_size, target_size, image::imageops::FilterType::Triangle);
    let mut ndarray_image = image::RgbImage::new(target_size , target_size);
    image::imageops::replace(&mut ndarray_image, &resized, target_size as i64,target_size as i64);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, target_size as usize, target_size as usize), |(_, c, y, x)| {
            let mut _a = ndarray_image.get_pixel(x as u32, y as u32)[c] as f32; 
            _a -= 127.5;
            _a *= 1.0 / 128.0;
            _a
    }).into();
    image
}

pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dotprod = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    dotprod / (norm_a * norm_b)
}
