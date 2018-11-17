use ndarray::prelude::*;

use super::{Input, Op};
use {Tensor, Result};

#[derive(Debug)]
pub struct DecodeJpeg {}

impl DecodeJpeg {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> TractResult<DecodeJpeg> {
        Ok(DecodeJpeg {})
    }
}

pub fn decode_one(input: &[u8]) -> TractResult<Array3<u8>> {
    use image::GenericImage;
    let image = ::image::load_from_memory(input)?;
    let dim = image.dimensions();
    Ok(Array1::from_vec(image.raw_pixels()).into_shape((dim.0 as usize, dim.1 as usize, 3))?)
}

impl Op for DecodeJpeg {
    fn eval(&self, mut inputs: Vec<Input>) -> TractResult<Vec<Input>> {
        let m_input = args_1!(inputs);
        let input = m_input.as_u8s().ok_or("Expected a string")?;
        let image = decode_one(input.as_slice().unwrap())?;
        Ok(vec![Tensor::U8(image.into_dyn()).into()])
    }
}

#[derive(Debug)]
pub struct ResizeBilinear {}

impl ResizeBilinear {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> TractResult<ResizeBilinear> {
        Ok(ResizeBilinear {})
    }
}

impl Op for ResizeBilinear {
    fn eval(&self, mut inputs: Vec<Input>) -> TractResult<Vec<Input>> {
        use std::cmp::min;
        let (m_images, m_sizes) = args_2!(inputs);
        let images = m_images
            .into_tensor()
            .take_f32s()
            .ok_or("Expect input #0 to be images")?;
        let sizes = m_sizes.as_i32s().ok_or("Expect input #1 to be sizes")?;
        let batches = images.shape()[0];
        let old_height = images.shape()[1];
        let old_width = images.shape()[2];
        let channels = images.shape()[3];
        let images = images.into_shape((batches, old_height, old_width, channels))?;
        let new_height = sizes[0] as usize;
        let new_width = sizes[1] as usize;
        let new_shape = (batches, new_height, new_width, channels);
        let result = Array4::from_shape_fn(new_shape, |(b, y, x, c)| {
            let proj_x = old_width as f32 * x as f32 / new_width as f32;
            let proj_y = old_height as f32 * y as f32 / new_width as f32;
            let old_x = proj_x as usize;
            let old_y = proj_y as usize;
            let q11 = images[(b, old_y, old_x, c)];
            let q12 = images[(b, min(old_y + 1, old_height - 1), old_x, c)];
            let q21 = images[(b, old_y, min(old_x + 1, old_width - 1), c)];
            let q22 = images[(
                b,
                min(old_y + 1, old_height - 1),
                min(old_x + 1, old_width - 1),
                c,
            )];
            let dx = proj_x - old_x as f32;
            let dy = proj_y - old_y as f32;
            (1.0 - dy) * ((1.0 - dx) * q11 + dx * q21) + dy * ((1.0 - dx) * q12 + dx * q22)
        });
        Ok(vec![Tensor::F32(result.into_dyn()).into()])
    }
}
