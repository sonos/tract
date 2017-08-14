use ndarray::prelude::*;

use {Matrix, Result};
use super::Op;

#[derive(Debug)]
pub struct DecodeJpeg {}

impl DecodeJpeg {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<DecodeJpeg> {
        Ok(DecodeJpeg {})
    }
}

fn decode_one(input: &[u8]) -> Result<Array3<u8>> {
    use image::GenericImage;
    let image = ::image::load_from_memory(input)?;
    let dim = image.dimensions();
    Ok(Array1::from_vec(image.raw_pixels()).into_shape((
        dim.0 as usize,
        dim.1 as usize,
        3,
    ))?)
}

impl Op for DecodeJpeg {
    fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        let input: ArrayD<u8> = inputs.remove(0).take_u8s().ok_or(
            "Expect input #0 to be buffers",
        )?;
        let image = decode_one(input.as_slice().unwrap())?;
        Ok(vec![Matrix::U8(image.into_dyn())])
    }
}
