use ::{Matrix, Result};
use super::Op;

pub struct ExpandDims;

impl Op for ExpandDims {
    fn eval(&self, inputs:Vec<Matrix>) -> Result<Vec<Matrix>> {
        let data = inputs[0].as_f32s().ok_or("Expect input #0 to be f32")?;
        let dims = inputs[1].as_i32s().ok_or("Expect input #1 to be i32")?;
        let mut shape = data.shape().to_vec();
        for d in dims {
            if *d >= 0 {
                shape.insert(*d as usize, 1);
            } else {
                Err(format!("unimplemented ExpandDims with negative parameter"))?
            }
        }
        Ok(vec!(Matrix::F32(data.clone().into_shape(shape)?)))
    }
}

