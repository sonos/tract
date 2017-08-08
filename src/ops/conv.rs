use {Matrix, Result};
use super::Op;

pub enum DataFormat {
    NHWC,
}

#[derive(Debug)]
pub enum Padding {
    Valid,
    Same,
}

pub struct Conv2D {
    pub _data_format: DataFormat,
    pub padding: Padding,
    pub strides: Vec<usize>,
}

impl Conv2D {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Conv2D> {
        let data_format = pb.get_attr().get("data_format").ok_or(
            "expect data_format in Conv2D args",
        )?;
        if data_format.get_s() == b"NCHW" {
            Err("NCHW data_format not implemented")?
        }
        let strides = pb.get_attr()
            .get("strides")
            .ok_or("expect strides in Conv2D args")?
            .get_list()
            .get_i()
            .iter()
            .map(|a| *a as usize)
            .collect();
        let padding = pb.get_attr().get("padding").ok_or(
            "expect padding in Conv2D args",
        )?;
        let padding = match padding.get_s() {
            b"VALID" => Padding::Valid,
            b"SAME" => Padding::Same,
            _ => Err("Only VALID padding supported for now on Conv2D")?,
        };
        Ok(Conv2D {
            _data_format: DataFormat::NHWC,
            padding,
            strides,
        })
    }
}

impl Op for Conv2D {
    fn eval(&self, mut inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        // [ filter_rows, filter_cols, in_depth, out_depth]
        let filter = inputs.remove(1).take_f32s().ok_or("Expect input #1 to be f32")?;
        // [ batch, in_rows, in_cols, in_depth ]
        let data = inputs.remove(0).take_f32s().ok_or("Expect input #0 to be f32")?;
        /*
        println!("data shape: {:?}", data.shape());
        println!("filter shape: {:?}", filter.shape());
        println!("strides: {:?}", self.strides);
        println!("data:\n{:?}", data);
        println!("filter:\n{:?}", filter);
        */
        if self.strides.len() != 4 || self.strides[0] != 1 && self.strides[3] != 1 ||
            self.strides[1] != self.strides[2]
        {
            Err(format!(
                "strides must be of the form [1, s, s, 1], found {:?}",
                self.strides
            ))?
        }
        if data.shape().len() != 4 || filter.shape().len() != 4 {
            Err(format!(
                "data and filter must be of dimension 4. data is {:?}, filter is {:?}",
                data.shape(),
                filter.shape()
            ))?
        }
        if data.shape()[3] != filter.shape()[2] {
            Err(format!(
                "data fourth dim (in_depth) must match filter third (data is {:?}, filter is {:?})",
                data.shape(),
                filter.shape()
            ))?
        }
        let batches = data.shape()[0];
        let in_rows = data.shape()[1];
        let in_cols = data.shape()[2];
        let in_depth = data.shape()[3];
        let filter_rows = filter.shape()[0];
        let filter_cols = filter.shape()[1];
        let out_depth = filter.shape()[3];

        let data = data.into_shape((batches, in_rows, in_cols, in_depth))?;
        let filter = filter.into_shape((filter_rows, filter_cols, in_depth, out_depth))?;

        let (out_height, out_width) = match self.padding {
            Padding::Same => (
                (in_rows as f32 / self.strides[1] as f32) as usize,
                (in_cols as f32 / self.strides[2] as f32) as usize,
            ),
            Padding::Valid => (
                ((in_rows - filter_rows + 1) as f32 / self.strides[1] as f32) as usize,
                ((in_cols - filter_cols + 1) as f32 / self.strides[2] as f32) as usize,
            ),
        };
        let out_shape = ::ndarray::IxDyn(
            &[
                data.shape()[0],
                out_height,
                out_width,
                out_depth,
            ],
        );
        // prepare local patches
        let patches_size = ((out_height*out_width) as usize, filter_rows*filter_cols*in_depth);
        unsafe {
            let mut patches = ::ndarray::Array2::<f32>::uninitialized(patches_size);
            for i_x in 0..out_width {
                for i_y in 0..out_height {
                    let mut patch_row = patches.row_mut(i_y*out_width+i_x);
                    for f_x in 0..filter_cols {
                        for f_y in 0..filter_rows {
                            for d in 0..in_depth {
                                patch_row[f_y*in_depth*filter_cols+f_x*in_depth+d] = data[(0, i_y+f_y, i_x+f_x, d)];
                            }
                        }
                    }
                }
            }
            let mut filters_mat = ::ndarray::Array2::<f32>::uninitialized((patches_size.1, out_depth));
            for f_x in 0..filter_cols {
                for f_y in 0..filter_rows {
                    for d in 0..in_depth {
                        let mut filter_row = filters_mat.row_mut(f_y*in_depth*filter_cols+f_x*in_depth+d);
                        for od in 0..out_depth {
                            filter_row[od] = filter[(f_y, f_x, d, od)];
                        }
                    }
                }
            }
            let result = patches.dot(&filters_mat).into_shape(out_shape.clone())?;
            return Ok(vec![Matrix::F32(result)])
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use Matrix;
    use super::*;

    fn mk(sizes: &[usize]) -> Matrix {
        let data = ::ndarray::Array::range(1f32, sizes.iter().product::<usize>() as f32 + 1.0, 1.0)
            .into_shape(sizes)
            .unwrap();
        Matrix::F32(data)
    }

    fn verify(input: &[usize], filter: &[usize], stride: usize, padding: Padding, expect: &[f32]) {
        let strides = vec![1, stride, stride, 1];
        let result = Conv2D {
            padding: padding,
            strides: strides,
            _data_format: DataFormat::NHWC,
        }.eval(vec![mk(input), mk(filter)])
            .unwrap()
            .remove(0);
        assert_eq!(expect.len(), result.shape().iter().product::<usize>());
        let found = result
            .take_f32s()
            .unwrap()
            .into_shape((expect.len()))
            .unwrap();
        assert_eq!(expect, found.as_slice().unwrap());
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn testConv2D1x1Filter() {
        verify(&[1,2,3,3], &[1, 1, 3, 3], 1, Padding::Valid, &[
        30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0,
        204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0 ]);
    }
}
