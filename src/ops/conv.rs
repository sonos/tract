use std::cmp;

use {Matrix, Result};
use super::Op;

enum DataFormat {
    NHWC,
}

#[derive(Debug)]
enum Padding {
    Valid,
    Same,
}

pub struct Conv2D {
    _data_format: DataFormat,
    padding: Padding,
    strides: Vec<usize>,
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
    fn eval(&self, inputs: Vec<Matrix>) -> Result<Vec<Matrix>> {
        let data = inputs[0].as_f32s().ok_or("Expect input #0 to be f32")?;
        let kernel = inputs[1].as_f32s().ok_or("Expect input #1 to be i32")?;
        /*
        println!("input shape: {:?}", data.shape());
        println!("kernel shape: {:?}", kernel.shape());
        println!("strides: {:?}", self.strides);
        */
        if data.shape()[1] != kernel.shape()[0] || data.shape()[2] != kernel.shape()[1] ||
            data.shape()[3] != kernel.shape()[2]
        {
            Err("dimension mismatch between data and kernel for Conv2D")?
        }
        let (out_height, out_width) = match self.padding {
            Padding::Same => (
                (data.shape()[1] as f32 / self.strides[1] as f32) as isize,
                (data.shape()[2] as f32 / self.strides[2] as f32) as isize,
            ),
            Padding::Valid => (
                ((data.shape()[1] - kernel.shape()[0] + 1) as f32 /
                     self.strides[1] as f32) as isize,
                ((data.shape()[2] - kernel.shape()[1] + 1) as f32 /
                     self.strides[2] as f32) as isize,
            ),
        };
        /*
        let pad_along_height = cmp::max(
            0isize,
            ((out_height - 1) * self.strides[1] as isize + kernel.shape()[0] as isize -
                 data.shape()[1] as isize),
        ) as usize;
        let pad_along_width = cmp::max(
            0isize,
            ((out_width - 1) * self.strides[2] as isize + kernel.shape()[1] as isize -
                 data.shape()[2] as isize),
        ) as usize;
        */
        let dims = ::ndarray::IxDyn(
            &[
                data.shape()[0],
                out_height as usize,
                out_width as usize,
                kernel.shape()[3],
            ],
        );
        let strides = &self.strides;
        let result = ::ndarray::ArrayD::from_shape_fn(dims, |dim| {
            use ndarray::Dimension;
            let dim = dim.as_array_view();
            let mut sum = 0.0;
            for di in 0..kernel.shape()[0] {
                for dj in 0..kernel.shape()[1] {
                    for q in 0..kernel.shape()[2] {
                        if self.strides[1] * dim[1] + di < data.shape()[1] && strides[2] * dim[2] + dj < data.shape()[2] {
                            sum += data[[dim[0], strides[1] * dim[1] + di, strides[2] * dim[2] + dj, q]] * kernel[[di, dj, q, dim[3]]]
                        }
                    }
                }
            }
            sum
        });
        Ok(vec![Matrix::F32(result)])
    }
}
