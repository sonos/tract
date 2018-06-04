use std::marker::PhantomData;

use super::local_patch::*;
use super::{Op, TensorView};
use analyser::helpers::infer_forward_concrete;
use analyser::{ShapeFact, TensorFact};
use ndarray::prelude::*;
use tensor::Datum;
use Result;

#[derive(Debug, new)]
pub struct Conv2D<T: Datum>(LocalPatch, PhantomData<T>);

pub fn conv2d(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let dtype = pb.get_attr_datatype("T")?;
    let patch = LocalPatch::build(pb)?;
    Ok(boxed_new!(Conv2D(dtype)(patch)))
}

impl<T: Datum> Op for Conv2D<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        let (m_data, m_filter) = args_2!(inputs);
        let data = T::mat_into_array(m_data.into_tensor())?;
        let filter = T::mat_to_view(&*m_filter)?;
        let data = into_4d(data)?;
        let images = BatchImageWrapper(data.view());

        let filter_rows = filter.shape()[0];
        let filter_cols = filter.shape()[1];
        let out_depth = filter.shape()[3];

        let (out_height, out_width) =
            self.0
                .adjusted_dim(images.h(), images.w(), (filter_rows, filter_cols));

        let filter = filter
            .view()
            .into_shape((filter_rows * filter_cols * images.d(), out_depth))?;

        let mut transformed: Vec<T> = Vec::with_capacity(out_height * out_width * out_depth);
        for image in data.outer_iter() {
            let patches = self.0.mk_patches(image, (filter_rows, filter_cols))?;
            transformed.extend(patches.dot(&filter).into_iter());
        }
        let transformed = Array::from_vec(transformed)
            .into_shape((images.n(), out_height, out_width, out_depth))?
            .into_dyn();
        Ok(vec![T::array_into_tensor(transformed).into()])
    }

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if inputs.len() != 2 {
            bail!("Conv2D operation only supports two inputs.");
        }

        if let Some(output) = infer_forward_concrete(self, &inputs)? {
            return Ok(Some(output));
        }

        // If we don't know the actual value, we can still compute the shape.
        fn try_infer_forward_concrete_shape<T>(
            op: &Conv2D<T>,
            inputs: Vec<&TensorFact>,
        ) -> Result<Option<ShapeFact>>
        where
            T: Datum,
        {
            let input_shape = unwrap_or_none!(inputs[0].shape.concretize());
            let filter_shape = unwrap_or_none!(inputs[1].shape.concretize());

            let shape = match (input_shape.as_slice(), filter_shape.as_slice()) {
                (
                    [batch, in_height, in_width, in_channels],
                    [filter_height, filter_width, in_channels_2, out_channels],
                ) if in_channels == in_channels_2 =>
                {
                    let (height, width) =
                        op.0
                            .adjusted_dim(*in_height, *in_width, (*filter_height, *filter_width));

                    // TODO(liautaud): Take the data_format parameter into account.
                    shapefact![(*batch), height, width, (*out_channels)]
                }

                _ => bail!("The input and filter dimensions are invalid."),
            };

            Ok(Some(shape))
        };

        let output = TensorFact {
            datatype: inputs[0].datatype,
            shape: try_infer_forward_concrete_shape(self, inputs)?.unwrap_or(shapefact![
                _,
                _,
                _,
                _
            ]),
            value: valuefact!(_),
        };

        Ok(Some(vec![output]))
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        if outputs.len() < 1 {
            bail!("Conv2D operation only supports one output.");
        }

        match outputs[0].shape.concretize() {
            Some(shape) => match shape.as_slice() {
                [batch, _, _, out_channels] => {
                    let input = TensorFact {
                        datatype: outputs[0].datatype,
                        shape: shapefact![(*batch), _, _, _],
                        value: valuefact!(_),
                    };

                    let filter = TensorFact {
                        datatype: outputs[0].datatype,
                        shape: shapefact![_, _, _, (*out_channels)],
                        value: valuefact!(_),
                    };

                    Ok(Some(vec![input, filter]))
                }

                _ => bail!("The output dimensions are invalid."),
            },

            // If we don't have concrete dimensions yet, we can still
            // give the shape that we want.
            None => {
                let input = TensorFact {
                    datatype: outputs[0].datatype,
                    shape: shapefact![_, _, _, _],
                    value: valuefact!(_),
                };

                let filter = TensorFact {
                    datatype: outputs[0].datatype,
                    shape: shapefact![_, _, _, _],
                    value: valuefact!(_),
                };

                Ok(Some(vec![input, filter]))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use Tensor;

    fn mk(sizes: &[usize]) -> Tensor {
        ::ndarray::Array::range(1f32, sizes.iter().product::<usize>() as f32 + 1.0, 1.0)
            .into_shape(sizes)
            .unwrap()
            .into()
    }

    fn verify(input: &[usize], filter: &[usize], stride: usize, padding: Padding, expect: &[f32]) {
        let result = Conv2D::<f32>::new(LocalPatch {
            padding: padding,
            h_stride: stride,
            v_stride: stride,
            _data_format: DataFormat::NHWC,
        }).eval(vec![mk(input).into(), mk(filter).into()])
            .unwrap()
            .remove(0);
        assert_eq!(expect.len(), result.shape().iter().product::<usize>());
        let found = result
            .into_tensor()
            .take_f32s()
            .unwrap()
            .into_shape(expect.len())
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

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn testConv2D1x2Filter() {
        verify(&[1, 2, 3, 3], &[1, 2, 3, 3] , 1, Padding::Valid, &[
        231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0,
        936.0, 1029.0
    ])}

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn testConv2D2x1Filter() {
        verify(&[1, 2, 3, 3], &[2, 1, 3, 3] , 1, Padding::Valid,
          &[465.0, 504.0, 543.0, 618.0, 675.0, 732.0, 771.0, 846.0, 921.0]);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn testConv2D2x2Filter() {
        verify(&[1, 2, 3, 3], &[2, 2, 3, 3] , 1, Padding::Valid,
               &[ 2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0 ])
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn testConv2D2x2FilterStride2() {
        verify(&[1, 2, 3, 3], &[2, 2, 3, 3] , 2, Padding::Valid,
               &[2271.0, 2367.0, 2463.0])
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn testConv2D2x2FilterStride2Same() {
        verify(&[1, 2, 3, 3], &[2, 2, 3, 3] , 2, Padding::Same,
               &[2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]);
    }

    #[test]
    fn test_conv_1() {
        let conv = Conv2D::<f32>::new(LocalPatch {
            padding: Padding::Same,
            h_stride: 1,
            v_stride: 1,
            _data_format: DataFormat::NHWC,
        });
        // NHWC
        let data: Tensor = Tensor::f32s(&[1, 1, 1, 1], &[1f32]).unwrap();
        // HWIO
        let filter = Tensor::f32s(&[3, 1, 1, 1], &[0.0, 1.0, 0.0]).unwrap();
        let exp: Tensor = Tensor::f32s(&[1, 1, 1, 1], &[1.0]).unwrap();

        let result = conv.eval(vec![data.into(), filter.into()])
            .unwrap()
            .remove(0);
        assert_eq!(exp, result.into_tensor());
    }

    #[test]
    fn test_conv_2() {
        let conv = Conv2D::<f32>::new(LocalPatch {
            padding: Padding::Same,
            h_stride: 1,
            v_stride: 1,
            _data_format: DataFormat::NHWC,
        });
        let data =
            Tensor::f32s(&[1, 2, 2, 1], &[142.3088, 48.891083, 208.3187, -11.274994]).unwrap();
        let filter: Tensor = Tensor::f32s(
            &[2, 2, 1, 1],
            &[160.72833, 107.84076, 247.50552, -38.738464],
        ).unwrap();
        let exp: Tensor =
            Tensor::f32s(&[1, 2, 2, 1], &[80142.31, 5067.5586, 32266.81, -1812.2109]).unwrap();

        assert!(exp.close_enough(&conv.eval(vec![data.into(), filter.into()]).unwrap()[0]))
    }
}
