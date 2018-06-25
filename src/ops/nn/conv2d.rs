use std::collections::HashMap;
use std::marker::PhantomData;

use super::local_patch::*;
use ops::{Buffer, BufferItem, Attr, Op, TensorView};
use analyser::helpers::infer_forward_concrete;
use analyser::{ShapeFact, TensorFact};
use ndarray::prelude::*;
use ndarray::{Axis, stack};
use tensor::Datum;
use Result;

#[derive(Debug, Clone, new)]
pub struct Conv2D<T: Datum>(LocalPatch, PhantomData<T>);

pub fn conv2d(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let dtype = pb.get_attr_datatype("T")?;
    let patch = LocalPatch::build(pb)?;
    Ok(boxed_new!(Conv2D(dtype)(patch)))
}

impl<T: Datum> Op for Conv2D<T> {
    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        // TODO(liautaud): Implement serialization for LocalPatch.
        hashmap!{
            "T" => Attr::DataType(T::datatype()),
        }
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: Vec<TensorView>) -> Result<Vec<TensorView>> {
        let (m_data, m_filter) = args_2!(inputs);
        let data = T::tensor_into_array(m_data.into_tensor())?;
        let filter = T::tensor_to_view(&*m_filter)?;
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

        let mut transformed: Vec<T> = Vec::with_capacity(images.n() * out_height * out_width * out_depth);

        // Loop over each batch.
        for image in data.outer_iter() {
            let patches = self.0.mk_patches(image, (filter_rows, filter_cols))?;
            transformed.extend(patches.dot(&filter).into_iter());
        }

        let transformed = Array::from_vec(transformed)
            .into_shape((images.n(), out_height, out_width, out_depth))?
            .into_dyn();
        Ok(vec![T::array_into_tensor(transformed).into()])
    }

    /// Evaluates one step of the operation on the given input tensors.
    fn step(
        &self,
        mut inputs: Vec<(Option<usize>, Option<TensorView>)>,
        buffer: &mut Buffer,
    ) -> Result<Option<Vec<TensorView>>> {
        // We only support the VALID padding strategy for now, with the
        // streaming dimension being either the width or the height.

        // The idea is that, regardless of the strides, we need at least
        // as many chunks in the buffer as the size of the filter in the
        // streaming dimension to compute our first output chunk. Then,
        // we pop the min(buffer_size, k) first chunks from the buffer,
        // ignore the next max(k - buffer_size, 0) chunks, and wait for
        // the k following chunks to compte one output chunk, with k the
        // strides in the streaming dimension.

        let (mut data, mut filter) = args_2!(inputs);

        if self.0.padding != Padding::Valid {
            bail!("Conv2D only supports VALID padding when streaming.");
        }

        if filter.0.is_some() || filter.1.is_none() {
            bail!("Filter input should not be streamed.");
        }

        if data.0.is_none()  {
            bail!("Data input should be streamed.");
        }

        // Maybe there is no incoming chunk.
        if data.1.is_none() {
            return Ok(None);
        }

        // Maybe the data is streamed along the batch dimension.
        let dim = data.0.unwrap();
        if dim == 0 {
            let result = self.eval(vec![
                data.1.take().unwrap(),
                filter.1.take().unwrap()
            ])?;

            return Ok(Some(result))
        }

        if dim < 1 || dim > 2 {
            bail!("Conv2D only supports batch, width and height streaming.");
        }

        let data = data.1.take().unwrap().into_tensor();
        let data = into_4d(T::tensor_into_array(data)?)?;
        let data_size = data.shape()[dim];
        debug_assert!(data_size == 1);

        let filter = T::tensor_into_array(filter.1.take().unwrap().into_tensor())?;
        let filter_size = filter.shape()[dim - 1];

        // The buffer contains both `skip` (the number of chunks to skip) and
        // `prev` (the chunks from previous iterations which are still needed
        // to compute the next convolution).
        let empty_view = || {
            let array = match dim {
                1 => Array::zeros((data.shape()[0], 0, data.shape()[2], data.shape()[3])),
                2 => Array::zeros((data.shape()[0], data.shape()[1], 0, data.shape()[3])),
                _ => panic!()
            };

            T::array_into_tensor(array.into_dyn()).into()
        };

        buffer.initialize(|| vec![
            BufferItem::Usize(0),
            BufferItem::View(empty_view())
        ])?;

        let skip = buffer.take_usize(0)?;
        if skip > 0 {
            buffer.set_usize(0, skip - 1)?;
            return Ok(None)
        }

        let prev = buffer.take_view(1)?.into_tensor();
        let prev = into_4d(T::tensor_into_array(prev)?)?;
        let next = stack(Axis(dim), &[prev.view(), data.view()])?;
        let next_size = next.shape()[dim];

        // Maybe we don't have enough chunks to compute the convolution yet.
        if next_size < filter_size {
            buffer.set_usize(0, skip - 1)?;
            buffer.set_view(1, T::array_into_tensor(next.into_dyn()).into())?;
            return Ok(None)
        }

        // Otherwise we compute the convolution using the non-streaming implementation.
        // FIXME(liautaud): THERE SHOULDN'T BE A CLONE HERE!
        let next_view = T::array_into_tensor(next.clone().into_dyn()).into();
        let filter_view = T::array_into_tensor(filter).into();
        let result = self.eval(vec![next_view, filter_view])?;

        let stride = [self.0.v_stride, self.0.h_stride][dim - 1];

        if stride > next_size {
            // Maybe we must pop more chunks from the buffer than it currently contains.
            buffer.set_usize(0, stride - next_size)?;
            buffer.set_view(1, empty_view())?;
        } else {
            // Otherwise we pop the right number of chunks to prepare the next iteration.
            let next = match dim {
                1 => next.slice_move(s![.., stride..next_size, .., ..]),
                2 => next.slice_move(s![.., .., stride..next_size, ..]),
                _ => bail!("Conv2D only supports batch, width and height streaming.")
            };

            buffer.set_usize(0, 0)?;
            buffer.set_view(1, T::array_into_tensor(next.into_dyn()).into())?;
        }

        Ok(Some(result))
    }

    /// Infers properties about the output tensors from the input tensors.
    fn infer_forward(&self, inputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        use analyser::DimFact::*;

        if inputs.len() != 2 {
            bail!("Conv2D operation only supports two inputs.");
        }

        if let Some(output) = infer_forward_concrete(self, &inputs)? {
            return Ok(Some(output));
        }

        let shape = match (inputs[0].shape.dims.as_slice(), inputs[1].shape.dims.as_slice()) {
            ([batch, in_height, in_width, in_channels],
             [filter_height, filter_width, in_channels_2, out_channels]) => {
                if let (&Only(ic1), &Only(ic2)) = (in_channels, in_channels_2) {
                    if ic1 != ic2 {
                        bail!("The in_channels parameters of the input and filter must be equal.");
                    }
                }

                let (height, width) = match (in_height, in_width, filter_height, filter_width) {
                    (&Only(ih), &Only(iw), &Only(fh), &Only(fw)) => {
                        let (h, w) = self.0.adjusted_dim(ih, iw, (fh, fw));
                        (Only(h), Only(w))
                    },

                    _ => (Any, Any)
                };

                // TODO(liautaud): Take the data_format parameter into account.
                ShapeFact::closed(vec![*batch, height, width, *out_channels])
            }

            _ if inputs[0].shape.open || inputs[1].shape.open => shapefact![_, _, _, _],
            _ => bail!("The input and filter dimensions are invalid."),
        };

        let output = TensorFact {
            datatype: inputs[0].datatype,
            shape,
            value: valuefact!(_),
        };

        Ok(Some(vec![output]))
    }

    /// Infers properties about the input tensors from the output tensors.
    fn infer_backward(&self, outputs: Vec<&TensorFact>) -> Result<Option<Vec<TensorFact>>> {
        use analyser::DimFact::*;

        if outputs.len() < 1 {
            bail!("Conv2D operation only supports one output.");
        }

        let (input_shape, filter_shape) = match outputs[0].shape.dims.as_slice() {
            [batch, _, _, out_channels] =>
                (ShapeFact::closed(vec![*batch, Any, Any, Any]),
                 ShapeFact::closed(vec![Any, Any, Any, *out_channels])),

            _ if outputs[0].shape.open =>
                (shapefact![_, _, _, _], shapefact![_, _, _, _]),

            _ => bail!("The output dimensions are invalid."),
        };

        let input = TensorFact {
            datatype: outputs[0].datatype,
            shape: input_shape,
            value: valuefact!(_),
        };

        let filter = TensorFact {
            datatype: outputs[0].datatype,
            shape: filter_shape,
            value: valuefact!(_),
        };

        Ok(Some(vec![input, filter]))
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
