use std::collections::HashMap;
use std::marker::PhantomData;

use super::local_patch::*;
use analyser::interface::*;
use ndarray::prelude::*;
use ndarray::{stack, Axis, Slice};
use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Conv2D<T: Datum>(LocalPatch, PhantomData<T>);

#[derive(Debug, Clone)]
pub struct Buffer<T: Datum> {
    // The number of future chunks to skip before storing them again.
    skip: usize,

    // An array which stores the previous chunks which are still needed to
    // compute the next convolution.
    prev: Option<Array4<T>>,
}

impl<T: Datum> OpBuffer for Buffer<T> {}

pub fn conv2d(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let dtype = pb.get_attr_datum_type("T")?;
    let patch = LocalPatch::build(pb)?;
    Ok(boxed_new!(Conv2D(dtype)(patch)))
}

impl<T: Datum> Conv2D<T> {
    /// Performs a 2D convolution on an input tensor and a filter.
    fn convolve(
        &self,
        data: &Array4<T>,
        filter: ArrayViewD<T>,
        pad_rows: bool,
        pad_cols: bool,
    ) -> Result<(Array4<T>)> {
        let images = BatchImageWrapper(data.view());

        let filter_rows = filter.shape()[0];
        let filter_cols = filter.shape()[1];
        let out_depth = filter.shape()[3];

        let out_height = self.0
            .adjusted_rows(images.h().into(), filter_rows)
            .to_integer()? as usize;
        let out_width = self.0
            .adjusted_cols(images.w().into(), filter_cols)
            .to_integer()? as usize;

        let filter = filter
            .view()
            .into_shape((filter_rows * filter_cols * images.d(), out_depth))?;

        let mut transformed: Vec<T> =
            Vec::with_capacity(images.n() * out_height * out_width * out_depth);

        // Loop over each batch.
        for image in data.outer_iter() {
            let patches = self.0
                .mk_patches(image, (filter_rows, filter_cols), pad_rows, pad_cols)?;
            transformed.extend(patches.dot(&filter).into_iter());
        }

        let transformed = Array::from_vec(transformed).into_shape((
            images.n(),
            out_height,
            out_width,
            out_depth,
        ))?;

        Ok(transformed)
    }
}

impl<T: Datum> Op for Conv2D<T> {
    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        let mut attributes = hashmap!{
            "T" => Attr::DatumType(T::datum_type()),
        };

        attributes.extend(self.0.get_attributes());
        attributes
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> Result<TVec<Value>> {
        let (m_data, m_filter) = args_2!(inputs);
        let data = m_data.into_array()?;
        let filter = m_filter.to_array_view()?;
        let data = into_4d(data)?;

        Ok(tvec![self.convolve(&data, filter, true, true)?.into_dyn().into(),
        ])
    }

    /// Returns a new streaming buffer for the operation.
    fn new_buffer(&self) -> Box<OpBuffer> {
        let buffer: Buffer<T> = Buffer {
            skip: 0,
            prev: None,
        };

        Box::new(buffer)
    }

    /// Evaluates one step of the operation on the given input tensors.
    fn step(
        &self,
        mut inputs: TVec<StepValue>,
        buffer: &mut Box<OpBuffer>,
    ) -> Result<Option<TVec<Value>>> {
        // We only support the VALID padding strategy for now, with the
        // streaming dimension being either the width or the height.

        // The idea is that, regardless of the strides, we need at least
        // as many chunks in the buffer as the size of the filter in the
        // streaming dimension to compute our first output chunk. Then,
        // we pop the min(buffer_size, k) first chunks from the buffer,
        // ignore the next max(k - buffer_size, 0) chunks, and wait for
        // the k following chunks to compte one output chunk, with k the
        // strides in the streaming dimension.

        let (data, filter) = args_2!(inputs);
        let filter = filter.into_const().ok_or("filter can not be streamed")?;
        let Stream { info, chunk, .. } = data.into_stream().ok_or("data must be streamed")?;
        let chunk = if let Some(chunk) = chunk {
            chunk
        } else {
            return Ok(None);
        };
        let dim = info.axis;

        // Maybe the data is streamed along the batch dimension.
        if dim == 0 {
            let result = self.eval(tvec![chunk, filter])?;
            return Ok(Some(result));
        }

        if dim < 1 || dim > 2 {
            bail!("Conv2D only supports batch, width and height streaming.");
        }

        let data = into_4d(chunk.into_array()?)?;
        let data_size = data.shape()[dim];
        if data_size != 1 {
            bail!("Conv2D expects chunks of dim 1 in streamed dim");
        }

        let filter = filter.to_array_view()?;
        let filter_size = filter.shape()[dim - 1];

        // Generates an empty 4-dimensional array of the right shape.
        let empty_array = || match dim {
            1 => Array::zeros((data.shape()[0], 0, data.shape()[2], data.shape()[3])),
            2 => Array::zeros((data.shape()[0], data.shape()[1], 0, data.shape()[3])),
            _ => panic!(),
        };

        let buffer = buffer
            .downcast_mut::<Buffer<T>>()
            .ok_or("The buffer can't be downcasted to Buffer<T>.")?;

        if buffer.prev.is_none() {
            buffer.prev = Some(empty_array());
        }

        let skip = &mut buffer.skip;
        let prev = buffer.prev.as_mut().unwrap();

        if *skip > 0 {
            *skip -= 1;
            return Ok(None);
        }

        let mut next = stack(Axis(dim), &[prev.view(), data.view()])?;
        let next_size = next.shape()[dim];

        // Maybe we don't have enough chunks to compute the convolution yet.
        if next_size < filter_size {
            *skip = 0;
            *prev = next;
            return Ok(None);
        }

        // Otherwise we compute the convolution using the non-streaming implementation.
        let result = self.convolve(&next, filter, dim != 1, dim != 2)?.into_dyn();
        let stride = [self.0.v_stride, self.0.h_stride][dim - 1];

        if stride > next_size {
            // Maybe we must pop more chunks from the buffer than it currently contains.
            *skip = stride - next_size;
            *prev = empty_array();
        } else {
            // Otherwise we pop the right number of chunks to prepare the next iteration.
            next.slice_axis_inplace(Axis(dim), Slice::from(stride..));
            *skip = 0;
            *prev = next;
        }

        Ok(Some(tvec![result.into()]))
    }
}

impl<T: Datum> InferenceRulesOp for Conv2D<T> {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver
            .equals(&inputs.len, 2)
            .equals(&outputs.len, 1)
            .equals(&inputs[0].datum_type, T::datum_type())
            .equals(&inputs[1].datum_type, T::datum_type())
            .equals(&outputs[0].datum_type, T::datum_type())
            .equals(&inputs[0].rank, 4)
            .equals(&inputs[1].rank, 4)
            .equals(&outputs[0].rank, 4)
            .equals(&inputs[0].shape[0], &outputs[0].shape[0])
            .equals(&inputs[0].shape[3], &inputs[1].shape[2])
            .equals(&outputs[0].shape[3], &inputs[1].shape[3])
            .given(&inputs[0].shape[1], move |solver, h| {
                solver.given(&inputs[1].shape[0], move |solver, kh| {
                    if let Ok(kh) = kh.to_integer() {
                        let oh = self.0.adjusted_rows(h, kh as usize);
                        solver.equals(&outputs[0].shape[1], oh);
                    }
                });
            })
            .given(&inputs[0].shape[2], move |solver, w| {
                solver.given(&inputs[1].shape[1], move |solver, kw| {
                    if let Ok(kw) = kw.to_integer() {
                        let ow = self.0.adjusted_cols(w, kw as usize);
                        solver.equals(&outputs[0].shape[2], ow);
                    }
                });
            });
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
        }).eval(tvec![mk(input).into(), mk(filter).into()])
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

        let result = conv.eval(tvec![data.into(), filter.into()])
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

        assert!(exp.close_enough(
            &conv.eval(tvec![data.into(), filter.into()]).unwrap()[0],
            true
        ))
    }

    #[test]
    fn inference_1() {
        use ops::InferenceOp;
        let op = Conv2D::<f32>::new(LocalPatch {
            padding: Padding::Valid,
            h_stride: 1,
            v_stride: 3,
            _data_format: DataFormat::NHWC,
        });
        let img = TensorFact::from(ArrayD::<f32>::zeros(vec![1, 1, 7, 1]));
        let ker = TensorFact::from(ArrayD::<f32>::zeros(vec![1, 3, 1, 1]));

        let (_, mut output_facts) = op.infer(tvec![img, ker], tvec![TensorFact::default()])
            .unwrap();

        output_facts[0].reduce();
        assert_eq!(
            output_facts,
            tvec![TensorFact::dt_shape(
                DatumType::F32,
                shapefact!(1, 1, (7 - 3 + 1), 1)
            )]
        );
    }

    #[test]
    fn inference_2() {
        use ops::InferenceOp;
        let op = Conv2D::<f32>::new(LocalPatch {
            padding: Padding::Same,
            h_stride: 1,
            v_stride: 1,
            _data_format: DataFormat::NHWC,
        });
        let img = TensorFact::from(ArrayD::<f32>::zeros(vec![1, 1, 1, 1]));
        let ker = TensorFact::from(ArrayD::<f32>::zeros(vec![1, 1, 1, 1]));

        let (_, output_facts) = op.infer(tvec![img, ker], tvec![TensorFact::default()])
            .unwrap();

        assert_eq!(
            output_facts,
            tvec![TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 1, 1))]
        );
    }
}
