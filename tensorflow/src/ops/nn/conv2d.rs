use std::marker::PhantomData;

use super::local_patch::*;
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use tract_core::ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Conv2D<T: Datum + LinalgScalar>(LocalPatch, PhantomData<T>);

/*
pub fn conv2d(pb: &::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
    let dtype = pb.get_attr_datum_type("T")?;
    let patch = LocalPatch::build(pb)?;
    Ok(boxed_new!(Conv2D(dtype)(patch)))
}
*/

pub fn conv2d(pb: &::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
    use tract_core::ops::nn::*;
    let data_format = if pb.get_attr_opt_raw_str("data_format")?.unwrap_or(b"NHWC") == b"NHWC" {
        DataFormat::NHWC
    } else {
        DataFormat::NCHW
    };
    let strides: Vec<usize> = pb.get_attr_list_int("strides")?;
    if strides.len() != 4 || strides[0] != 1 && strides[3] != 1 {
        Err(format!(
            "strides must be of the form [1, h, v, 1], found {:?}",
            strides
        ))?
    };
    let padding = pb.get_attr_raw_str("padding")?;
    let padding = match padding {
        b"VALID" => ::tract_core::ops::nn::PaddingSpec::Valid,
        b"SAME" => ::tract_core::ops::nn::PaddingSpec::SameUpper,
        s => Err(format!(
            "unsupported Padding {}",
            String::from_utf8_lossy(s)
        ))?,
    };
    Ok(Box::new(Conv::new(
        data_format,
        true,
        None,
        None,
        padding,
        Some(strides[1..3].into()),
        1,
    )))
}

impl<T: Datum + LinalgScalar> Conv2D<T> {
    /// Performs a 2D convolution on an input tensor and a filter.
    fn convolve(
        &self,
        data: &Array4<T>,
        filter: ArrayViewD<T>,
        pad_rows: bool,
        pad_cols: bool,
    ) -> TractResult<(Array4<T>)> {
        let images = BatchImageWrapper(data.view());

        let filter_rows = filter.shape()[0];
        let filter_cols = filter.shape()[1];
        let out_depth = filter.shape()[3];

        let out_height = self
            .0
            .adjusted_rows(images.h().into(), filter_rows)
            .to_integer()? as usize;
        let out_width = self
            .0
            .adjusted_cols(images.w().into(), filter_cols)
            .to_integer()? as usize;

        let filter = filter
            .view()
            .into_shape((filter_rows * filter_cols * images.d(), out_depth))?;

        let mut transformed: Vec<T> =
            Vec::with_capacity(images.n() * out_height * out_width * out_depth);

        // Loop over each batch.
        for image in data.outer_iter() {
            let patches =
                self.0
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

impl<T: Datum + LinalgScalar> Op for Conv2D<T> {
    fn name(&self) -> &str {
        "tf.Conv2D"
    }
}

impl<T: Datum + LinalgScalar> StatelessOp for Conv2D<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (m_data, m_filter) = args_2!(inputs);
        let data = m_data.to_array()?;
        let filter = m_filter.to_array_view()?;
        let data = into_4d(data)?;

        Ok(tvec![
            self.convolve(&data, filter, true, true)?.into_dyn().into(),
        ])
    }
}

impl<T: Datum + LinalgScalar> InferenceRulesOp for Conv2D<T> {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 2)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, T::datum_type())?;
        s.equals(&inputs[1].datum_type, T::datum_type())?;
        s.equals(&outputs[0].datum_type, T::datum_type())?;
        s.equals(&inputs[0].rank, 4)?;
        s.equals(&inputs[1].rank, 4)?;
        s.equals(&outputs[0].rank, 4)?;
        s.equals(&inputs[0].shape[0], &outputs[0].shape[0])?;
        s.equals(&inputs[0].shape[3], &inputs[1].shape[2])?;
        s.equals(&outputs[0].shape[3], &inputs[1].shape[3])?;
        s.given_2(&inputs[0].shape[1], &inputs[1].shape[0], move |s, h, kh| {
            if let Ok(kh) = kh.to_integer() {
                let oh = self.0.adjusted_rows(h, kh as usize);
                s.equals(&outputs[0].shape[1], oh)?;
            }
            Ok(())
        })?;
        s.given_2(&inputs[0].shape[2], &inputs[1].shape[1], move |s, w, kw| {
            if let Ok(kw) = kw.to_integer() {
                let ow = self.0.adjusted_cols(w, kw as usize);
                s.equals(&outputs[0].shape[2], ow)?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use tract_core::ops::nn::{Conv, DataFormat, PaddingSpec};
    use tract_core::Tensor;

    fn mk(sizes: &[usize]) -> Tensor {
        ::ndarray::Array::range(1f32, sizes.iter().product::<usize>() as f32 + 1.0, 1.0)
            .into_shape(sizes)
            .unwrap()
            .into()
    }

    /*
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
    */

    fn make_conv(h_stride: usize, v_stride: usize, padding: Padding) -> Box<Op> {
        Box::new(Conv::new(
            DataFormat::NHWC,
            true,
            None,
            None,
            match padding {
                Padding::Valid => PaddingSpec::Valid,
                Padding::Same => PaddingSpec::SameUpper,
            },
            Some(tvec![v_stride, h_stride]),
            1,
        ))
    }

    fn verify(input: Tensor, filter: Tensor, stride: usize, padding: Padding, expect: &[f32]) {
        let result = make_conv(stride, stride, padding)
            .as_stateless()
            .unwrap()
            .eval(tvec![input.into(), filter.into()])
            .unwrap()
            .remove(0);
        assert_eq!(expect.len(), result.shape().iter().product::<usize>());
        let found = result.to_array_view::<f32>().unwrap();
        let expect = ArrayD::from_shape_vec(found.shape(), expect.to_vec()).unwrap();
        assert_eq!(expect, found);
    }

    #[test]
    fn testConv2D3CNoopFilter() {
        verify(
            mk(&[1, 2, 3, 3]),
            arr4(&[[[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]]).into(),
            1,
            Padding::Valid,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0,
            ],
        )
    }

    #[test]
    fn testConv2D1x1Filter() {
        verify(
            mk(&[1, 2, 3, 3]),
            mk(&[1, 1, 3, 3]),
            1,
            Padding::Valid,
            &[
                30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0,
                174.0, 216.0, 258.0, 210.0, 261.0, 312.0,
            ],
        );
    }

    #[test]
    fn testConv2D1x2Filter() {
        verify(
            mk(&[1, 2, 3, 3]),
            mk(&[1, 2, 3, 3]),
            1,
            Padding::Valid,
            &[
                231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0, 936.0,
                1029.0,
            ],
        )
    }

    #[test]
    fn testConv2D2x1Filter() {
        verify(
            mk(&[1, 2, 3, 3]),
            mk(&[2, 1, 3, 3]),
            1,
            Padding::Valid,
            &[
                465.0, 504.0, 543.0, 618.0, 675.0, 732.0, 771.0, 846.0, 921.0,
            ],
        );
    }

    #[test]
    fn testConv2D2x2Filter() {
        verify(
            mk(&[1, 2, 3, 3]),
            mk(&[2, 2, 3, 3]),
            1,
            Padding::Valid,
            &[2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0],
        )
    }

    #[test]
    fn testConv2D2x2FilterStride2() {
        verify(
            mk(&[1, 2, 3, 3]),
            mk(&[2, 2, 3, 3]),
            2,
            Padding::Valid,
            &[2271.0, 2367.0, 2463.0],
        )
    }

    #[test]
    fn testConv2D2x2FilterStride2Same() {
        verify(
            mk(&[1, 2, 3, 3]),
            mk(&[2, 2, 3, 3]),
            2,
            Padding::Same,
            &[2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0],
        );
    }

    #[test]
    fn test_conv_1() {
        let conv = make_conv(1, 1, Padding::Same);
        // NHWC
        let data: SharedTensor = arr4(&[[[[1f32]]]]).into();
        // HWIO
        let filter: SharedTensor = arr4(&[[[[0.0f32]]], [[[1.0]]], [[[0.0]]]]).into();
        let exp: SharedTensor = arr4(&[[[[1f32]]]]).into();

        let result = conv
            .as_stateless()
            .unwrap()
            .eval(tvec![data, filter])
            .unwrap()
            .remove(0);
        assert_eq!(exp, result);
    }

    #[test]
    fn test_conv_2() {
        let conv = make_conv(1, 1, Padding::Same);
        let data: SharedTensor =
            arr4(&[[[[142.3088f32], [48.891083]], [[208.3187], [-11.274994]]]]).into();
        let filter: SharedTensor = arr4(&[
            [[[160.72833f32]], [[107.84076]]],
            [[[247.50552]], [[-38.738464]]],
        ]).into();
        let exp: SharedTensor = arr4(&[[[[80142.31f32], [5067.5586]], [[32266.81], [-1812.2109]]]]).into();
        let got = &conv.as_stateless().unwrap().eval(tvec![data, filter]).unwrap()[0];
        println!("{:?}", got);
        println!("{:?}", exp);
        assert!(exp.close_enough(&got, true));
    }

    #[test]
    fn inference_1() {
        let op = make_conv(1, 3, Padding::Valid);
        let img = TensorFact::from(ArrayD::<f32>::zeros(vec![1, 1, 7, 1]));
        let ker = TensorFact::from(ArrayD::<f32>::zeros(vec![1, 3, 1, 1]));
        let any = TensorFact::default();

        let (_, output_facts) = op.infer_facts(tvec![&img, &ker], tvec![&any]).unwrap();

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
        let op = make_conv(1, 1, Padding::Same);
        let img = TensorFact::from(ArrayD::<f32>::zeros(vec![1, 1, 1, 1]));
        let ker = TensorFact::from(ArrayD::<f32>::zeros(vec![1, 1, 1, 1]));
        let any = TensorFact::default();

        let (_, output_facts) = op.infer_facts(tvec![&img, &ker], tvec![&any]).unwrap();

        assert_eq!(
            output_facts,
            tvec![TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 1, 1))]
        );
    }
}
