use tract_hir::internal::*;
use tract_hir::ops::cnn;
use tract_hir::ops::nn::DataFormat;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

pub fn conv2d(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let strides = super::strides(pb)?;
    let mut op =
        cnn::Conv::default().hwio().padding(super::padding(pb)?).strides(strides[1..3].into());
    if super::data_format(pb)? == DataFormat::NHWC {
        op = op.nhwc()
    }
    Ok(expand(op))
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use tract_hir::ops::cnn::{Conv, PaddingSpec};
    use tract_ndarray::*;

    fn mk(sizes: &[usize]) -> Tensor {
        Array::range(1f32, sizes.iter().product::<usize>() as f32 + 1.0, 1.0)
            .into_shape_with_order(sizes)
            .unwrap()
            .into()
    }

    fn make_conv(h_stride: usize, v_stride: usize, padding: PaddingSpec) -> Box<dyn InferenceOp> {
        expand(Conv::default().nhwc().hwio().padding(padding).strides(tvec![v_stride, h_stride]))
    }

    fn verify(input: Tensor, filter: Tensor, stride: usize, padding: PaddingSpec, expect: &[f32]) {
        let result = make_conv(stride, stride, padding)
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
            tensor4(&[[[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]]),
            1,
            PaddingSpec::Valid,
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
            PaddingSpec::Valid,
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
            PaddingSpec::Valid,
            &[231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0, 936.0, 1029.0],
        )
    }

    #[test]
    fn testConv2D2x1Filter() {
        verify(
            mk(&[1, 2, 3, 3]),
            mk(&[2, 1, 3, 3]),
            1,
            PaddingSpec::Valid,
            &[465.0, 504.0, 543.0, 618.0, 675.0, 732.0, 771.0, 846.0, 921.0],
        );
    }

    #[test]
    fn testConv2D2x2Filter() {
        verify(
            mk(&[1, 2, 3, 3]),
            mk(&[2, 2, 3, 3]),
            1,
            PaddingSpec::Valid,
            &[2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0],
        )
    }

    #[test]
    fn testConv2D2x2FilterStride2() {
        verify(
            mk(&[1, 2, 3, 3]),
            mk(&[2, 2, 3, 3]),
            2,
            PaddingSpec::Valid,
            &[2271.0, 2367.0, 2463.0],
        )
    }

    #[test]
    fn testConv2D2x2FilterStride2Same() {
        verify(
            mk(&[1, 2, 3, 3]),
            mk(&[2, 2, 3, 3]),
            2,
            PaddingSpec::SameUpper,
            &[2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0],
        );
    }

    #[test]
    fn test_conv_1() {
        let conv = make_conv(1, 1, PaddingSpec::SameUpper);
        // NHWC
        let data = tensor4(&[[[[1f32]]]]);
        // HWIO
        let filter = tensor4(&[[[[0.0f32]]], [[[1.0]]], [[[0.0]]]]);
        let exp = tensor4(&[[[[1f32]]]]);

        let result = conv.eval(tvec![data.into(), filter.into()]).unwrap();
        result[0].close_enough(&exp, Approximation::Approximate).unwrap()
    }

    #[test]
    fn test_conv_2() {
        let conv = make_conv(1, 1, PaddingSpec::SameUpper);
        let data = tensor4(&[[[[142.3088f32], [48.891083]], [[208.3187], [-11.274994]]]]);
        let filter =
            tensor4(&[[[[160.72833f32]], [[107.84076]]], [[[247.50552]], [[-38.738464]]]]);
        let exp = tensor4(&[[[[80142.31f32], [5067.5586]], [[32266.81], [-1812.2109]]]]);
        let got = &conv.eval(tvec![data.into(), filter.into()]).unwrap()[0];
        //println!("{:?}", got);
        //println!("{:?}", exp);
        exp.close_enough(got, true).unwrap()
    }

    #[test]
    fn inference_1() {
        let mut op = make_conv(1, 3, PaddingSpec::Valid);
        let img = InferenceFact::from(Tensor::zero::<f32>(&[1, 1, 7, 1]).unwrap());
        let ker = InferenceFact::from(Tensor::zero::<f32>(&[1, 3, 1, 1]).unwrap());
        let any = InferenceFact::default();

        let (_, output_facts, _) = op.infer_facts(tvec![&img, &ker], tvec![&any], tvec!()).unwrap();

        assert_eq!(output_facts, tvec![f32::fact([1, 1, (7 - 3 + 1), 1]).into()]);
    }

    #[test]
    fn inference_2() {
        let mut op = make_conv(1, 1, PaddingSpec::SameUpper);
        let img = InferenceFact::from(Tensor::zero::<f32>(&[1, 1, 1, 1]).unwrap());
        let ker = InferenceFact::from(Tensor::zero::<f32>(&[1, 1, 1, 1]).unwrap());
        let any = InferenceFact::default();

        let (_, output_facts, _) = op.infer_facts(tvec![&img, &ker], tvec![&any], tvec!()).unwrap();

        assert_eq!(output_facts, tvec![f32::fact([1, 1, 1, 1]).into()]);
    }
}
