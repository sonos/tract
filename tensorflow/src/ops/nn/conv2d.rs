use tract_core::ops::prelude::*;
use tract_core::ops::nn::*;

pub fn conv2d(pb: &crate::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
    let data_format = super::data_format(pb)?;
    let padding = super::padding(pb)?;
    let strides = super::strides(pb)?;
    Ok(Box::new(Conv::new(
        data_format,
        KernelFormat::HWIO,
        None,
        None,
        padding,
        Some(strides[1..3].into()),
        1,
    )))
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use ndarray::*;
    use tract_core::ops::nn::{Conv, DataFormat, KernelFormat, PaddingSpec};
    use tract_core::Tensor;

    fn mk(sizes: &[usize]) -> Tensor {
        ::ndarray::Array::range(1f32, sizes.iter().product::<usize>() as f32 + 1.0, 1.0)
            .into_shape(sizes)
            .unwrap()
            .into()
    }

    fn make_conv(h_stride: usize, v_stride: usize, padding: PaddingSpec) -> Box<Op> {
        Box::new(Conv::new(
            DataFormat::NHWC,
            KernelFormat::HWIO,
            None,
            None,
            padding,
            Some(tvec![v_stride, h_stride]),
            1,
        ))
    }

    fn verify(input: Tensor, filter: Tensor, stride: usize, padding: PaddingSpec, expect: &[f32]) {
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
            PaddingSpec::Valid,
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
        let conv = make_conv(1, 1, PaddingSpec::SameUpper);
        let data: SharedTensor =
            arr4(&[[[[142.3088f32], [48.891083]], [[208.3187], [-11.274994]]]]).into();
        let filter: SharedTensor = arr4(&[
            [[[160.72833f32]], [[107.84076]]],
            [[[247.50552]], [[-38.738464]]],
        ])
        .into();
        let exp: SharedTensor =
            arr4(&[[[[80142.31f32], [5067.5586]], [[32266.81], [-1812.2109]]]]).into();
        let got = &conv
            .as_stateless()
            .unwrap()
            .eval(tvec![data, filter])
            .unwrap()[0];
        println!("{:?}", got);
        println!("{:?}", exp);
        assert!(exp.close_enough(&got, true));
    }

    #[test]
    fn inference_1() {
        let op = make_conv(1, 3, PaddingSpec::Valid);
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
        let op = make_conv(1, 1, PaddingSpec::SameUpper);
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
