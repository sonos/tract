use crate::internal::*;
use crate::ops::cnn::*;
use crate::ops::matmul::*;
use crate::ops::nn::DataFormat::*;

fn test_conv_q_and_bias(a0: i32, b0: i32, c0: i32, c_scale: f32, k: i8, i: i8, bias: i32) {
    use super::*;
    let mut model = TypedModel::default();
    let source = model.add_source("input", TypedFact::dt_shape(i8::datum_type(), &[1, 1])).unwrap();
    let mut q_params = QParams::noop_static(i32::datum_type());
    q_params.a0 = QParam::Static(rctensor0(a0));
    q_params.b0 = QParam::Static(rctensor0(b0));
    q_params.c_scale = QParam::Static(rctensor0(c_scale));
    q_params.c0 = QParam::Static(rctensor0(c0));
    let conv = ConvUnary {
        pool_spec: PoolSpec {
            data_format: CHW,
            kernel_shape: tvec![1],
            padding: PaddingSpec::Valid,
            dilations: None,
            strides: None,
            output_channel_override: Some(1),
        },
        kernel_fmt: KernelFormat::OIHW,
        kernel: rctensor3(&[[[k]]]),
        group: 1,
        bias: Some(rctensor1(&[bias])),
        q_params: Some((i32::datum_type(), q_params)),
    };
    let output = model.wire_node("conv", conv, &[source]).unwrap();
    model.set_output_outlets(&output).unwrap();

    let input = tvec!(tensor2(&[[i]]));
    fn round_away(x: f32) -> f32 {
        x.abs().round() * x.signum()
    }
    let expected =
        round_away((((k as i32) - a0) * ((i as i32) - b0) + bias) as f32 / c_scale) as i32 + c0;
    dbg!(&expected);

    let expected = tensor2(&[[expected]]);

    //        dbg!(&model);
    let output = model.clone().into_runnable().unwrap().run(input.clone()).unwrap();
    assert_eq!(&*output[0], &expected);

    let output = model.declutter().unwrap().into_runnable().unwrap().run(input.clone()).unwrap();
    assert_eq!(&*output[0], &expected);

    let output = model.into_optimized().unwrap().into_runnable().unwrap().run(input).unwrap();
    assert_eq!(&*output[0], &expected);
}

proptest::proptest! {
    #[test]
    fn conv_q_and_bias_prop(a0 in 0i32..5, b0 in 0i32..5, c0 in 0i32..5, c_scale in 0f32..1., k in 0i8..5, i in 0i8..5, bias in 0i32..5) {
        test_conv_q_and_bias(a0, b0, c0, c_scale, i, k, bias)
    }
}

#[test]
fn conv_q_and_bias_0() {
    test_conv_q_and_bias(0, 0, 0, 0.4447719, 0, 0, 1)
}

#[test]
fn conv_q_and_bias_1() {
    test_conv_q_and_bias(1, 0, 0, 0.4447719, 0, 1, 0)
}
