#![cfg(feature = "conform")]
#![allow(non_snake_case)]
extern crate env_logger;
#[macro_use]
extern crate log;
#[macro_use]
extern crate proptest;
extern crate tract_tensorflow;

mod utils;

use crate::utils::*;
use proptest::prelude::*;
use tract_tensorflow::conform::*;
use tract_tensorflow::prelude::*;
use tract_tensorflow::tfpb;
use tract_tensorflow::tfpb::tensorflow::DataType::DtFloat;

fn convolution_pb(
    v_stride: usize,
    h_stride: usize,
    valid: bool,
    kernel: &Tensor,
) -> TractResult<Vec<u8>> {
    let conv = tfpb::node()
        .name("conv")
        .op("Conv2D")
        .input("data")
        .input("kernel")
        .attr("strides", vec![1, v_stride as i64, h_stride as i64, 1])
        .attr("padding", if valid { "VALID" } else { "SAME" })
        .attr("T", DtFloat);

    let graph =
        tfpb::graph().node(placeholder_f32("data")).node(const_f32("kernel", kernel)).node(conv);

    Ok(graph.write_to_bytes()?)
}

fn img_and_ker() -> BoxedStrategy<(Tensor, Tensor, (usize, usize))> {
    (1usize..4, 1usize..3, 1usize..3, 1usize..4)
        .prop_flat_map(|(ic, kh, kw, kc)| (1usize..2, kh..10, kw..10, Just((ic, kh, kw, kc))))
        .prop_flat_map(|(ib, ih, iw, (ic, kh, kw, kc))| {
            let i_size = ib * iw * ih * ic;
            let k_size = kw * kh * kc * ic;
            (
                Just((ib, ih, iw, ic)),
                Just((kh, kw, ic, kc)),
                ::proptest::collection::vec(-9i32..9, i_size..i_size + 1),
                ::proptest::collection::vec(-9i32..9, k_size..k_size + 1),
                (1..(kh + 1), 1..(kw + 1)),
            )
        })
        .prop_map(|(img_shape, ker_shape, img, ker, strides)| {
            (
                tract_ndarray::Array::from(img.into_iter().map(|i| i as f32).collect::<Vec<_>>())
                    .into_shape_with_order(img_shape)
                    .unwrap()
                    .into(),
                tract_ndarray::Array::from(ker.into_iter().map(|i| i as f32).collect::<Vec<_>>())
                    .into_shape_with_order(ker_shape)
                    .unwrap()
                    .into(),
                strides,
            )
        })
        .boxed()
}

proptest! {
    #[test]
    fn conv_compare((ref i, ref k, ref strides) in img_and_ker(),
                       valid in ::proptest::bool::ANY) {
        let model = convolution_pb(strides.0, strides.1, valid,& k).unwrap();
        compare(&model, vec!(("data", i.clone())), "conv")?;
    }

    #[test]
    fn conv_infer_facts((ref i, ref k, ref strides) in img_and_ker(),
                       valid in ::proptest::bool::ANY) {
        if valid {
            prop_assume!(i.shape()[1] >= k.shape()[0]);
            prop_assume!(i.shape()[2] >= k.shape()[1]);
        }
        let model = convolution_pb(strides.0, strides.1, valid, &k).unwrap();
        infer(&model, vec!(("data", i.clone())),  "conv")?;
    }
}

#[test]
fn conv_infer_facts_1() {
    let i: Tensor = tract_ndarray::ArrayD::<f32>::zeros(vec![1, 2, 2, 2]).into();
    let k: Tensor = tract_ndarray::ArrayD::<f32>::zeros(vec![2, 2, 2, 1]).into();
    let model = convolution_pb(1, 1, false, &k).unwrap();
    infer(&model, vec![("data", i.into())], "conv").unwrap();
}

#[test]
fn conv_eval_1() {
    let i: Tensor = Tensor::from(arr4(&[[[[0.0f32, 0.0], [1.0, 0.0]]]]));
    let k: Tensor = Tensor::from(arr4(&[[[[0.0f32], [0.0]], [[1.0], [0.0]]]]));
    let model = convolution_pb(1, 1, false, &k).unwrap();
    compare(&model, vec![("data", i.into())], "conv").unwrap();
}

#[test]
fn conv_eval_2() {
    let i: Tensor = Tensor::from(arr4(&[[[[0.0f32, -1.0]]]]));
    let k: Tensor = Tensor::from(arr4(&[[[[0.0f32, 0.0], [1.0, 0.0]]]]));
    let model = convolution_pb(1, 1, false, &k).unwrap();
    compare(&model, vec![("data", i.into())], "conv").unwrap();
}

#[test]
fn conv_eval_3() {
    let i: Tensor = Tensor::from(arr4(&[[[[0.0f32]]]]));
    let k: Tensor = Tensor::from(arr4(&[[[[0.0f32]]]]));
    let model = convolution_pb(1, 1, false, &k).unwrap();
    compare(&model, vec![("data", i.into())], "conv").unwrap();
}

#[test]
fn conv_eval_4() {
    let i: Tensor = Tensor::from(arr4(&[[[[1.0f32], [1.0], [0.0]]]]));
    let k: Tensor = Tensor::from(arr4(&[[[[0f32, 0.0]], [[0.0, -1.0]]]]));
    let model = convolution_pb(1, 1, true, &k).unwrap();
    compare(&model, vec![("data", i.into())], "conv").unwrap();
}

#[test]
fn conv_eval_5() {
    let i: Tensor = Tensor::from(arr4(&[[[[0.0f32, 0.0]], [[0.0, 1.0]]]]));
    let k: Tensor = Tensor::from(arr4(&[[[[0f32, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 1.0]]]]));
    let model = convolution_pb(1, 1, false, &k).unwrap();
    compare(&model, vec![("data", i.into())], "conv").unwrap();
}

#[test]
fn conv_eval_6() {
    let i: Tensor = Tensor::from(arr4(&[[[[0.0f32], [0.0]], [[0.0], [1.0]]]]));
    let k: Tensor = Tensor::from(arr4(&[[[[0f32]], [[1.0]]]]));
    let model = convolution_pb(1, 1, false, &k).unwrap();
    compare(&model, vec![("data", i.into())], "conv").unwrap();
}

#[test]
fn conv_eval_7() {
    let i: Tensor = Tensor::from(arr4(&[[[[0.0f32]], [[0.0]], [[0.0]], [[1.0]]]]));
    let k: Tensor = Tensor::from(arr4(&[[[[0.0f32]]], [[[1.0]]]]));
    let model = convolution_pb(2, 1, false, &k).unwrap();
    compare(&model, vec![("data", i.into())], "conv").unwrap();
}

#[test]
fn conv_eval_8() {
    let i: Tensor = Tensor::from(arr4(&[[[[0.0f32], [0.0]], [[0.0], [1.0]]]]));
    let k: Tensor = Tensor::from(arr4(&[[[[1.0f32]]]]));
    let model = convolution_pb(1, 1, false, &k).unwrap();
    compare(&model, vec![("data", i.into())], "conv").unwrap();
}

#[test]
fn conv_eval_mobilenet_v2() {
    let i: Tensor = Tensor::from(arr4(&[[[[0.0f32, -1.0]]]]));
    let k: Tensor = Tensor::from(arr4(&[[[[0.0f32, 0.0], [1.0, 0.0]]]]));
    let model = convolution_pb(1, 1, false, &k).unwrap();
    compare(&model, vec![("data", i.into())], "conv").unwrap();
}
