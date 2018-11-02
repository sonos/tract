#![allow(non_snake_case)]
#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate log;
extern crate ndarray;
#[macro_use]
extern crate proptest;
extern crate protobuf;
extern crate simplelog;
extern crate tensorflow;
#[macro_use]
extern crate tract_core;
extern crate tract_tensorflow;

mod conform;

use conform::*;
use ndarray::prelude::*;
use proptest::prelude::*;
use tract_tensorflow::tfpb;
use tract_tensorflow::tfpb::types::DataType::DT_FLOAT;

use tract_core::Tensor as TractTensor;

fn img_and_pool(
    ih: usize,
    iw: usize,
    ic: usize,
    kh: usize,
    kw: usize,
) -> BoxedStrategy<(TractTensor, (usize, usize), String, usize)> {
    (1..ih, 1..iw, 1..ic)
        .prop_flat_map(move |(ih, iw, ic)| {
            (
                Just((ih, iw, ic)),
                (1..kh.min(ih + 1).max(2), 1..kw.min(iw + 1).max(2)),
            )
        }).prop_flat_map(|((ih, iw, ic), k)| {
            let i_size = iw * ih * ic;
            (
                Just((1, ih, iw, ic)),
                Just(k),
                ::proptest::collection::vec(-255f32..255f32, i_size..i_size + 1),
                prop_oneof!("VALID", "SAME"),
                1..(k.0.min(k.1).max(2)),
            )
        }).prop_map(|(img_shape, k, img, padding, stride)| {
            (
                Array::from_vec(img).into_shape(img_shape).unwrap().into(),
                k,
                padding,
                stride,
            )
        }).boxed()
}

proptest! {
    #[test]
    fn maxpool((ref i, k, ref padding, stride) in img_and_pool(32, 32, 5, 16, 16)) {
        let graph = tfpb::graph()
            .node(placeholder_f32("data"))
            .node(tfpb::node()
                .name("pool")
                .op("MaxPool")
                .input("data")
                .attr("T", DT_FLOAT)
                .attr("strides", vec![1, stride as i64, stride as i64, 1])
                .attr("ksize", vec![1, k.0 as i64, k.1 as i64, 1])
                .attr("padding", &**padding))
            .write_to_bytes()?;

        compare(&graph, vec!(("data", i.clone())), "pool")?;
    }
}

proptest! {
    #[test]
    fn avgpool((ref i, k, ref padding, stride) in img_and_pool(32, 32, 5, 16, 16)) {
        let graph = tfpb::graph()
            .node(placeholder_f32("data"))
            .node(tfpb::node()
                .name("pool")
                .op("AvgPool")
                .input("data")
                .attr("T", DT_FLOAT)
                .attr("strides", vec![1, stride as i64, stride as i64, 1])
                .attr("ksize", vec![1, k.0 as i64, k.1 as i64, 1])
                .attr("padding", &**padding))
            .write_to_bytes()?;

        compare(&graph, vec!(("data", i.clone())), "pool")?;
    }
}
