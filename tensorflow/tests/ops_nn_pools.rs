#![cfg(feature = "conform")]
#![allow(non_snake_case)]
#[macro_use]
extern crate log;
extern crate env_logger;
extern crate ndarray;
#[macro_use]
extern crate proptest;
extern crate protobuf;
extern crate tensorflow;
extern crate tract_core;
extern crate tract_tensorflow;

mod utils;

use crate::utils::*;
use ndarray::prelude::*;
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use protobuf::Message;
use tract_tensorflow::conform::*;
use tract_tensorflow::tfpb;
use tract_tensorflow::tfpb::types::DataType::DT_FLOAT;

fn img_and_pool() -> BoxedStrategy<(Array4<f32>, (usize, usize), String, usize)> {
    (1usize..8, 1usize..8, 1usize..8)
        .prop_flat_map(move |(ih, iw, ic)| (Just((ih, iw, ic)), (1usize..3, 1usize..3)))
        .prop_flat_map(|((ih, iw, ic), k)| {
            let i_size = iw * ih * ic;
            (
                Just((1, ih, iw, ic)),
                Just(k),
                ::proptest::collection::vec((-10..10).prop_map(|a| a as f32), i_size..i_size + 1),
                prop_oneof!("VALID", "SAME"),
                1usize..3,
            )
        })
        .prop_map(|(img_shape, k, img, padding, stride)| {
            (Array::from_vec(img).into_shape(img_shape).unwrap(), k, padding, stride)
        })
        .boxed()
}

fn pool(
    op: &str,
    i: &Array4<f32>,
    k: (usize, usize),
    padding: &str,
    stride: usize,
) -> TestCaseResult {
    if padding == "VALID" {
        prop_assume!(i.shape()[1] >= k.0);
        prop_assume!(i.shape()[2] >= k.1);
    }
    let graph = tfpb::graph()
        .node(placeholder_f32("data"))
        .node(
            tfpb::node()
                .name("pool")
                .op(op)
                .input("data")
                .attr("T", DT_FLOAT)
                .attr("strides", vec![1, stride as i64, stride as i64, 1])
                .attr("ksize", vec![1, k.0 as i64, k.1 as i64, 1])
                .attr("padding", padding),
        )
        .write_to_bytes()?;
    compare(&graph, vec![("data", i.clone().into())], "pool")
}

proptest! {
    #[test]
    fn proptest_maxpool((ref i, k, ref padding, stride) in img_and_pool()) {
        pool("MaxPool", i, k, padding, stride)?;
    }
}

proptest! {
    #[test]
    fn proptest_avgpool((ref i, k, ref padding, stride) in img_and_pool()) {
        pool("AvgPool", i, k, padding, stride)?;
    }
}

#[test]
fn maxpool_1() {
    pool("MaxPool", &Array4::<f32>::zeros((1, 1, 4, 1)), (1, 2), "SAME", 1).unwrap();
}

#[test]
fn avgpool_1() {
    pool("AvgPool", &Array4::<f32>::zeros((1, 1, 6, 1)), (1, 1), "SAME", 2).unwrap();
}
