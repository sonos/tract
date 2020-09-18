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
use proptest::collection::vec;
use proptest::prelude::*;
use tract_tensorflow::conform::*;
use tract_tensorflow::prelude::*;
use tract_tensorflow::tfpb;
use tract_tensorflow::tfpb::tensorflow::DataType;

fn random_uniform_float(shape: &[i32], seed: (i32, i32)) -> proptest::test_runner::TestCaseResult {
    let graph = tfpb::graph().node(const_i32("shape", &tensor1(&*shape))).node(
        tfpb::node()
            .name("op")
            .op("RandomUniform")
            .input("shape")
            .attr("T", DataType::DtInt32)
            .attr("dtype", DataType::DtFloat)
            .attr("seed", seed.0)
            .attr("seed2", seed.1),
    );
    let graph = graph.write_to_bytes().unwrap();
    compare::<&'static str>(&graph, vec![], "op")
}

proptest! {
    #[test]
    fn proptest_random_uniform_float(shape in vec(1..5, 0..4), seed in ((1..4),(1..4))) {
        random_uniform_float(&*shape, seed)?
    }
}

#[test]
fn random_uniform_float_1() {
    random_uniform_float(&[], (1, 1)).unwrap();
}
