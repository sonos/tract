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

fn fake_quant_with_min_max_vars(
    input: f32,
    min: f32,
    max: f32,
    narrow_range: bool,
    num_bits: i64,
) -> proptest::test_runner::TestCaseResult {
    let graph = tfpb::graph()
        .node(const_f32("input", &tensor0(input)))
        .node(const_f32("min", &tensor0(min)))
        .node(const_f32("max", &tensor0(max)))
        .node(
            tfpb::node()
                .name("op")
                .op("FakeQuantWithMinMaxVars")
                .input("input")
                .input("min")
                .input("max")
                .attr("narrow_range", narrow_range)
                .attr("num_bits", num_bits),
        );
    let graph = graph.write_to_bytes().unwrap();
    compare::<&'static str>(&graph, vec![], "op")
}

//may fails due to rounding errors
proptest! {
    #[test]
    #[ignore]
    fn ops_fake_quant_with_min_max_vars(input in -10f32..10f32, min in -10f32..-0.1f32, max in 0.1f32..10f32, narrow_range: bool, num_bits in 2..8i64) {

        fake_quant_with_min_max_vars(input, min, max, narrow_range, num_bits)?
    }
}

#[test]
fn trivial0() -> std::result::Result<(), TestCaseError> {
    fake_quant_with_min_max_vars(0.0, -1., 1., false, 2)
}

//fails due to rounding errors (recip != div)
#[test]
#[ignore]
fn trivial1() -> std::result::Result<(), TestCaseError> {
    fake_quant_with_min_max_vars(-0.059596814, -0.9537212, 0.8271718, true, 8)
}
