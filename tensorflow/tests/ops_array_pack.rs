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

use ndarray::prelude::*;
use conform::*;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::Tensor as TractTensor;
use tract_tensorflow::tfpb;
use tract_tensorflow::tfpb::types::DataType::DT_INT32;

fn strat() -> BoxedStrategy<(usize, Vec<TractTensor>)> {
    // input rank
    (1usize..8)
        // rank, dimensions, number of inputs
        .prop_flat_map(|r| (0usize..r, vec(1usize..5, r..r + 1), 1..5))
        .prop_map(|(ax, dims, n)| {
            let size = dims.iter().map(|a| *a).product::<usize>();
            let mats: Vec<TractTensor> = (0..n)
                .map(|ix| {
                    TractTensor::from(
                        Array::from_shape_vec(dims.clone(), ((ix * 1000)..).take(size).collect())
                            .unwrap(),
                    )
                }).collect();
            (ax, mats)
        }).boxed()
}

proptest! {
    #[test]
    fn pack((axis, ref inputs) in strat()) {
        let mut graph = tfpb::graph();
        let mut graph_inputs = vec!();
        let mut pack = tfpb::node()
            .name("op")
            .op("Pack")
            .attr("T", DT_INT32)
            .attr("N", inputs.len() as i64)
            .attr("axis", axis as i64);
        for (ix,input) in inputs.iter().enumerate() {
            let input_name = format!("input-{}", ix);
            graph = graph.node(placeholder_i32(&input_name));
            pack = pack.input(&input_name);
            graph_inputs.push((input_name, input.clone()));
        }
        graph = graph.node(pack);
        let graph = graph.write_to_bytes().unwrap();
        compare(&graph, graph_inputs, "op")?
    }
}
