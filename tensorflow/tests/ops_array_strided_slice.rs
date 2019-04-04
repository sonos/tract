#![cfg(feature = "conform")]
#![allow(non_snake_case)]
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate ndarray;
#[macro_use]
extern crate proptest;
extern crate protobuf;
extern crate tract_core;
extern crate tract_tensorflow;

mod utils;

use crate::utils::*;
use ndarray::prelude::*;
use proptest::prelude::*;
use protobuf::Message;
use tract_core::prelude::*;
use tract_tensorflow::conform::*;
use tract_tensorflow::tfpb;
use tract_tensorflow::tfpb::types::DataType::DT_INT32;

fn strided_slice_strat() -> BoxedStrategy<(
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    (i32, i32, i32, i32, i32),
)> {
    ::proptest::collection::vec(
        (1..5).prop_flat_map(|n| {
            // each dim max
            (
                Just(n),       // input size
                0..n,          // begin
                0..n,          // end
                1..2.max(n),   // stride, abs
                any::<bool>(), // make begin negative
                any::<bool>(), // make end negative
            )
        }),
        1..4, // rank
    )
    .prop_flat_map(|dims| {
        let rank = dims.iter().len();
        (Just(dims), (0..(1 << rank), 0..(1 << rank), Just(0), Just(0), 0..(1 << rank)))
    })
    .prop_map(|(dims, masks)| {
        let shape = dims.iter().map(|d| d.0 as usize).collect::<Vec<_>>();
        let size: i32 = shape.iter().map(|d| *d as i32).product();
        (
            Tensor::from(Array::from_shape_vec(shape, (0..size).collect()).unwrap()),
            Array::from_vec(dims.iter().map(|d| if d.4 { d.1 - d.0 } else { d.1 }).collect())
                .into(),
            Array::from_vec(dims.iter().map(|d| if d.5 { d.2 - d.0 } else { d.2 }).collect())
                .into(),
            Array::from_vec(
                dims.iter()
                    .enumerate()
                    .map(|(ix, d)| {
                        if d.2 == d.1 || masks.4 & (1 << ix) != 0 {
                            1
                        } else {
                            d.3 as i32 * (d.2 as i32 - d.1 as i32).signum()
                        }
                    })
                    .collect(),
            )
            .into(),
            masks,
        )
    })
    .boxed()
}

proptest! {
    #[test]
    fn strided_slice((ref i, ref b, ref e, ref s, ref masks) in strided_slice_strat()) {
        let graph = tfpb::graph()
            .node(placeholder_i32("input"))
            .node(placeholder_i32("begin"))
            .node(placeholder_i32("end"))
            .node(placeholder_i32("stride"))
            .node(tfpb::node().name("op")
                  .attr("T", DT_INT32)
                  .attr("Index", DT_INT32)
                  .attr("begin_mask", masks.0 as i64)
                  .attr("end_mask", masks.1 as i64)
                  .attr("shrink_axis_mask", masks.4 as i64)
                  .input("input").input("begin")
                  .input("end").input("stride")
                  .op("StridedSlice")
            ).write_to_bytes().unwrap();

        let inputs = vec!(("input", i.clone()),("begin", b.clone()), ("end", e.clone()), ("stride", s.clone()));
        let res = compare(&graph, inputs, "op")?;
        res
    }
}

#[test]
fn strided_slice_1() {
    let graph = tfpb::graph()
        .node(placeholder_i32("input"))
        .node(placeholder_i32("begin"))
        .node(placeholder_i32("end"))
        .node(placeholder_i32("stride"))
        .node(
            tfpb::node()
                .name("op")
                .attr("T", DT_INT32)
                .attr("Index", DT_INT32)
                .input("input")
                .input("begin")
                .input("end")
                .input("stride")
                .op("StridedSlice"),
        )
        .write_to_bytes()
        .unwrap();

    let inputs = vec![
        ("input", arr2(&[[0, 6], [0, 0]]).into()),
        ("begin", arr1(&[0]).into()),
        ("end", arr1(&[2]).into()),
        ("stride", arr1(&[1]).into()),
    ];
    compare(&graph, inputs, "op").unwrap()
}

#[test]
fn strided_slice_2() {
    let graph = tfpb::graph()
        .node(placeholder_i32("input"))
        .node(placeholder_i32("begin"))
        .node(placeholder_i32("end"))
        .node(placeholder_i32("stride"))
        .node(
            tfpb::node()
                .name("op")
                .attr("T", DT_INT32)
                .attr("Index", DT_INT32)
                .attr("shrink_axis_mask", 1 as i64)
                .input("input")
                .input("begin")
                .input("end")
                .input("stride")
                .op("StridedSlice"),
        )
        .write_to_bytes()
        .unwrap();

    let inputs = vec![
        ("input", arr1(&[0]).into()),
        ("begin", arr1(&[0]).into()),
        ("end", arr1(&[0]).into()),
        ("stride", arr1(&[1]).into()),
    ];
    compare(&graph, inputs, "op").unwrap()
}
