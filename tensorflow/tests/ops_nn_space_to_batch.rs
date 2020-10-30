#![cfg(feature = "conform")]
#![allow(non_snake_case)]
extern crate env_logger;
#[macro_use]
extern crate log;
#[macro_use]
extern crate proptest;
extern crate tensorflow;
extern crate tract_tensorflow;

mod utils;

use crate::utils::*;
use proptest::prelude::*;
use tract_ndarray::prelude::*;
use tract_tensorflow::conform::*;
use tract_tensorflow::prelude::*;
use tract_tensorflow::tfpb;
use tract_tensorflow::tfpb::tensorflow::DataType::DtFloat;

fn space_to_batch_strat() -> BoxedStrategy<(Tensor, Tensor, Tensor)> {
    use proptest::collection::vec;
    (1usize..4, vec(1usize..8, 1usize..4), vec(1usize..8, 1usize..4))
        .prop_flat_map(|(b, spatial_dims, non_spatial_dims)| {
            (
                Just(b),
                Just(spatial_dims.clone()),
                Just(non_spatial_dims),
                vec(1usize..4, spatial_dims.len()..spatial_dims.len() + 1),
                vec(0usize..4, spatial_dims.len()..spatial_dims.len() + 1),
            )
        })
        .prop_filter("block < input", |&(_, ref sd, _, ref bs, _)| {
            bs.iter().zip(sd.iter()).all(|(bs, is)| bs <= is)
        })
        .prop_map(
            |(b, sd, nsd, bs, left_pad): (
                usize,
                Vec<usize>,
                Vec<usize>,
                Vec<usize>,
                Vec<usize>,
            )| {
                let mut input_shape = vec![b];
                input_shape.extend(&sd);
                input_shape.extend(&nsd);
                let input = ArrayD::from_shape_vec(
                    input_shape.clone(),
                    (0..input_shape.iter().cloned().product()).map(|i| (1 + i) as f32).collect(),
                )
                .unwrap();
                let block_size = Array1::from_shape_fn(sd.len(), |i| bs[i] as i32).into_dyn();
                let padding = Array2::<i32>::from_shape_fn((sd.len(), 2), |(d, locus)| {
                    if locus == 0 {
                        left_pad[d] as i32
                    } else {
                        block_size[d] - (sd[d] + left_pad[d]) as i32 % block_size[d]
                    }
                });
                (input.into(), block_size.into(), padding.into_dyn().into())
            },
        )
        .boxed()
}

proptest! {
    #[test]
    fn space_to_batch((ref i, ref bs, ref p) in space_to_batch_strat()) {
        let graph = tfpb::graph()
            .node(placeholder_f32("input"))
            .node(const_i32("block_shape", bs))
            .node(const_i32("paddings", p))
            .node(tfpb::node().name("op").op("SpaceToBatchND").input("input")
            .input("block_shape")
            .input("paddings")
            .attr("T", DtFloat)
            );
        let graph = graph.write_to_bytes().unwrap();
        let inputs = vec!(("input", i.clone()));
        compare(&graph, inputs, "op")?
    }
}

fn batch_to_space_strat() -> BoxedStrategy<(Tensor, Tensor, Tensor)> {
    use crate::tract_tensorflow::tract_hir::internal::EvalOp;
    space_to_batch_strat()
        .prop_map(|(i, bs, p)| {
            let batches: Tensor =
                tract_tensorflow::ops::nn::s2b::raw::SpaceToBatch::new(f32::datum_type())
                    .eval(tvec![i.into(), bs.clone().into(), p.clone().into()])
                    .unwrap()
                    .remove(0)
                    .into_tensor();
            (batches, bs, p)
        })
        .boxed()
}

proptest! {
    #[test]
    fn batch_to_space((ref b, ref bs, ref c) in batch_to_space_strat()) {
        let graph = tfpb::graph()
            .node(placeholder_f32("input"))
            .node(const_i32("block_shape", bs))
            .node(const_i32("crops", c))
            .node(tfpb::node().name("op").op("BatchToSpaceND").input("input")
            .input("block_shape")
            .input("crops")
            .attr("T", DtFloat)
            );
        let graph = graph.write_to_bytes().unwrap();
        let inputs = vec!(("input", b.clone()));
        compare(&graph, inputs, "op")?
    }
}

#[test]
fn space_to_batch_1() {
    let graph = tfpb::graph()
        .node(placeholder_f32("input"))
        .node(const_i32("block_shape", &Tensor::from(arr1(&[2i32, 2]))))
        .node(const_i32("paddings", &Tensor::from(arr2(&[[0i32, 0], [0, 0]]))))
        .node(
            tfpb::node()
                .name("op")
                .op("SpaceToBatchND")
                .input("input")
                .input("block_shape")
                .input("paddings")
                .attr("T", DtFloat),
        );
    let graph = graph.write_to_bytes().unwrap();
    let i = tensor4(&[[[[1.0f32], [2.0]], [[3.0], [4.0]]]]);
    let inputs = vec![("input", i)];
    compare(&graph, inputs, "op").unwrap()
}

#[test]
fn batch_to_space_1() {
    let graph = tfpb::graph()
        .node(placeholder_f32("input"))
        .node(const_i32("block_shape", &Tensor::from(arr1(&[2i32, 2]))))
        .node(const_i32("crops", &Tensor::from(arr2(&[[0i32, 0], [0, 0]]))))
        .node(
            tfpb::node()
                .name("op")
                .op("BatchToSpaceND")
                .input("input")
                .input("block_shape")
                .input("crops")
                .attr("T", DtFloat),
        );
    let graph = graph.write_to_bytes().unwrap();
    let i = tensor4(&[[[[1.0f32]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]]);
    let inputs = vec![("input", i)];
    compare(&graph, inputs, "op").unwrap()
}
