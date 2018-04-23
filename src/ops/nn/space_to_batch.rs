use std::marker::PhantomData;

use Result;
use super::{Input, Op};
use matrix::Datum;

pub fn space_to_batch_nd(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let datatype = pb.get_attr_datatype("T")?;
    Ok(boxed_new!(SpaceToBatch(datatype)()))
}
pub fn batch_to_space_nd(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let datatype = pb.get_attr_datatype("T")?;
    Ok(boxed_new!(BatchToSpace(datatype)()))
}

#[derive(Debug,new)]
pub struct SpaceToBatch<T:Datum>(PhantomData<T>);

impl<T:Datum> Op for SpaceToBatch<T> {
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        let (input, block_shape, paddings) = args_3!(inputs);
        let block_shape = block_shape.as_i32s().ok_or("block shape expected as I32")?;
        let paddings = paddings.as_i32s().ok_or("paddings expected as I32")?;
        let mut data = T::mat_into_array(input.into_matrix())?;

        for (ix, pad) in paddings.outer_iter().enumerate() {
            if pad[0] != 0 {
                let mut pad_shape = data.shape().to_vec();
                pad_shape[ix+1] = pad[0] as usize;
                let tmp = ::ndarray::stack(::ndarray::Axis(ix+1),
                    &[::ndarray::ArrayD::zeros(pad_shape).view(), data.view()])?;
                data = tmp;
            }
            if pad[1] != 0 {
                let mut pad_shape = data.shape().to_vec();
                pad_shape[ix+1] = pad[1] as usize;
                let tmp = ::ndarray::stack(::ndarray::Axis(ix+1),
                    &[data.view(), ::ndarray::ArrayD::zeros(pad_shape).view()])?;
                data = tmp;
            }
        }
        let mut reshaped = vec!(data.shape()[0]);
        let block_size = block_shape.iter().map(|a| *a as usize).product::<usize>();
        let mut final_shape = vec!(block_size*data.shape()[0]);
        for (m, &block_shape_dim) in block_shape.iter().enumerate() {
            reshaped.push(data.shape()[m+1] / block_shape_dim as usize);
            reshaped.push(block_shape_dim as usize);
            final_shape.push(data.shape()[m+1] / block_shape_dim as usize);
        }
        reshaped.extend(&data.shape()[block_shape.len()+1..]);
        final_shape.extend(&data.shape()[block_shape.len()+1..]);
        let data = data.into_shape(reshaped)?;

        let mut permuted_axes:Vec<_> = (0..block_shape.len()).map(|x| 2*x+2).collect();
        permuted_axes.push(0);
        permuted_axes.extend((0..block_shape.len()).map(|x| 2*x+1));
        permuted_axes.extend((block_shape.len()*2+1)..data.ndim());
        let data = data.permuted_axes(permuted_axes);
        let data:Vec<T> = data.into_iter().map(|x| *x).collect();
        let data = ::ndarray::ArrayD::from_shape_vec(final_shape, data)?;

        Ok(vec!(T::array_into_mat(data).into()))
    }
}

#[derive(Debug,new)]
pub struct BatchToSpace<T:Datum>(PhantomData<T>);

impl<T:Datum> Op for BatchToSpace<T> {
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        use ndarray::*;
        let (input, block_shape, crops) = args_3!(inputs);
        let block_shape = block_shape.as_i32s().ok_or("block shape expected as I32")?;
        let crops = crops.as_i32s().ok_or("crops expected as I32")?;
        let data = T::mat_into_array(input.into_matrix())?;
        let input_shape = data.shape().to_vec();
        let crops = crops.clone().into_shape((block_shape.len(), 2))?;

        let block_size = block_shape.iter().map(|a| *a as usize).product::<usize>();

        // block_dim_1 .. block_dim_n, batches/bloc_size, dim_1, .. dim_n, chan_1, .., chan_n
        let mut unflatten_blocked_shape = vec!();
        unflatten_blocked_shape.extend(block_shape.iter().map(|a| *a as usize));
        let batches = data.shape()[0]/block_size;
        unflatten_blocked_shape.push(batches);
        unflatten_blocked_shape.extend(&data.shape()[1..]);
        let data = data.into_shape(&*unflatten_blocked_shape)?;
        let mut permuted_axes = vec!(block_shape.len());
        let mut padded_shape = vec!(batches);
        for i in 0..block_shape.shape()[0] {
            permuted_axes.push(block_shape.len()+1+i);
            permuted_axes.push(i);
            padded_shape.push(block_shape[i] as usize *input_shape[i+1]);
        }
        permuted_axes.extend((1+block_shape.len()*2)..data.ndim());
        padded_shape.extend(&input_shape[1+block_shape.len()..]);
        let data = data.permuted_axes(permuted_axes);
        let data:Vec<T> = data.into_iter().map(|x| *x).collect();
        let data = ::ndarray::ArrayD::from_shape_vec(padded_shape, data)?;
        let mut data = data;
        for (i, crop) in crops.outer_iter().enumerate() {
            if crop[0] != 0 || crop[1] != 0 {
                let end = data.shape()[1+i] as usize;
                let range = (crop[0] as usize)..(end-crop[1] as usize);
                data = data.slice_axis(Axis(i+1), range.into()).map(|x| *x).to_owned();
            }
        }
        Ok(vec!(T::array_into_mat(data).into()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use ndarray::*;
    use ops::nn::arr4;

    // https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd
    #[test]
    fn space_to_batch_nd_1() {
        assert_eq!(
            SpaceToBatch::<i32>::new().eval(vec!(
                arr4(&[[[[1], [2]], [[3], [4]]]]).into(),
                arr1(&[2, 2]).into(),
                arr2(&[[0, 0],[0, 0]]).into())
            ).unwrap(),
            vec!(arr4(&[[[[1i32]]], [[[2]]], [[[3]]], [[[4]]]]).into()),
        )
    }

    #[test]
    fn space_to_batch_nd_2() {
        assert_eq!(
            SpaceToBatch::<i32>::new().eval(vec!(
                arr4(&[[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).into(),
                arr1(&[2, 2]).into(),
                arr2(&[[0, 0],[0, 0]]).into())
            ).unwrap(),
            vec!(arr4(&[[[[1i32, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]).into()),
        )
    }

    #[test]
    fn space_to_batch_nd_3() {
        assert_eq!(
            SpaceToBatch::<i32>::new().eval(vec!(
                arr4(&[[[[1], [2], [3], [4]],
                        [[5], [6], [7], [8]],
                        [[9], [10], [11], [12]],
                        [[13], [14], [15], [16]]]]).into(),
                arr1(&[2, 2]).into(),
                arr2(&[[0, 0],[0, 0]]).into())
            ).unwrap(),
            vec!(arr4(&[[[[1], [3]], [[9], [11]]],
                        [[[2], [4]], [[10], [12]]],
                        [[[5], [7]], [[13], [15]]],
                        [[[6], [8]], [[14], [16]]]]).into()),
        )
    }

    #[test]
    fn space_to_batch_nd_4() {
        assert_eq!(
            SpaceToBatch::<i32>::new().eval(vec!(
                arr4(&[[[[1], [2], [3], [4]],
                        [[5], [6], [7], [8]]],
                        [[[9], [10], [11], [12]],
                        [[13], [14], [15], [16]]]]
                        ).into(),
                arr1(&[2, 2]).into(),
                arr2(&[[0, 0],[2, 0]]).into())
            ).unwrap(),
            vec!(arr4(&[[[[0], [1], [3]]],
                        [[[0], [9], [11]]],
                        [[[0], [2], [4]]],
                        [[[0], [10], [12]]],
                        [[[0], [5], [7]]],
                        [[[0], [13], [15]]],
                        [[[0], [6], [8]]],
                        [[[0], [14], [16]]]]).into()),
        )
    }

    #[test]
    fn batch_to_space_nd_1() {
        assert_eq!(
            BatchToSpace::<i32>::new().eval(vec!(
                arr4(&[[[[1]]], [[[2]]], [[[3]]], [[[4]]]]).into(),
                arr1(&[2, 2]).into(),
                arr2(&[[0, 0], [0, 0]]).into())
            ).unwrap(),
            vec!(arr4(&[[[[1], [2]], [[3], [4]]]]).into())
        )
    }

    #[test]
    fn batch_to_space_nd_2() {
        assert_eq!(
            BatchToSpace::<i32>::new().eval(vec!(
                arr4(&[[[[1i32, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]).into(),
                arr1(&[2, 2]).into(),
                arr2(&[[0, 0], [0, 0]]).into())
            ).unwrap(),
            vec!(arr4(&[[[[1i32, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).into())
        )
    }

    #[test]
    fn batch_to_space_nd_3() {
        assert_eq!(
            BatchToSpace::<i32>::new().eval(vec!(
                arr4(&[[[[1i32], [3]], [[9], [11]]], [[[2], [4]], [[10], [12]]], [[[5], [7]], [[13], [15]]], [[[6], [8]], [[14], [16]]]]).into(),
                arr1(&[2, 2]).into(),
                arr2(&[[0, 0], [0, 0]]).into())
            ).unwrap(),
            vec!(arr4(&[[[[1i32], [2], [3], [4]], [[5], [6], [7], [8]], [[9], [10], [11], [12]], [[13], [14], [15], [16]]]]).into())
        )
    }

    #[test]
    fn batch_to_space_nd_4() {
        assert_eq!(
            BatchToSpace::<i32>::new().eval(vec!(
                arr4(&[[[[0i32], [1], [3]]], [[[0], [9], [11]]], [[[0], [2], [4]]], [[[0], [10], [12]]], [[[0], [5], [7]]], [[[0], [13], [15]]], [[[0], [6], [8]]], [[[0], [14], [16]]]]).into(),
                arr1(&[2, 2]).into(),
                arr2(&[[0, 0], [2, 0]]).into())
            ).unwrap(),
            vec!(arr4(&[[[[1], [2], [3], [4]], [[5], [6], [7], [8]]], [[[9], [10], [11], [12]], [[13], [14], [15], [16]]]]).into())
        )
    }
}

#[cfg(all(test, feature = "tensorflow"))]
pub mod proptests {
    #![allow(non_snake_case)]
    use proptest::prelude::*;
    use ndarray::prelude::*;
    use protobuf::core::Message;
    use tfpb;
    use tfpb::types::DataType::*;
    use ops::proptests::*;
    use Matrix;
    use ops::nn::arr4;

    fn space_to_batch_strat() -> BoxedStrategy<(Matrix, Matrix, Matrix)> {
        use proptest::collection::vec;
        (1usize..4, vec(1usize..8, 1usize..4), vec(1usize..8, 1usize..4))
            .prop_flat_map(|(b, spatial_dims, non_spatial_dims)| {
            (Just(b), Just(spatial_dims.clone()), Just(non_spatial_dims),
            vec(1usize..4, spatial_dims.len()..spatial_dims.len()+1),
            vec(0usize..4, spatial_dims.len()..spatial_dims.len()+1),
            )
        })
        .prop_filter("block < input", |&(_, ref sd, _, ref bs, _)|
            bs.iter().zip(sd.iter()).all(|(bs,is)| bs <= is)
        )
        .prop_map(|(b, sd, nsd, bs, left_pad):(usize, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>)| {
            let mut input_shape = vec!(b);
            input_shape.extend(&sd);
            input_shape.extend(&nsd);
            let input = ArrayD::from_shape_vec(input_shape.clone(),
                  (0..input_shape.iter().cloned().product()).map(|i| (1 + i) as f32).collect()).unwrap();
            let block_size = Array1::from_shape_fn(sd.len(), |i| bs[i] as i32).into_dyn();
            let padding = Array2::<i32>::from_shape_fn((sd.len(), 2),
             |(d, locus)| (if locus == 0 { left_pad[d] as i32 } else { block_size[d] - (sd[d] + left_pad[d]) as i32 % block_size[d] } )
            );
            (input.into(), block_size.into(), padding.into_dyn().into())
        }).boxed()
    }

    proptest! {
        #[test]
        fn space_to_batch((ref i, ref bs, ref p) in space_to_batch_strat()) {
            let graph = tfpb::graph()
                .node(placeholder_f32("input"))
                .node(placeholder("block_shape", DT_INT32, tensor_shape(bs.shape())))
                .node(placeholder_i32("paddings"))
                .node(tfpb::node().name("op").op("SpaceToBatchND").input("input")
                .input("block_shape")
                .input("paddings")
                .attr("T", DT_FLOAT)
                );
            let graph = graph.write_to_bytes()?;
            let inputs = vec!(("input", i.clone()), ("block_shape", bs.clone()), ("paddings", p.clone()));
            compare(&graph, inputs, "op")?
        }
    }

    fn batch_to_space_strat() -> BoxedStrategy<(Matrix, Matrix, Matrix)> {
        space_to_batch_strat().prop_map(|(i, bs, p)| {
            use ops::Op;
            let batches:Matrix = super::SpaceToBatch::<f32>::new().eval(vec!(i.into(), bs.clone().into(), p.clone().into())).unwrap().remove(0).into_matrix();
            (batches, bs, p)
        }).boxed()
    }

    proptest! {
        #[test]
        fn batch_to_space((ref b, ref bs, ref c) in batch_to_space_strat()) {
            let graph = tfpb::graph()
                .node(placeholder_f32("input"))
                .node(placeholder("block_shape", DT_INT32, tensor_shape(bs.shape())))
                .node(placeholder_i32("crops"))
                .node(tfpb::node().name("op").op("BatchToSpaceND").input("input")
                .input("block_shape")
                .input("crops")
                .attr("T", DT_FLOAT)
                );
            let graph = graph.write_to_bytes()?;
            let inputs = vec!(("input", b.clone()), ("block_shape", bs.clone()), ("crops", c.clone()));
            compare(&graph, inputs, "op")?
        }
    }

    #[test]
    fn space_to_batch_1() {
        use ndarray::*;
        let graph = tfpb::graph()
            .node(placeholder_f32("input"))
            .node(placeholder("block_shape", DT_INT32, tensor_shape(&[2])))
            .node(placeholder_i32("paddings"))
            .node(tfpb::node().name("op").op("SpaceToBatchND")
                .input("input")
                .input("block_shape")
                .input("paddings")
                .attr("T", DT_FLOAT)
            );
        let graph = graph.write_to_bytes().unwrap();
        let i = arr4(&[[[[1.0f32], [2.0]], [[3.0], [4.0]]]]).into();
        let bs = arr1(&[2, 2]).into();
        let p = arr2(&[[0, 0],[0, 0]]).into();
        let inputs = vec!(("input", i), ("block_shape", bs), ("paddings", p));
        compare(&graph, inputs, "op").unwrap()
    }

    #[test]
    fn batch_to_space_1() {
        use ndarray::*;
        let graph = tfpb::graph()
            .node(placeholder_f32("input"))
            .node(placeholder("block_shape", DT_INT32, tensor_shape(&[2])))
            .node(placeholder_i32("crops"))
            .node(tfpb::node().name("op").op("BatchToSpaceND")
                .input("input")
                .input("block_shape")
                .input("crops")
                .attr("T", DT_FLOAT)
            );
        let graph = graph.write_to_bytes().unwrap();
        let i = arr4(&[[[[1.0f32]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]]).into();
        let bs = arr1(&[2, 2]).into();
        let p = arr2(&[[0, 0],[0, 0]]).into();
        let inputs = vec!(("input", i), ("block_shape", bs), ("crops", p));
        compare(&graph, inputs, "op").unwrap()
    }
}
