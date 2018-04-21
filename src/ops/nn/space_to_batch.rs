use std::marker::PhantomData;

use Result;
use super::{Input, Op};
use matrix::Datum;

pub fn space_to_batch_nd(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let datatype = pb.get_attr_datatype("T")?;
    Ok(boxed_new!(SpaceToBatch(datatype)()))
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

        let mut permuted_axis:Vec<_> = (0..block_shape.len()).map(|x| 2*x+2).collect();
        permuted_axis.push(0);
        permuted_axis.extend((0..block_shape.len()).map(|x| 2*x+1));
        permuted_axis.extend((block_shape.len()*2+1)..data.ndim());
        let data = data.permuted_axes(permuted_axis);
        let data:Vec<T> = data.into_iter().map(|x| *x).collect();
        let data = ::ndarray::ArrayD::from_shape_vec(final_shape, data)?;

        Ok(vec!(T::array_into_mat(data).into()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use ndarray::*;

    pub fn arr4<A,V,U,T>(xs: &[V]) -> Array4<A>
        where V: FixedInitializer<Elem=U> + Clone,
               U: FixedInitializer<Elem=T> + Clone,
               T: FixedInitializer<Elem=A> + Clone,
               A: Clone,
    {
        let mut xs = xs.to_vec();
        let dim = Ix4(xs.len(), V::len(), U::len(), T::len());
        let ptr = xs.as_mut_ptr();
        let len = xs.len();
        let cap = xs.capacity();
        let expand_len = len * V::len() * U::len() * T::len();
        ::std::mem::forget(xs);
        unsafe {
            let v = if ::std::mem::size_of::<A>() == 0 {
                Vec::from_raw_parts(ptr as *mut A, expand_len, expand_len)
            } else if V::len() == 0 || U::len() == 0 || T::len() == 0 {
                Vec::new()
            } else {
                let expand_cap = cap * V::len() * U::len() * T::len();
                Vec::from_raw_parts(ptr as *mut A, expand_len, expand_cap)
            };
            ArrayBase::from_shape_vec_unchecked(dim, v)
        }
    }

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
}

#[cfg(all(test, feature = "tensorflow"))]
pub mod proptests {
    #![allow(non_snake_case)]
    use proptest::prelude::*;
    use ndarray::prelude::*;
    use protobuf::core::Message;
    use tfpb;
    use tfpb::types::DataType::DT_INT32;
    use ops::proptests::*;
    use Matrix;


/*
    fn space_to_batch_strat() -> BoxedStrategy<(Matrix, Matrix, Matrix)> {
        (Just(
    }
    proptest! {
        #[test]
        fn space_to_batch((ref i, ref bs, ref p) in strided_slice_strat()) {
            let graph = tfpb::graph()
                .node(placeholder_f32("input"))
                .node(placeholder_i32("block_shape"));
                .node(placeholder_i32("paddings"))
                .node(tfpb::node().name("op").op("SpaceToBatchND").input("input")
                .input("block_shape")
                .input("paddings")
                );

            let inputs = vec!(("input", i), ("paddings", p), ("block_shape", bs));
            compare(&graph, inputs, "op")?
        }
    }
    */
}
