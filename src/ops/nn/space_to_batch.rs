use std::marker::PhantomData;

use Result;
use super::{Input, Op};
use ndarray::prelude::*;
use matrix::Datum;

pub fn space_to_batch(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let datatype = pb.get_attr_datatype("T")?;
    Ok(boxed_new!(SpaceToBatch(datatype)()))
}

#[derive(Debug,new)]
pub struct SpaceToBatch<T:Datum>(PhantomData<T>);

impl<T:Datum> Op for SpaceToBatch<T> {
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        println!("");
        let (input, block_shape, paddings) = args_3!(inputs);
        println!("input_shape: {:?}", input.shape());
        let block_shape = block_shape.as_i32s().ok_or("block shape expected as I32")?;
        let paddings = paddings.as_i32s().ok_or("paddings expected as I32")?;

        let data = T::mat_to_view(&input)?;
        let block_size = block_shape.iter().map(|a| *a as usize).product::<usize>();
        println!("block_shape: {:?} -> {}", block_shape, block_size);
        let mut output_shape = vec!(input.shape()[0] * block_size);
        for (m,block_shape_dim) in block_shape.iter().enumerate() {
            output_shape.push(input.shape()[m+1] / block_shape[m] as usize);
        }
        output_shape.extend(&input.shape()[block_shape.len()+1..]);
        let output = unsafe { Array::zeros(output_shape) };

        /*
        // input batch
        for n in 0..input.shape()[0] {
        }
            println!("indexes: {:?}", indexes);
            let first_coord = indexes[0];
            indexes[0] /= block_size;
            let mut remaining = first_coord % block_size;
            for (m,bdim) in block_shape.to_vec().iter().enumerate().rev() {
            }
            0
        });
        */
        Ok(vec!(output.into()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use Matrix;
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
