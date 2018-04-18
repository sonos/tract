use std::marker::PhantomData;

use Result;
use super::{Input, Op};
use tfpb::types::DataType;
use matrix::Datum;

#[derive(Debug, Default, new)]
pub struct Pack<T: Datum> {
    axis: usize,
    _phantom: PhantomData<T>,
}

pub fn pack(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let dtype = pb.get_attr_datatype("T")?;
    let axis = pb.get_attr_int("axis")?;
    Ok(boxed_new!(Pack(dtype)(axis)))
}

impl<T> Op for Pack<T>
where
    T: Datum,
{
    fn eval(&self, inputs: Vec<Input>) -> Result<Vec<Input>> {
        use ndarray::Axis;
        let views = inputs
            .iter()
            .map(|m| {
                Ok(T::mat_to_view(&*m)?.insert_axis(Axis(self.axis)))
//                Ok(T::mat_to_view(&*m)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let array = ::ndarray::stack(Axis(self.axis), &*views)?;
        Ok(vec![T::array_into_mat(array).into()])
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use Matrix;
    use super::*;
    use ndarray::arr2;

    #[test]
    fn pack_0() {
        let inputs = vec!(
                Matrix::i32s(&[2], &[1,4]).unwrap().into(),
                Matrix::i32s(&[2], &[2,5]).unwrap().into(),
                Matrix::i32s(&[2], &[3,6]).unwrap().into(),
        );
        assert_eq!(Pack::<i32>::new(0).eval(inputs.clone()).unwrap().remove(0).into_matrix(),
                   Matrix::from(arr2(&[[1, 4], [2, 5], [3, 6]])));
        assert_eq!(Pack::<i32>::new(1).eval(inputs.clone()).unwrap().remove(0).into_matrix(),
                   Matrix::from(arr2(&[[1, 2, 3], [4, 5, 6]])));
    }

    #[test]
    fn pack_1() {
        let pack = Pack::<i32>::new(0);
        let input = Matrix::i32s(&[0], &[]).unwrap();
        let exp: Matrix = Matrix::i32s(&[1, 0], &[]).unwrap();
        let found = pack.eval(vec![input.into()]).unwrap();

        assert!(
            exp.close_enough(&found[0]),
            "expected: {:?} found: {:?}",
            exp,
            found[0]
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
    use proptest::collection::vec;
    use Matrix;

    fn strat() -> BoxedStrategy<(usize, Vec<Matrix>)> {
        // input rank
        (1usize..8)
            // rank, dimensions, number of inputs
            .prop_flat_map(|r| (0usize..r, vec(1usize..5, r..r+1), 1..5))
            .prop_map(|(ax, dims, n)| {
                let size = dims.iter().map(|a| *a).product::<usize>();
                let mats:Vec<Matrix> = (0..n).map(|ix| {
                    Matrix::from(Array::from_shape_vec(dims.clone(), ((ix*1000)..).take(size).collect()).unwrap())
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

}
