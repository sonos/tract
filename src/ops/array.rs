use ndarray::prelude::*;

use {Matrix, Result};
use super::{Input, Op, OpRegister};

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("ConcatV2", ConcatV2::build);
    reg.insert("ExpandDims", ExpandDims::build);
    reg.insert("Identity", Identity::build);
    reg.insert("Placeholder", Placeholder::build);
    reg.insert("Reshape", Reshape::build);
    reg.insert("Squeeze", Squeeze::build);
    reg.insert("StridedSlice", StridedSlice::build);
}

#[derive(Debug)]
pub struct ConcatV2 {
    n: usize,
}

impl ConcatV2 {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(ConcatV2 {
            n: pb.get_attr().get("N").unwrap().get_i() as _,
        }))
    }
}

impl Op for ConcatV2 {
    fn eval(&self, inputs: Vec<Input>) -> Result<Vec<Input>> {
        let axis: i32 = inputs[self.n]
            .as_i32s()
            .ok_or("Expected a i32 matrix")?
            .iter()
            .next()
            .unwrap()
            .clone();
        let mats: Vec<_> = inputs[0..self.n]
            .iter()
            .map(|mat| mat.as_f32s().unwrap().view())
            .collect();
        let result = ::ndarray::stack(Axis(axis as usize), &*mats)?;
        let result = Matrix::from(result);
        Ok(vec![result.into()])
    }
}

#[derive(Debug)]
pub struct ExpandDims;

impl ExpandDims {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(ExpandDims))
    }
}

impl Op for ExpandDims {
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        let (data, dims) = args_2!(inputs);
        let data = data.into_matrix()
            .take_f32s()
            .ok_or("Expected a f32 matrix")?;
        let dims = dims.as_i32s().ok_or("Expected a i32 matrix")?;
        let mut shape = data.shape().to_vec();
        for d in dims.iter() {
            if *d >= 0 {
                shape.insert(*d as usize, 1);
            } else {
                Err(format!("unimplemented ExpandDims with negative parameter"))?
            }
        }
        Ok(vec![Matrix::from(data.into_shape(shape)?).into()])
    }
}

#[derive(Debug)]
pub struct Identity;

impl Identity {
    pub fn build(_: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Identity))
    }
}

impl Op for Identity {
    fn eval(&self, inputs: Vec<Input>) -> Result<Vec<Input>> {
        Ok(inputs)
    }
}

#[derive(Debug)]
pub struct Placeholder;

impl Placeholder {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Placeholder))
    }
}

impl Op for Placeholder {
    fn eval(&self, _inputs: Vec<Input>) -> Result<Vec<Input>> {
        panic!("Placeholder should not get evaluated")
    }
}

#[derive(Debug)]
pub struct Reshape {}

impl Reshape {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(Reshape {}))
    }
}

impl Op for Reshape {
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        let (input, dims) = args_2!(inputs);
        let input = input
            .into_matrix()
            .take_f32s()
            .ok_or("Expected a f32 matrix")?;
        let mut dims: Vec<i32> = dims.as_i32s()
            .ok_or("Expected a i32 matrix")?
            .iter()
            .cloned()
            .collect();
        if dims.contains(&-1) {
            let prod: i32 = dims.iter().map(|a| *a).filter(|a| *a != -1i32).product();
            for a in dims.iter_mut() {
                if *a == -1 {
                    *a = input.len() as i32 / prod;
                }
            }
        }
        let dims: Vec<usize> = dims.into_iter().map(|a| a as usize).collect();
        Ok(vec![
            Matrix::from(input.into_shape(&*dims)?.into_dyn()).into(),
        ])
    }
}

#[derive(Debug)]
pub struct Squeeze {
    dims: Vec<isize>,
}

impl Squeeze {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        let dims = pb.get_attr()
            .get("squeeze_dims")
            .ok_or("Squeeze expect squeeze_dims attribute")?;
        let mut dims: Vec<isize> = dims.get_list()
            .get_i()
            .into_iter()
            .map(|x| *x as isize)
            .collect();
        dims.sort();
        dims.reverse();
        Ok(Box::new(Squeeze { dims }))
    }
}

impl Op for Squeeze {
    fn eval(&self, inputs: Vec<Input>) -> Result<Vec<Input>> {
        let data = inputs[0].as_f32s().ok_or("Expect input #0 to be f32")?;
        let mut shape = data.shape().to_vec();
        for d in &self.dims {
            if *d >= 0 {
                shape.remove(*d as usize);
            } else {
                Err(format!("unimplemented Squeeze with negative parameter"))?
            }
        }
        Ok(vec![Matrix::from(data.clone().into_shape(shape)?).into()])
    }
}

#[derive(Debug)]
pub struct StridedSlice {}

impl StridedSlice {
    pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        Ok(Box::new(StridedSlice {}))
    }
}

impl Op for StridedSlice {
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        let (input, begin, end, strides) = args_4!(inputs);
        let input = input.as_i32s().ok_or("Input expected as I32")?;
        let begin = begin.as_i32s().ok_or("Begin expected as I32")?;
        let end = end.as_i32s().ok_or("End expected as I32")?;
        let strides = strides.as_i32s().ok_or("Strides expected as I32")?;
        let shape: Vec<usize> = (0..input.shape().len())
            .map(|d| ((end[d] - begin[d]) / strides[d]) as usize)
            .collect();
        let output = Array::from_shape_fn(shape, |coords| {
            let coord: Vec<_> = coords
                .slice()
                .iter()
                .enumerate()
                .map(|(d, i)| {
                    let signed = *i as i32 * strides[d] + begin[d];
                    let pos = if signed >= 0 {
                        signed
                    } else {
                        input.shape()[d] as i32 + signed
                    };
                    pos as usize
                })
                .collect();
            input[&*coord]
        });
        println!("output: {:?}", output);
        Ok(vec![Matrix::I32(output.into()).into()])
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use Matrix;
    use super::*;

    // https://www.tensorflow.org/api_docs/python/tf/strided_slice
    #[test]
    fn strided_slice_1() {
        let input: Matrix = ::ndarray::arr3(&[
            [[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]],
        ]).into();
        let begin: Matrix = ::ndarray::arr1(&[1, 0, 0]).into();
        let end: Matrix = ::ndarray::arr1(&[2, 1, 3]).into();
        let strides: Matrix = ::ndarray::arr1(&[1, 1, 1]).into();
        let op = StridedSlice {};
        let output = op.eval(vec![input.into(), begin.into(), end.into(), strides.into()])
            .unwrap();
        let exp: Matrix = ::ndarray::arr3(&[[[3, 3, 3]]]).into();
        assert_eq!(exp, *output[0]);
    }

    #[test]
    fn strided_slice_2() {
        let input: Matrix = ::ndarray::arr3(&[
            [[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]],
        ]).into();
        let begin: Matrix = ::ndarray::arr1(&[1, 0, 0]).into();
        let end: Matrix = ::ndarray::arr1(&[2, 2, 3]).into();
        let strides: Matrix = ::ndarray::arr1(&[1, 1, 1]).into();
        let op = StridedSlice {};
        let output = op.eval(vec![input.into(), begin.into(), end.into(), strides.into()])
            .unwrap();
        let exp: Matrix = ::ndarray::arr3(&[[[3, 3, 3], [4, 4, 4]]]).into();
        assert_eq!(exp, *output[0]);
    }

    #[test]
    fn strided_slice_3() {
        let input: Matrix = ::ndarray::arr3(&[
            [[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]],
        ]).into();
        let begin: Matrix = ::ndarray::arr1(&[1, -1, 0]).into();
        let end: Matrix = ::ndarray::arr1(&[2, -3, 3]).into();
        let strides: Matrix = ::ndarray::arr1(&[1, -1, 1]).into();
        let op = StridedSlice {};
        let output = op.eval(vec![input.into(), begin.into(), end.into(), strides.into()])
            .unwrap();
        let exp: Matrix = ::ndarray::arr3(&[[[4, 4, 4], [3, 3, 3]]]).into();
        assert_eq!(exp, *output[0]);
    }

    #[test]
    fn strided_slice_4() {
        let input: Matrix = ::ndarray::arr3(&[
            [[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]],
        ]).into();
        let begin: Matrix = ::ndarray::arr1(&[1, 0, 0]).into();
        let end: Matrix = ::ndarray::arr1(&[2, 2, 4]).into();
        let strides: Matrix = ::ndarray::arr1(&[1, 1, 2]).into();
        let op = StridedSlice {};
        let output = op.eval(vec![input.into(), begin.into(), end.into(), strides.into()])
            .unwrap();
        let exp: Matrix = ::ndarray::arr3(&[[[3, 3], [4, 4]]]).into();
        assert_eq!(exp, *output[0]);
    }
}

#[cfg(all(test, feature = "tensorflow"))]
pub mod proptests {
    #![allow(non_snake_case)]
    use proptest::prelude::*;
    use ndarray::prelude::*;
    use protobuf::core::Message;
    use tfpb;
    use tfpb::types::DataType::DT_FLOAT;
    use ops::proptests::*;
    use Matrix;

    fn strided_slice_strat() -> BoxedStrategy<(Matrix, Matrix, Matrix, Matrix)> {
        ::proptest::collection::vec(0usize..10, 1usize..8)
            .prop_flat_map(|shape| {
                let dims = shape.len();
                let items:usize = shape.iter().product();
                (Just(shape),
                 ::proptest::collection::vec(-100i32..100, items..items+1),
                 ::proptest::collection::vec(-10i32..10, dims..dims+1),
                 ::proptest::collection::vec(-10i32..10, dims..dims+1),
                 ::proptest::collection::vec(-10i32..10, dims..dims+1),
                )
            }).prop_map(|(shape, input, begin, end, stride)| {
                let shape = shape.into_iter().map(|s| s as usize).collect::<Vec<_>>();
                (Array::from_vec(input).into_shape(&*shape).unwrap().into(),
                Array::from_vec(begin).into(),
                Array::from_vec(end).into(),
                Array::from_vec(stride).into())
            }).boxed()
    }

    proptest! {
        #[test]
        fn strided_slice((ref i, ref b, ref e, ref s) in strided_slice_strat()) {
            let graph = tfpb::graph()
                .node(placeholder("input"))
                .node(placeholder("begin"))
                .node(placeholder("end"))
                .node(placeholder("stride"))
                .node(tfpb::node().name("op").input("input").input("begin").input("end").input("stride").op("StridedSlice")
                ).write_to_bytes().unwrap();

            let inputs = vec!(("input", i.clone()),("begin", b.clone()), ("end", e.clone()), ("stride", s.clone()));
            let expected = ::tf::for_slice(&graph)?.run(inputs.clone(), "op");
            prop_assume!(expected.is_ok());
            let expected = expected.unwrap();
            let found = ::Model::for_reader(&*graph)?.run_with_names(inputs, "op").unwrap();
            prop_assert!(expected[0].close_enough(&found[0]), "expected: {:?} found: {:?}", expected, found)
        }
    }
}
