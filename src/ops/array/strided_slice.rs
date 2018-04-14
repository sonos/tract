use ndarray::prelude::*;
use {Matrix, Result};
use ops::{Input, Op};

pub fn build(_pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    Ok(Box::new(StridedSlice))
}

#[derive(Debug)]
pub struct StridedSlice;

impl Op for StridedSlice {
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        let (input, begin, end, strides) = args_4!(inputs);
        let input = input.as_i32s().ok_or("Input expected as I32")?;
        let begin = begin.as_i32s().ok_or("Begin expected as I32")?;
        let end = end.as_i32s().ok_or("End expected as I32")?;
        let strides = strides.as_i32s().ok_or("Strides expected as I32")?;
        let shape: Vec<usize> = (0..input.shape().len())
            .map(|d| {
                let e = if end[d] >= 0 {
                    end[d]
                } else {
                    input.shape()[d] as i32 + end[d]
                };
                let b = if begin[d] >= 0 {
                    begin[d]
                } else {
                    input.shape()[d] as i32 + begin[d]
                };
                (((strides[d].abs() as i32 - 1) + (e - b).abs()) / strides[d].abs()) as usize
            })
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
        Ok(vec![Matrix::I32(output.into()).into()])
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use Matrix;
    use super::*;
    use ndarray::*;

    fn run<I, B, E, S>(input: I, begin: B, end: E, strides: S) -> Matrix
    where
        I: Into<Matrix>,
        B: Into<Matrix>,
        E: Into<Matrix>,
        S: Into<Matrix>,
    {
        let op = StridedSlice {};
        op.eval(vec![
            input.into().into(),
            begin.into().into(),
            end.into().into(),
            strides.into().into(),
        ]).unwrap()
            .pop()
            .unwrap()
            .into_matrix()
    }

    // https://www.tensorflow.org/api_docs/python/tf/strided_slice
    #[test]
    fn strided_slice_1() {
        assert_eq!(
            run(
                arr3(&[
                    [[1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4]],
                    [[5, 5, 5], [6, 6, 6]],
                ]),
                arr1(&[1, 0, 0]),
                arr1(&[2, 1, 3]),
                arr1(&[1, 1, 1])
            ),
            Matrix::from(arr3(&[[[3, 3, 3]]])),
        );
    }

    #[test]
    fn strided_slice_2() {
        assert_eq!(
            run(
                arr3(&[
                    [[1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4]],
                    [[5, 5, 5], [6, 6, 6]],
                ]),
                arr1(&[1, 0, 0]),
                arr1(&[2, 2, 3]),
                arr1(&[1, 1, 1])
            ),
            Matrix::from(arr3(&[[[3, 3, 3], [4, 4, 4]]])),
        );
    }

    #[test]
    fn strided_slice_3() {
        assert_eq!(
            run(
                arr3(&[
                    [[1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4]],
                    [[5, 5, 5], [6, 6, 6]],
                ]),
                arr1(&[1, -1, 0]),
                arr1(&[2, -3, 3]),
                arr1(&[1, -1, 1])
            ),
            Matrix::from(arr3(&[[[4, 4, 4], [3, 3, 3]]])),
        );
    }

    #[test]
    fn strided_slice_4() {
        assert_eq!(
            run(
                arr3(&[
                    [[1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4]],
                    [[5, 5, 5], [6, 6, 6]],
                ]),
                arr1(&[1, 0, 0]),
                arr1(&[2, 2, 4]),
                arr1(&[1, 1, 2])
            ),
            Matrix::from(arr3(&[[[3, 3], [4, 4]]])),
        );
    }

    #[test]
    fn strided_slice_5() {
        assert_eq!(
            run(arr1(&[0, 0]), arr1(&[0]), arr1(&[-1]), arr1(&[1])),
            Matrix::from(arr1(&[0]))
        )
    }

    #[test]
    fn strided_slice_6() {
        assert_eq!(
            run(
                arr2(&[[1, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]]),
                arr1(&[-3, -4]),
                arr1(&[-1, -1]),
                arr1(&[1, 2])
            ),
            Matrix::from(arr2(&[[1, 0], [3, 0]]))
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

    fn strided_slice_strat() -> BoxedStrategy<(Matrix, Matrix, Matrix, Matrix)> {
        ::proptest::collection::vec(
            (1..10).prop_flat_map(|n| {
                (
                    Just(n),
                    0..n,
                    0..n,
                    if n <= 2 { 1..2 } else { (1..n) },
                    any::<bool>(),
                    any::<bool>(),
                )
            }),
            1..4,
        ).prop_flat_map(|dims| {
            let shape = dims.iter().map(|d| d.0 as usize).collect::<Vec<_>>();
            let items: usize = shape.iter().product();
            (
                Just(dims),
                ::proptest::collection::vec(-100i32..100, items..items + 1),
            )
        })
            .prop_map(|(dims, input)| {
                let shape = dims.iter().map(|d| d.0 as usize).collect::<Vec<_>>();
                (
                    Array::from_vec(input).into_shape(&*shape).unwrap().into(),
                    Array::from_vec(
                        dims.iter()
                            .map(|d| {
                                if d.4 {
                                    d.1
                                } else {
                                    d.1 - d.0
                                }
                            })
                            .collect(),
                    ).into(),
                    Array::from_vec(
                        dims.iter()
                            .map(|d| {
                                if d.5 {
                                    d.2
                                } else {
                                    d.2 - d.0
                                }
                            })
                            .collect(),
                    ).into(),
                    Array::from_vec(
                        dims.iter()
                            .map(|d| {
                                if d.2 == d.1 {
                                    1
                                } else {
                                    d.3 as i32 * (d.2 as i32 - d.1 as i32).signum()
                                }
                            })
                            .collect(),
                    ).into(),
                )
            })
            .boxed()
    }

    proptest! {
        #[test]
        fn strided_slice((ref i, ref b, ref e, ref s) in strided_slice_strat()) {
            let graph = tfpb::graph()
                .node(placeholder_i32("input"))
                .node(placeholder_i32("begin"))
                .node(placeholder_i32("end"))
                .node(placeholder_i32("stride"))
                .node(tfpb::node().name("op").attr("T", DT_INT32).attr("Index", DT_INT32).input("input").input("begin").input("end").input("stride").op("StridedSlice")
                ).write_to_bytes().unwrap();

            let inputs = vec!(("input", i.clone()),("begin", b.clone()), ("end", e.clone()), ("stride", s.clone()));
            compare(&graph, inputs, "op")?
        }
    }
}
