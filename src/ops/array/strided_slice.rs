use ndarray::prelude::*;
use {Matrix, Result};
use ops::{Input, Op};

pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let begin_mask = pb.get_attr()
        .get("begin_mask")
        .map(|a| a.get_i())
        .unwrap_or(0);
    let end_mask = pb.get_attr()
        .get("end_mask")
        .map(|a| a.get_i())
        .unwrap_or(0);
    let shrink_axis_mask = pb.get_attr()
        .get("shrink_axis_mask")
        .map(|a| a.get_i())
        .unwrap_or(0);
    Ok(Box::new(StridedSlice {
        begin_mask,
        end_mask,
        shrink_axis_mask,
    }))
}

#[derive(Debug, Default)]
pub struct StridedSlice {
    begin_mask: i64,
    end_mask: i64,
    shrink_axis_mask: i64,
}

impl Op for StridedSlice {
    fn eval(&self, mut inputs: Vec<Input>) -> Result<Vec<Input>> {
        let (input, begin, end, strides) = args_4!(inputs);
        let input = input.as_i32s().ok_or("Input expected as I32")?;
        let begin = begin.as_i32s().ok_or("Begin expected as I32")?;
        let end = end.as_i32s().ok_or("End expected as I32")?;
        let strides = strides.as_i32s().ok_or("Strides expected as I32")?;
        struct Dim {
            begin: i32,
            stride: i32,
            len: usize,
            shrink: bool,
        };
        let bounds: Vec<Dim> = (0..input.shape().len())
            .map(|d| {
                let max = input.shape()[d] as i32;
                // deal with negative indexing
                let b = if begin[d] >= 0 {
                    begin[d]
                } else {
                    max + begin[d]
                };
                let e = if end[d] >= 0 { end[d] } else { max + end[d] };

                // deal with shrinking
                if self.shrink_axis_mask & 1 << d != 0 {
                    return Dim {
                        begin: b,
                        stride: 1,
                        len: 1,
                        shrink: true,
                    };
                }

                // deal with begin and end masks
                let s = strides[d];
                let b = if (self.begin_mask >> d) & 1 == 1 {
                    if s.signum() > 0 {
                        0
                    } else {
                        max - 1
                    }
                } else {
                    b
                };
                let e = if (self.end_mask >> d) & 1 == 1 {
                    if s.signum() < 0 {
                        -1
                    } else {
                        max
                    }
                } else {
                    e
                };
                let len = (((s.abs() as i32 - 1) + (e - b).abs()) / s.abs()) as usize;
                Dim {
                    begin: b,
                    stride: s,
                    len,
                    shrink: false,
                }
            })
            .collect();
        //        println!("input shape: {:?}, bounds: {:?}", input.shape(), bounds);
        let shape: Vec<usize> = bounds.iter().map(|d| d.len).collect();
        let reshape: Vec<usize> = bounds.iter().filter(|d| !d.shrink).map(|d| d.len).collect();
        //        println!("output shape: {:?}", shape);
        let output = Array::from_shape_fn(shape, |coords| {
            let coord: Vec<_> = coords
                .slice()
                .iter()
                .enumerate()
                .map(|(d, i)| (*i as i32 * bounds[d].stride + bounds[d].begin) as usize)
                .collect();
            input[&*coord]
        });
        let output = output.into_shape(reshape)?;
        Ok(vec![Matrix::I32(output.into()).into()])
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use Matrix;
    use super::*;
    use ndarray::*;

    fn run<I, B, E, S>(op: StridedSlice, input: I, begin: B, end: E, strides: S) -> Matrix
    where
        I: Into<Matrix>,
        B: Into<Matrix>,
        E: Into<Matrix>,
        S: Into<Matrix>,
    {
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
                StridedSlice::default(),
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
                StridedSlice::default(),
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
                StridedSlice::default(),
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
                StridedSlice::default(),
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
            run(
                StridedSlice::default(),
                arr1(&[0, 0]),
                arr1(&[0]),
                arr1(&[-1]),
                arr1(&[1])
            ),
            Matrix::from(arr1(&[0]))
        )
    }

    #[test]
    fn strided_slice_6() {
        assert_eq!(
            run(
                StridedSlice::default(),
                arr2(&[[1, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]]),
                arr1(&[-3, -4]),
                arr1(&[-1, -1]),
                arr1(&[1, 2])
            ),
            Matrix::from(arr2(&[[1, 0], [3, 0]]))
        )
    }

    #[test]
    fn strided_slice_7() {
        assert_eq!(
            run(StridedSlice::default(),
                arr2(&[[0, 6], [0, 0]]),
                arr1(&[0]),
                arr1(&[2]),
                arr1(&[1])
            ),
            Matrix::from(arr2(&[[0, 6], [0, 0]]))
        )
    }

    #[test]
    fn strided_slice_begin_mask_1() {
        let mut op = StridedSlice::default();
        op.begin_mask = 1;
        assert_eq!(
            run(op, arr1(&[0, 1]), arr1(&[1]), arr1(&[1]), arr1(&[1])),
            Matrix::from(arr1(&[0]))
        )
    }

    #[test]
    fn strided_slice_shrink_1() {
        let mut op = StridedSlice::default();
        op.shrink_axis_mask = 1;
        assert_eq!(
            run(
                op,
                arr2(&[[0]]),
                arr1(&[0, 0]),
                arr1(&[0, 0]),
                arr1(&[1, 1])
            ),
            Matrix::I32(arr1(&[]).into_dyn())
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

    fn strided_slice_strat(
) -> BoxedStrategy<(Matrix, Matrix, Matrix, Matrix, (i32, i32, i32, i32, i32))> {
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
        ).prop_flat_map(|dims| {
            let rank = dims.iter().len();
            (
                Just(dims),
                (
                    0..(1 << rank),
                    0..(1 << rank),
                    Just(0),
                    Just(0),
                    0..(1 << rank),
                ),
            )
        })
            .prop_map(|(dims, masks)| {
                let shape = dims.iter().map(|d| d.0 as usize).collect::<Vec<_>>();
                let size: i32 = shape.iter().map(|d| *d as i32).product();
                (
                    Matrix::from(Array::from_shape_vec(shape, (0..size).collect()).unwrap()),
                    Array::from_vec(
                        dims.iter()
                            .map(|d| {
                                if d.4 {
                                    d.1 - d.0
                                } else {
                                    d.1
                                }
                            })
                            .collect(),
                    ).into(),
                    Array::from_vec(
                        dims.iter()
                            .map(|d| {
                                if d.5 {
                                    d.2 - d.0
                                } else {
                                    d.2
                                }
                            })
                            .collect(),
                    ).into(),
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
                    ).into(),
                    masks,
                )
            })
            .boxed()
    }

    proptest! {
        #[test]
        fn strided_slice((ref i, ref b, ref e, ref s, ref masks) in strided_slice_strat()) {
            // println!("trying {:?}", (i,b,e,s,masks));
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
            compare(&graph, inputs, "op")?
        }
    }
}
