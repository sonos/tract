use std::collections::HashMap;
use std::marker::PhantomData;

use analyser::interface::*;
use ndarray::prelude::*;
use ops::prelude::*;
use tensor::Datum;
use Result;

pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
    let begin_mask = pb.get_attr_opt_int("begin_mask")?.unwrap_or(0);
    let end_mask = pb.get_attr_opt_int("end_mask")?.unwrap_or(0);
    let shrink_axis_mask = pb.get_attr_opt_int("shrink_axis_mask")?.unwrap_or(0);
    let datum_type = pb.get_attr_datum_type("T")?;
    Ok(boxed_new!(StridedSlice(datum_type)(
        begin_mask,
        end_mask,
        shrink_axis_mask
    )))
}

#[derive(Debug, Clone)]
struct StrideSliceBuffer {
    skip: Option<usize>
}
impl OpBuffer for StrideSliceBuffer {}

#[derive(Debug, Default, Clone, new)]
pub struct StridedSlice<T: Datum> {
    begin_mask: i64,
    end_mask: i64,
    shrink_axis_mask: i64,
    _phantom: PhantomData<T>,
}

#[derive(Debug)]
struct Dim {
    begin: i32,
    stride: i32,
    len: usize,
    shrink: bool,
}

impl<T: Datum> StridedSlice<T> {
    fn must_shrink(&self, ix: usize) -> bool {
        self.shrink_axis_mask & (1 << ix) != 0
    }
    fn ignore_begin(&self, ix: usize) -> bool {
        self.begin_mask & (1 << ix) != 0
    }
    fn ignore_end(&self, ix: usize) -> bool {
        self.end_mask & (1 << ix) != 0
    }
    fn prepare_one_dim(
        &self,
        ix: usize,
        dim: usize,
        begin: &ArrayView1<i32>,
        end: &ArrayView1<i32>,
        strides: &ArrayView1<i32>,
    ) -> Dim {
        let dim = dim as i32;
        // deal with too small dim begin/end/stride for input rank
        if ix >= begin.len() {
            return Dim {
                begin: 0,
                stride: 1,
                len: dim as usize,
                shrink: false,
            };
        }

        // deal with negative indexing
        let b = if begin[ix] >= 0 {
            begin[ix]
        } else {
            dim + begin[ix]
        };
        let e = if end[ix] >= 0 { end[ix] } else { dim + end[ix] };

        // deal with shrinking
        if self.must_shrink(ix) {
            return Dim {
                begin: b,
                stride: 1,
                len: 1,
                shrink: true,
            };
        }

        // deal with begin and end masks
        let s = strides[ix];
        let b = if self.ignore_begin(ix) {
            if s.signum() > 0 {
                0
            } else {
                dim - 1
            }
        } else {
            b
        };
        let e = if self.ignore_end(ix) {
            if s.signum() < 0 {
                -1
            } else {
                dim
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
    }

    fn prepare(
        &self,
        input_shape: &[usize],
        begin: &ArrayView1<i32>,
        end: &ArrayView1<i32>,
        strides: &ArrayView1<i32>,
    ) -> (Vec<Dim>, Vec<usize>, Vec<usize>) {
        trace!(
            "StridedSlice {:?} computing shapes: input_shape:{:?} begin:{:?} end:{:?} strides:{:?}",
            self,
            input_shape,
            begin,
            end,
            strides
        );
        let bounds: Vec<Dim> = (0..input_shape.len())
            .map(|ix| self.prepare_one_dim(ix, input_shape[ix], begin, end, strides))
            .collect();
        trace!("StridedSlice bounds {:?}", bounds);
        let mid_shape: Vec<usize> = bounds.iter().map(|d| d.len).collect();
        let end_shape: Vec<usize> = bounds.iter().filter(|d| !d.shrink).map(|d| d.len).collect();
        (bounds, mid_shape, end_shape)
    }
}

impl<T: Datum> Op for StridedSlice<T> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: Vec<Value>) -> Result<Vec<Value>> {
        let (input, begin, end, strides) = args_4!(inputs);
        let input = T::tensor_to_view(&input)?;
        let begin = begin.as_i32s().ok_or("Begin expected as I32")?;
        let end = end.as_i32s().ok_or("End expected as I32")?;
        let strides = strides.as_i32s().ok_or("Strides expected as I32")?;
        let (bounds, mid_shape, end_shape) = self.prepare(
            input.shape(),
            &begin.view().into_dimensionality()?,
            &end.view().into_dimensionality()?,
            &strides.view().into_dimensionality()?,
        );
        let output = Array::from_shape_fn(mid_shape, |coords| {
            let coord: Vec<_> = coords
                .slice()
                .iter()
                .enumerate()
                .map(|(d, i)| (*i as i32 * bounds[d].stride + bounds[d].begin) as usize)
                .collect();
            input[&*coord]
        });
        let output = output.into_shape(end_shape)?;
        trace!("StrideSlice output: {:?}", output);
        Ok(vec![T::array_into_tensor(output.into()).into()])
    }

    /// Returns the attributes of the operation and their values.
    fn get_attributes(&self) -> HashMap<&'static str, Attr> {
        hashmap!{
            "begin_mask"       => Attr::I64(self.begin_mask),
            "end_mask"         => Attr::I64(self.end_mask),
            "shrink_axis_mask" => Attr::I64(self.shrink_axis_mask),
        }
    }

    /// Returns a new streaming buffer for the operation.
    fn new_buffer(&self) -> Box<OpBuffer> {
        Box::new(StrideSliceBuffer { skip: None })
    }

    fn step(
        &self,
        mut inputs: Vec<StepValue>,
        buffer: &mut Box<OpBuffer>,
    ) -> Result<Option<Vec<Value>>> {
        let (input, begin, end, strides) = args_4!(inputs);

        let begin = begin.into_const().ok_or("begin can not be streamed")?;
        let end = end.into_const().ok_or("end can not be streamed")?;
        let strides = strides.into_const().ok_or("strides can not be streamed")?;

        let mut begin = begin.into_tensor().take_i32s().ok_or("Begin expected as I32")?;
        let mut end = end.into_tensor().take_i32s().ok_or("End expected as I32")?;
        let mut strides = strides.into_tensor().take_i32s().ok_or("Strides expected as I32")?;

        let (dim, input) = input.into_stream().ok_or("data must be streamed")?;

        let input = if let Some(input) = input {
            input
        } else {
            return Ok(None)
        };

        if input.shape()[dim] != 1 {
            bail!("StridedSlice assumes streaming chunk of 1")
        }
        let buffer = buffer
            .downcast_mut::<StrideSliceBuffer>()
            .ok_or("The buffer can't be downcasted to Buffer<T>.")?;
        if buffer.skip.is_none() {
            buffer.skip = if self.ignore_begin(dim) || begin[dim] < 0 {
                    Some(0)
                } else {
                    Some(begin[dim] as usize)
                }
        };
        let skip = buffer.skip.as_mut().unwrap();
        if *skip > 0 {
            *skip -= 1;
            Ok(None)
        } else {
            *skip = strides[dim] as usize - 1;
            begin[dim] = 0;
            end[dim] = 1;
            strides[dim] = 1;
            Ok(Some(self.eval(vec!(input, begin.into(), end.into(), strides.into()))?))
        }
    }
}

impl<T: Datum> InferenceRulesOp for StridedSlice<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver
            .equals(&inputs.len, 4)
            .equals(&outputs.len, 1)
            .equals(&inputs[1].datum_type, DatumType::I32)
            .equals(&inputs[2].datum_type, DatumType::I32)
            .equals(&inputs[3].datum_type, DatumType::I32)
            .equals(&inputs[0].datum_type, &outputs[0].datum_type)
            .equals(&inputs[1].rank, 1)
            .equals(&inputs[2].rank, 1)
            .equals(&inputs[3].rank, 1)
            .equals_all(wrap!(
                &inputs[1].shape[0],
                &inputs[2].shape[0],
                &inputs[3].shape[0]
            ))
            .given(&inputs[0].shape, move |solver, input_shape: ShapeFact| {
                if input_shape.open {
                    return;
                }
                solver.given(&inputs[1].value, move |solver, begin: Tensor| {
                    let input_shape = input_shape.clone();
                    solver.given(&inputs[2].value, move |solver, end: Tensor| {
                        let input_shape = input_shape.clone();
                        let begin = begin.clone();
                        solver.given(&inputs[3].value, move |solver, stride: Tensor| {
                            let begin = begin
                                .as_i32s()
                                .unwrap()
                                .view()
                                .into_dimensionality()
                                .unwrap();
                            let end = end.as_i32s().unwrap().view().into_dimensionality().unwrap();
                            let stride = stride
                                .as_i32s()
                                .unwrap()
                                .view()
                                .into_dimensionality()
                                .unwrap();
                            let dims: Vec<IntFact> = input_shape
                                .dims
                                .iter()
                                .enumerate()
                                .filter_map(|(ix, d)| {
                                    if self.must_shrink(ix) {
                                        None
                                    } else {
                                        match d {
                                            DimFact::Only(d) => Some(IntFact::Only(
                                                self.prepare_one_dim(ix, *d, &begin, &end, &stride)
                                                    .len
                                                    as _,
                                            )),
                                            DimFact::Streamed => {
                                                Some(IntFact::Special(SpecialKind::Streamed))
                                            }
                                            DimFact::Any => Some(IntFact::Any),
                                        }
                                    }
                                })
                                .collect();
                            solver.equals(&outputs[0].rank, dims.len() as isize);
                            for (ix, d) in dims.iter().enumerate() {
                                solver.equals(&outputs[0].shape[ix], *d);
                            }
                        });
                    });
                });
            });
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use ndarray::*;
    use Tensor;

    fn eval<I, B, E, S>(op: StridedSlice<i32>, input: I, begin: B, end: E, strides: S) -> Tensor
    where
        I: Into<Tensor>,
        B: Into<Tensor>,
        E: Into<Tensor>,
        S: Into<Tensor>,
    {
        op.eval(vec![
            input.into().into(),
            begin.into().into(),
            end.into().into(),
            strides.into().into(),
        ]).unwrap()
            .pop()
            .unwrap()
            .into_tensor()
    }

    // https://www.tensorflow.org/api_docs/python/tf/strided_slice
    #[test]
    fn eval_1() {
        assert_eq!(
            eval(
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
            Tensor::from(arr3(&[[[3, 3, 3]]])),
        );
    }

    #[test]
    fn eval_2() {
        assert_eq!(
            eval(
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
            Tensor::from(arr3(&[[[3, 3, 3], [4, 4, 4]]])),
        );
    }

    #[test]
    fn eval_3() {
        assert_eq!(
            eval(
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
            Tensor::from(arr3(&[[[4, 4, 4], [3, 3, 3]]])),
        );
    }

    #[test]
    fn eval_4() {
        assert_eq!(
            eval(
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
            Tensor::from(arr3(&[[[3, 3], [4, 4]]])),
        );
    }

    #[test]
    fn eval_5() {
        assert_eq!(
            eval(
                StridedSlice::default(),
                arr1(&[0, 0]),
                arr1(&[0]),
                arr1(&[-1]),
                arr1(&[1])
            ),
            Tensor::from(arr1(&[0]))
        )
    }

    #[test]
    fn eval_6() {
        assert_eq!(
            eval(
                StridedSlice::default(),
                arr2(&[[1, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]]),
                arr1(&[-3, -4]),
                arr1(&[-1, -1]),
                arr1(&[1, 2])
            ),
            Tensor::from(arr2(&[[1, 0], [3, 0]]))
        )
    }

    #[test]
    fn eval_7() {
        assert_eq!(
            eval(
                StridedSlice::default(),
                arr2(&[[0, 6], [0, 0]]),
                arr1(&[0]),
                arr1(&[2]),
                arr1(&[1])
            ),
            Tensor::from(arr2(&[[0, 6], [0, 0]]))
        )
    }

    #[test]
    fn eval_begin_mask_1() {
        let mut op = StridedSlice::default();
        op.begin_mask = 1;
        assert_eq!(
            eval(op, arr1(&[0, 1]), arr1(&[1]), arr1(&[1]), arr1(&[1])),
            Tensor::from(arr1(&[0]))
        )
    }

    #[test]
    fn eval_shrink_1() {
        let mut op = StridedSlice::default();
        op.shrink_axis_mask = 1;
        assert_eq!(
            eval(
                op,
                arr2(&[[0]]),
                arr1(&[0, 0]),
                arr1(&[0, 0]),
                arr1(&[1, 1])
            ),
            Tensor::I32(arr1(&[]).into_dyn())
        )
    }

    #[test]
    fn inference_1() {
        use ops::InferenceOp;
        let op = StridedSlice::<f32>::new(5, 7, 0);
        let input = TensorFact::default().with_datum_type(DatumType::F32);
        let begin = TensorFact::from(arr1(&[0i32, 2, 0]));
        let end = TensorFact::from(arr1(&[0i32, 0, 0]));
        let strides = TensorFact::from(arr1(&[1i32, 1, 1]));

        let (input_facts, output_facts) =
            op.infer(
                vec![input, begin.clone(), end.clone(), strides.clone()],
                vec![TensorFact::default()],
            ).unwrap();
        assert_eq!(
            input_facts,
            vec![
                TensorFact::default()
                    .with_datum_type(DatumType::F32)
                    .with_shape(shapefact![..]),
                begin,
                end,
                strides,
            ]
        );
        assert_eq!(
            output_facts,
            vec![
                TensorFact::default()
                    .with_datum_type(DatumType::F32)
                    .with_shape(shapefact![..]),
            ]
        );
    }

    #[test]
    fn inference_2() {
        use ops::InferenceOp;
        let op = StridedSlice::<f32>::new(1, 1, 2);
        let input = TensorFact::default().with_datum_type(DatumType::F32);
        let begin = TensorFact::from(arr1(&[0i32, 0]));
        let end = TensorFact::from(arr1(&[0i32, 1]));
        let strides = TensorFact::from(arr1(&[1i32, 1]));

        let (input_facts, output_facts) =
            op.infer(
                vec![input, begin.clone(), end.clone(), strides.clone()],
                vec![TensorFact::default()],
            ).unwrap();
        assert_eq!(
            input_facts,
            vec![
                TensorFact::default()
                    .with_datum_type(DatumType::F32)
                    .with_shape(shapefact![..]),
                begin,
                end,
                strides,
            ]
        );
        assert_eq!(
            output_facts,
            vec![
                TensorFact::default()
                    .with_datum_type(DatumType::F32)
                    .with_shape(shapefact![..]),
            ]
        );
    }

}
