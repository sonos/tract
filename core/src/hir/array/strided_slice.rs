use crate::internal::*;
use crate::infer::*;
use ndarray::prelude::*;

#[derive(Debug, Clone)]
pub struct StridedSlice {
    pub optional_axes_input: Option<usize>,
    pub optional_steps_input: Option<usize>,
    pub begin_mask: i64,
    pub end_mask: i64,
    pub shrink_axis_mask: i64,
}

#[derive(Debug, Clone)]
struct Dim {
    begin: TDim,
    end: TDim,
    stride: i32,
    shrink: bool,
}

impl Dim {
    fn len(&self) -> TractResult<usize> {
        Ok((((self.stride.abs() as i32 - 1)
            + (self.end.clone() - &self.begin).to_integer()?.abs() as i32)
            / self.stride.abs()) as usize)
    }

    fn soft_len(&self) -> TractResult<TDim> {
        if let Ok(len) = (self.end.clone() - &self.begin).to_integer() {
            Ok((((self.stride.abs() as i32 - 1) + len.abs() as i32) / self.stride.abs()).to_dim())
        } else if self.stride == 1 {
            Ok(self.end.clone() - &self.begin)
        } else {
            bail!("Streaming dimensions with strides are not supported for now")
        }
    }
}

impl StridedSlice {
    pub fn tensorflow(begin_mask: i64, end_mask: i64, shrink_axis_mask: i64) -> StridedSlice {
        StridedSlice {
            begin_mask,
            end_mask,
            shrink_axis_mask,
            optional_axes_input: None,
            optional_steps_input: Some(3),
        }
    }

    pub fn onnx10(
        optional_axes_input: Option<usize>,
        optional_steps_input: Option<usize>,
    ) -> StridedSlice {
        StridedSlice {
            begin_mask: 0,
            end_mask: 0,
            shrink_axis_mask: 0,
            optional_axes_input,
            optional_steps_input,
        }
    }

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
        dim: &TDim,
        begin: &ArrayView1<TDim>,
        end: &ArrayView1<TDim>,
        strides: &[i32],
    ) -> Dim {
        // deal with too small dim begin/end/stride for input rank
        if ix >= begin.len() {
            return Dim { begin: 0.to_dim(), end: dim.clone(), stride: 1, shrink: false };
        }

        let stride = strides[ix];
        // deal with negative indexing
        fn must_add_to_len(bound: &TDim) -> bool {
            if let Some(b) = bound.as_const() {
                b < 0
            } else {
                bound.eval(100_000_000).unwrap() < 0 // FIXME
            }
        }
        let mut b: TDim =
            if must_add_to_len(&begin[ix]) { dim.clone() + &begin[ix] } else { begin[ix].clone() };
        let mut e: TDim =
            if must_add_to_len(&end[ix]) { dim.clone() + &end[ix] } else { end[ix].clone() };

        // begin and end > dimension -> clip
        let b_overflow = if let (Some(i32beg), Some(i32dim)) = (b.as_const(), dim.as_const()) {
            i32beg >= i32dim
        } else {
            false
        };
        let e_overflow = if let (Some(i32end), Some(i32dim)) = (e.as_const(), dim.as_const()) {
            i32end >= i32dim
        } else {
            false
        };

        // deal with shrinking
        // (weirdly, tf ignores begin_mask when shrink is used)
        if self.must_shrink(ix) {
            return Dim { begin: b.clone(), end: b.clone() + 1, stride: 1, shrink: true };
        }

        if stride.signum() > 0 {
            if self.ignore_begin(ix) {
                b = 0.to_dim();
            } else if b_overflow {
                b = dim.clone();
            }
            if self.ignore_end(ix) || e_overflow {
                e = dim.clone();
            }
        } else {
            if self.ignore_begin(ix) || b_overflow {
                b = dim.clone() - 1;
            }
            if self.ignore_end(ix) {
                e = -1.to_dim();
            } else if e_overflow {
                e = dim.clone() - 1;
            }
        }

        Dim { begin: b, end: e, stride, shrink: false }
    }

    fn slice_t<T: Datum>(
        &self,
        data: &Tensor,
        mid_shape: &[usize],
        bounds: &[Dim],
    ) -> TractResult<Tensor> {
        let input = data.to_array_view::<T>()?;
        Ok(Array::from_shape_fn(mid_shape, |coords| {
            let coord: Vec<_> = coords
                .slice()
                .iter()
                .enumerate()
                .map(|(d, i)| {
                    (*i as i32 * bounds[d].stride + bounds[d].begin.to_integer().unwrap() as i32)
                        as usize
                })
                .collect();
            input[&*coord].clone()
        })
        .into_tensor())
    }
}

impl StatelessOp for StridedSlice {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let casted_begin = inputs[1].cast_to::<TDim>()?;
        let begin = casted_begin.to_array_view::<TDim>()?.into_dimensionality()?;
        let casted_end = inputs[2].cast_to::<TDim>()?;
        let end = casted_end.to_array_view::<TDim>()?.into_dimensionality()?;
        let input_rank = inputs[0].rank();
        let strides: TVec<i32> = if let Some(i) = self.optional_steps_input {
            let strides = inputs[i].cast_to::<i32>()?;
            strides.as_slice::<i32>()?.into()
        } else {
            tvec![1; input_rank]
        };
        let axes: TVec<usize> = if let Some(i) = self.optional_axes_input {
            let axes = inputs[i].cast_to::<i32>()?;
            axes.as_slice::<i32>()?
                .iter()
                .map(|&i| if i < 0 { input_rank as i32 + i } else { i } as usize)
                .collect()
        } else {
            (0..input_rank).collect()
        };
        trace!(
            "StridedSlice {:?} computing shapes: input_shape:{:?} begin:{:?} end:{:?} strides:{:?} axes:{:?}",
            self,
            inputs[0].shape(),
            begin,
            end,
            strides,
            axes,
        );
        let bounds: TVec<Dim> = (0..inputs[0].rank())
            .map(|axis| {
                let dim = inputs[0].shape()[axis].to_dim();
                if let Some(ix) = axes.iter().position(|&x| x == axis) {
                    self.prepare_one_dim(ix, &dim, &begin, &end, &strides)
                } else {
                    Dim { begin: 0.to_dim(), end: dim, stride: 1, shrink: false }
                }
            })
            .collect();
        trace!("StridedSlice bounds {:?}", bounds);
        let mid_shape: Vec<usize> =
            bounds.iter().map(|d| d.len()).collect::<TractResult<Vec<usize>>>()?;
        let end_shape: Vec<usize> = bounds
            .iter()
            .filter(|d| !d.shrink)
            .map(|d| d.len())
            .collect::<TractResult<Vec<usize>>>()?;
        let dt = inputs[0].datum_type();
        let output =
            dispatch_datum!(Self::slice_t(dt)(self, inputs[0].as_ref(), &mid_shape, &bounds))?;
        let output = unsafe { output.into_shape(&end_shape)? };
        Ok(tvec![output.into_arc_tensor()])
    }
}

impl Op for StridedSlice {
    fn name(&self) -> Cow<str> {
        "tf.StridedSliceD".into()
    }

    not_a_typed_op!();
}

impl InferenceRulesOp for StridedSlice {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(
            &inputs,
            3 + self.optional_axes_input.is_some() as usize
                + self.optional_steps_input.is_some() as usize,
        )?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].rank, 1)?;
        s.equals(&inputs[2].rank, 1)?;
        s.equals(&inputs[1].shape[0], &inputs[2].shape[0])?;
        if let Some(axis) = self.optional_axes_input {
            s.equals(&inputs[1].shape, &inputs[axis].shape)?;
        };
        if let Some(step) = self.optional_steps_input {
            s.equals(&inputs[1].shape, &inputs[step].shape)?;
        };
        s.given(&inputs[0].shape, move |s, input_shape| {
            s.given_all(inputs[1..].iter().map(|i| &i.value), move |s, params| {
                let casted_begin = params[0].cast_to::<TDim>()?;
                let begin = casted_begin.to_array_view::<TDim>()?.into_dimensionality()?;
                let casted_end = params[1].cast_to::<TDim>()?;
                let end = casted_end.to_array_view::<TDim>()?.into_dimensionality()?;
                let strides = if let Some(i) = self.optional_steps_input {
                    let t = params[i - 1].cast_to::<i32>()?;
                    t.as_slice::<i32>()?.iter().cloned().collect()
                } else {
                    vec![1; input_shape.len()]
                };
                let mut current_out_dim = 0;
                for (ix, d) in input_shape.iter().enumerate() {
                    if !self.must_shrink(ix) {
                        let preped = self.prepare_one_dim(ix, d, &begin, &end, &strides);
                        s.equals(&outputs[0].shape[current_out_dim], preped.soft_len()?)?;
                        current_out_dim += 1;
                    }
                }
                s.equals(&outputs[0].rank, current_out_dim as i32)
            })
        })
    }

    as_op!();

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let params: TVec<Option<Arc<Tensor>>> = node.inputs[1..]
            .iter()
            .map(|i| Ok(target.outlet_fact(mapping[i])?.konst.clone()))
            .collect::<TractResult<_>>()?;
        if params.iter().all(|p| p.is_some()) {
            let params: TVec<&Tensor> = params.iter().map(|o| &**o.as_ref().unwrap()).collect();
            let casted_begin = params[0].cast_to::<TDim>()?;
            let begin = casted_begin.to_array_view::<TDim>()?.into_dimensionality()?;
            let casted_end = params[1].cast_to::<TDim>()?;
            let end = casted_end.to_array_view::<TDim>()?.into_dimensionality()?;
            let input_shape = target.outlet_fact(mapping[&node.inputs[0]])?.shape.clone();
            let strides: TVec<i32> = if let Some(i) = self.optional_steps_input {
                let strides = params[i - 1].cast_to::<i32>()?;
                strides.as_slice::<i32>()?.into()
            } else {
                tvec![1; input_shape.rank()]
            };
            if strides.iter().any(|&s| s < 0) {
                bail!("FIXME: negative strides are not yet supported by tract-core");
            }
            let axes: TVec<usize> = if let Some(i) = self.optional_axes_input {
                let axes = params[i - 1].cast_to::<i32>()?;
                axes.as_slice::<i32>()?
                    .iter()
                    .map(|&i| if i < 0 { input_shape.rank() as i32 + i } else { i } as usize)
                    .collect()
            } else {
                (0..input_shape.rank()).collect()
            };
            let mut wire = mapping[&node.inputs[0]];
            let input = target.outlet_fact(wire)?.clone();
            for (ix, &axis) in axes.iter().enumerate() {
                let d = input_shape.dim(axis);
                let preped = self.prepare_one_dim(ix, &d, &begin, &end, &strides);
                if preped.begin != 0.to_dim() || preped.end != input.shape.dim(ix) {
                    wire = target.wire_node(
                        format!("{}-Slice", node.name),
                        crate::ops::array::Slice::new(axis, preped.begin, preped.end),
                        [wire].as_ref(),
                    )?[0];
                }
                if preped.stride != 1 {
                    wire = target.wire_node(
                        format!("{}-Stride-{}", node.name, ix),
                        crate::ops::downsample::Downsample::new(ix, preped.stride as usize, 0),
                        [wire].as_ref(),
                    )?[0];
                }
            }
            let mut shrink = input
                .shape
                .iter()
                .enumerate()
                .filter(|(ix, d)| {
                    let preped = self.prepare_one_dim(*ix, &d, &begin, &end, &strides);
                    preped.shrink
                })
                .map(|pair| pair.0)
                .collect::<Vec<_>>();
            shrink.sort();
            for axis in shrink.iter().rev() {
                wire = target.wire_node(
                    format!("{}-RmDim-{}", node.name, axis),
                    AxisOp::Rm(*axis),
                    [wire].as_ref(),
                )?[0];
            }
            target.rename_node(wire.node, &*node.name)?;
            Ok(tvec!(wire))
        } else {
            bail!("StridedSlice in not typable")
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use ndarray::*;

    fn eval<I, B, E, S>(op: StridedSlice, input: I, begin: B, end: E, strides: S) -> Tensor
    where
        I: Into<Tensor>,
        B: Into<Tensor>,
        E: Into<Tensor>,
        S: Into<Tensor>,
    {
        op.eval(tvec![
            input.into().into(),
            begin.into().into(),
            end.into().into(),
            strides.into().into(),
        ])
        .unwrap()
        .pop()
        .unwrap()
        .into_tensor()
    }

    // https://www.tensorflow.org/api_docs/python/tf/strided_slice
    #[test]
    fn eval_1() {
        assert_eq!(
            eval(
                StridedSlice::tensorflow(0, 0, 0),
                arr3(&[[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]],]),
                tensor1(&[1, 0, 0]),
                tensor1(&[2, 1, 3]),
                tensor1(&[1, 1, 1])
            ),
            Tensor::from(arr3(&[[[3, 3, 3]]])),
        );
    }

    #[test]
    fn eval_2() {
        assert_eq!(
            eval(
                StridedSlice::tensorflow(0, 0, 0),
                arr3(&[[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]],]),
                tensor1(&[1, 0, 0]),
                tensor1(&[2, 2, 3]),
                tensor1(&[1, 1, 1])
            ),
            Tensor::from(arr3(&[[[3, 3, 3], [4, 4, 4]]])),
        );
    }

    #[test]
    fn eval_3() {
        assert_eq!(
            eval(
                StridedSlice::tensorflow(0, 0, 0),
                arr3(&[[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]],]),
                tensor1(&[1, -1, 0]),
                tensor1(&[2, -3, 3]),
                tensor1(&[1, -1, 1])
            ),
            Tensor::from(arr3(&[[[4, 4, 4], [3, 3, 3]]])),
        );
    }

    #[test]
    fn eval_4() {
        assert_eq!(
            eval(
                StridedSlice::tensorflow(0, 0, 0),
                tensor3(&[[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]],]),
                tensor1(&[1, 0, 0]),
                tensor1(&[2, 2, 4]),
                tensor1(&[1, 1, 2])
            ),
            tensor3(&[[[3, 3], [4, 4]]]),
        );
    }

    #[test]
    fn eval_5() {
        assert_eq!(
            eval(
                StridedSlice::tensorflow(0, 0, 0),
                tensor1(&[0, 0]),
                tensor1(&[0]),
                tensor1(&[-1]),
                tensor1(&[1])
            ),
            tensor1(&[0])
        )
    }

    #[test]
    fn eval_6() {
        assert_eq!(
            eval(
                StridedSlice::tensorflow(0, 0, 0),
                tensor2(&[[1, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]]),
                tensor1(&[-3, -4]),
                tensor1(&[-1, -1]),
                tensor1(&[1, 2])
            ),
            tensor2(&[[1, 0], [3, 0]])
        )
    }

    #[test]
    fn eval_7() {
        assert_eq!(
            eval(
                StridedSlice::tensorflow(0, 0, 0),
                tensor2(&[[0, 6], [0, 0]]),
                tensor1(&[0]),
                tensor1(&[2]),
                tensor1(&[1])
            ),
            tensor2(&[[0, 6], [0, 0]])
        )
    }

    #[test]
    fn eval_begin_mask_1() {
        let mut op = StridedSlice::tensorflow(0, 0, 0);
        op.begin_mask = 1;
        assert_eq!(
            eval(op, tensor1(&[0, 1]), tensor1(&[1]), tensor1(&[1]), tensor1(&[1])),
            Tensor::from(tensor1(&[0]))
        )
    }

    #[test]
    fn eval_shrink_1() {
        let mut op = StridedSlice::tensorflow(0, 0, 0);
        op.shrink_axis_mask = 1;
        assert_eq!(
            eval(op, arr2(&[[0]]), tensor1(&[0, 0]), tensor1(&[0, 0]), tensor1(&[1, 1])),
            tensor1::<i32>(&[])
        )
    }

    #[test]
    fn eval_shrink_to_scalar() {
        let mut op = StridedSlice::tensorflow(0, 0, 0);
        op.shrink_axis_mask = 1;
        assert_eq!(
            eval(op, tensor1(&[0]), tensor1(&[0]), tensor1(&[0]), tensor1(&[1])),
            tensor0::<i32>(0)
        )
    }

    #[test]
    fn inference_1() {
        let mut op = StridedSlice::tensorflow(5, 7, 0);
        let input = InferenceFact::default().with_datum_type(DatumType::F32);
        let begin = InferenceFact::from(tensor1(&[0i32, 2, 0]));
        let end = InferenceFact::from(tensor1(&[0i32, 0, 0]));
        let strides = InferenceFact::from(tensor1(&[1i32, 1, 1]));
        let any = InferenceFact::default();

        let (input_facts, output_facts, _) =
            op.infer_facts(tvec![&input, &begin, &end, &strides], tvec![&any], tvec!()).unwrap();
        assert_eq!(
            input_facts,
            tvec![
                InferenceFact::default().with_datum_type(DatumType::F32).with_shape(shapefactoid![..]),
                begin,
                end,
                strides,
            ]
        );
        assert_eq!(
            output_facts,
            tvec![InferenceFact::default()
                .with_datum_type(DatumType::F32)
                .with_shape(shapefactoid![..]),]
        );
    }

    #[test]
    fn inference_2() {
        let mut op = StridedSlice::tensorflow(1, 1, 2);
        let input = InferenceFact::default().with_datum_type(DatumType::F32);
        let begin = InferenceFact::from(tensor1(&[0i32, 0]));
        let end = InferenceFact::from(tensor1(&[0i32, 1]));
        let strides = InferenceFact::from(tensor1(&[1i32, 1]));
        let any = InferenceFact::default();

        let (input_facts, output_facts, _) =
            op.infer_facts(tvec![&input, &begin, &end, &strides], tvec![&any], tvec!()).unwrap();
        assert_eq!(
            input_facts,
            tvec![
                InferenceFact::default().with_datum_type(DatumType::F32).with_shape(shapefactoid![..]),
                begin,
                end,
                strides,
            ]
        );
        assert_eq!(
            output_facts,
            tvec![InferenceFact::default()
                .with_datum_type(DatumType::F32)
                .with_shape(shapefactoid![..]),]
        );
    }

    #[test]
    fn inference_3() {
        let mut op = StridedSlice::tensorflow(5, 7, 0);
        let input =
            InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, (TDim::stream() - 2), 16));
        let begin = InferenceFact::from(tensor1(&[0i32, 2, 0]));
        let end = InferenceFact::from(tensor1(&[0i32, 0, 0]));
        let strides = InferenceFact::from(tensor1(&[1i32, 1, 1]));
        let any = InferenceFact::default();

        let (_, output_facts, _) =
            op.infer_facts(tvec![&input, &begin, &end, &strides], tvec![&any], tvec!()).unwrap();

        assert_eq!(
            output_facts,
            tvec![InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, (TDim::stream() - 4), 16))]
        );
    }

    #[test]
    fn inference_4() {
        let mut op = StridedSlice::tensorflow(5, 7, 0);
        let input =
            InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, (TDim::stream() - 2), 16));
        let begin = InferenceFact::from(tensor1(&[0i32, 2, 0]));
        let end = InferenceFact::from(tensor1(&[0i32, 0, 0]));
        let strides = InferenceFact::from(tensor1(&[1i32, 1, 1]));
        let any = InferenceFact::default();

        let (_, output_facts, _) =
            op.infer_facts(tvec![&input, &begin, &end, &strides], tvec![&any], tvec!()).unwrap();

        assert_eq!(
            output_facts,
            tvec![InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, (TDim::stream() - 4), 16))]
        );
    }
}
