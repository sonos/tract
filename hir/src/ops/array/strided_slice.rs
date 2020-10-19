use crate::internal::*;
use tract_itertools::Itertools;

#[derive(Debug, Clone, Hash)]
pub struct StridedSlice {
    pub optional_axes_input: Option<usize>,
    pub optional_steps_input: Option<usize>,
    pub begin_mask: i64,
    pub end_mask: i64,
    pub shrink_axis_mask: i64,
}

tract_data::impl_dyn_hash!(StridedSlice);

#[derive(Debug, Clone, PartialEq)]
struct Dim {
    // position of the first element to return
    begin: TDim,
    // position of the first element not to return
    end: TDim,
    stride: i32,
    shrink: bool,
}

impl Dim {
    fn soft_len(&self) -> TractResult<TDim> {
        if let Ok(len) = (self.end.clone() - &self.begin).to_isize() {
            Ok((((self.stride.abs() as i32 - 1) + len.abs() as i32) / self.stride.abs()).to_dim())
        } else if self.stride == 1 {
            Ok(self.end.clone() - &self.begin)
        } else {
            bail!("Streaming dimensions with strides are not supported for now")
        }
    }
}

impl StridedSlice {
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
        begin: &Tensor,
        end: &Tensor,
        strides: &[i32],
    ) -> TractResult<Dim> {
        // cast bouds to Option<Dim>, dealing with ignore from mask, and spec shorted than dim
        // also for end, magic values in onnx :/
        let mut begin: Option<TDim> = if ix >= begin.len() {
            None
        } else {
            let begin = begin.cast_to::<TDim>()?;
            begin.as_slice::<TDim>()?.iter().nth(ix).cloned()
        };

        let mut end: Option<TDim> = if self.ignore_end(ix) || ix >= end.len() {
            None
        } else if end.datum_type() == i64::datum_type() {
            let end = *end.as_slice::<i64>()?.iter().nth(ix).unwrap();
            if end == std::i64::MAX || end == std::i64::MIN || end == std::i64::MIN + 1 {
                None
            } else {
                Some(end.to_dim())
            }
        } else {
            let end = end.cast_to::<TDim>()?;
            end.as_slice::<TDim>()?.iter().nth(ix).cloned()
        };

        let stride = strides.get(ix).cloned().unwrap_or(1);

        // deal with negative indexing
        fn fix_negative(bound: &mut TDim, dim: &TDim) {
            let neg = if let Ok(b) = bound.to_isize() {
                b < 0
            } else {
                let symbols = bound.symbols();
                if symbols.len() == 1 {
                    let sym = symbols.into_iter().nth(0).unwrap();
                    let values = SymbolValues::default().with(sym, 100_000_000);
                    bound.eval(&values).to_isize().unwrap() < 0
                } else {
                    false
                }
            };
            if neg {
                *bound = bound.clone() + dim;
            }
        }
        if let Some(begin) = begin.as_mut() {
            fix_negative(begin, dim)
        }
        if let Some(end) = end.as_mut() {
            fix_negative(end, dim)
        }

        if self.must_shrink(ix) {
            return Ok(Dim {
                begin: begin.clone().unwrap_or(0.to_dim()),
                end: begin.unwrap_or(0.to_dim()) + 1,
                stride: 1,
                shrink: true,
            });
        }

        // must happen after dealing with must_shrink :/
        if self.ignore_begin(ix) {
            begin = None;
        }

        let mut begin =
            begin.unwrap_or_else(|| if stride > 0 { 0.to_dim() } else { dim.clone() - 1 });
        if begin.to_isize().map(|b| b < 0).unwrap_or(false) {
            if stride < 0 {
                return Ok(Dim { begin: 0.to_dim(), end: 0.to_dim(), stride, shrink: false });
            } else {
                begin = 0.to_dim();
            }
        }
        if let (Ok(b), Ok(d)) = (begin.to_isize(), dim.to_isize()) {
            if b > d - 1 {
                if stride > 0 {
                    return Ok(Dim { begin: 0.to_dim(), end: 0.to_dim(), stride, shrink: false });
                } else {
                    begin = (d - 1).to_dim()
                }
            }
        }

        let mut end = end.unwrap_or_else(|| if stride > 0 { dim.clone() } else { (-1).to_dim() });
        if end.to_isize().map(|e| e < 0).unwrap_or(false) {
            if stride > 0 {
                return Ok(Dim { begin: 0.to_dim(), end: 0.to_dim(), stride, shrink: false });
            } else {
                end = -1.to_dim();
            }
        }
        if let (Ok(e), Ok(d)) = (end.to_isize(), dim.to_isize()) {
            if e > d - 1 {
                if stride > 0 {
                    end = d.to_dim()
                } else {
                    return Ok(Dim { begin: 0.to_dim(), end: 0.to_dim(), stride, shrink: false });
                }
            }
        }
        Ok(Dim { begin, end, stride, shrink: false })
    }
}

impl Expansion for StridedSlice {
    fn name(&self) -> Cow<str> {
        "StridedSlice".into()
    }

    op_hir!();

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
                let begin = &params[0];
                let end = &params[1];
                let strides = if let Some(i) = self.optional_steps_input {
                    let t = params[i - 1].cast_to::<i32>()?;
                    t.as_slice::<i32>()?.iter().cloned().collect()
                } else {
                    vec![1; input_shape.len()]
                };
                let axes: TVec<usize> = if let Some(i) = self.optional_axes_input {
                    let axes = params[i - 1].cast_to::<i32>()?;
                    axes.as_slice::<i32>()?
                        .iter()
                        .map(|&i| if i < 0 { input_shape.len() as i32 + i } else { i } as usize)
                        .collect()
                } else {
                    (0..input_shape.len()).collect()
                };
                let mut output_shape = input_shape.clone();
                let mut shrink = vec![];
                for (ix, axis) in axes.into_iter().enumerate() {
                    let preped =
                        self.prepare_one_dim(ix, &input_shape[axis], begin, end, &strides)?;
                    output_shape[axis] = preped.soft_len()?;
                    if preped.shrink {
                        shrink.push(axis);
                    }
                }
                for shrink in shrink.iter().sorted().rev() {
                    output_shape.remove(*shrink);
                }
                s.equals(&outputs[0].shape, output_shape)
            })
        })
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let params: TVec<Option<Arc<Tensor>>> = inputs[1..]
            .iter()
            .map(|i| Ok(target.outlet_fact(*i)?.konst.clone()))
            .collect::<TractResult<_>>()?;
        if params.iter().all(|p| p.is_some()) {
            let params: TVec<&Tensor> = params.iter().map(|o| &**o.as_ref().unwrap()).collect();
            let input_shape = target.outlet_fact(inputs[0])?.shape.clone();
            let strides: TVec<i32> = if let Some(i) = self.optional_steps_input {
                let strides = params[i - 1].cast_to::<i32>()?;
                strides.as_slice::<i32>()?.into()
            } else {
                tvec![1; input_shape.rank()]
            };
            let axes: TVec<usize> = if let Some(i) = self.optional_axes_input {
                let axes = params[i - 1].cast_to::<i32>()?;
                axes.as_slice::<i32>()?
                    .iter()
                    .map(|&i| if i < 0 { input_shape.rank() as i32 + i } else { i } as usize)
                    .collect()
            } else {
                (0..input_shape.rank()).collect()
            };
            let mut wire = inputs[0];
            let input = target.outlet_fact(wire)?.clone();
            for (ix, &axis) in axes.iter().enumerate() {
                let d = &input_shape[axis];
                let preped = self.prepare_one_dim(ix, &d, &params[0], &params[1], &strides)?;
                if preped.stride > 0 {
                    if preped.begin != 0.to_dim() || preped.end != input.shape[ix] {
                        wire = target.wire_node(
                            format!("{}.slice-axis-{}", prefix, axis),
                            crate::ops::array::Slice::new(axis, preped.begin, preped.end),
                            [wire].as_ref(),
                        )?[0];
                    }
                } else {
                    if preped.end != 0.to_dim() || preped.begin != input.shape[ix] {
                        wire = target.wire_node(
                            format!("{}.slice-axis-{}", prefix, axis),
                            crate::ops::array::Slice::new(axis, preped.end + 1, preped.begin + 1),
                            [wire].as_ref(),
                        )?[0];
                    }
                }
                if preped.stride != 1 {
                    wire = target.wire_node(
                        format!("{}.stride-axis-{}", prefix, axis),
                        crate::ops::downsample::Downsample::new(ix, preped.stride as isize, 0),
                        [wire].as_ref(),
                    )?[0];
                }
            }
            let mut shrink = input
                .shape
                .iter()
                .enumerate()
                .filter(|(ix, _d)| self.must_shrink(*ix))
                .map(|pair| pair.0)
                .collect::<Vec<_>>();
            shrink.sort();
            for axis in shrink.iter().rev() {
                wire = target.wire_node(
                    format!("{}.RmDim-{}", prefix, axis),
                    AxisOp::Rm(*axis),
                    [wire].as_ref(),
                )?[0];
            }
            target.rename_node(wire.node, prefix)?;
            Ok(tvec!(wire))
        } else {
            bail!("StridedSlice in not typable when params are dynamic: got:{:?}", params);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use tract_ndarray::{arr1, arr2, arr3};

    fn s() -> TDim {
        tract_pulse::internal::stream_dim()
    }

    pub fn strided_slice(begin_mask: i64, end_mask: i64, shrink_axis_mask: i64) -> StridedSlice {
        StridedSlice {
            begin_mask,
            end_mask,
            shrink_axis_mask,
            optional_axes_input: None,
            optional_steps_input: Some(3),
        }
    }

    fn eval<I, B, E, S>(op: StridedSlice, input: I, begin: B, end: E, strides: S) -> Tensor
    where
        I: Into<Tensor>,
        B: Into<Tensor>,
        E: Into<Tensor>,
        S: Into<Tensor>,
    {
        expand(op)
            .eval(tvec![
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
                strided_slice(0, 0, 0),
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
                strided_slice(0, 0, 0),
                arr3(&[[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]],]),
                tensor1(&[1, 0, 0]),
                tensor1(&[2, 2, 3]),
                tensor1(&[1, 1, 1])
            ),
            Tensor::from(arr3(&[[[3, 3, 3], [4, 4, 4]]])),
        );
    }

    #[test]
    fn eval_3_negative_stride() {
        assert_eq!(
            eval(
                strided_slice(0, 0, 0),
                arr3(&[[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]],]),
                tensor1(&[1, -1, 0]),
                tensor1(&[2, -3, 3]),
                tensor1(&[1, -1, 1])
            ),
            Tensor::from(arr3(&[[[4, 4, 4], [3, 3, 3]]])),
        );
    }

    #[test]
    fn eval_3_bis() {
        assert_eq!(
            eval(
                strided_slice(0, 0, 0),
                arr1(&[0, 1]),
                tensor1(&[-1]),
                tensor1(&[-3]),
                tensor1(&[-1])
            ),
            Tensor::from(arr1(&[1, 0]))
        );
    }

    #[test]
    fn eval_4() {
        assert_eq!(
            eval(
                strided_slice(0, 0, 0),
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
                strided_slice(0, 0, 0),
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
                strided_slice(0, 0, 0),
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
                strided_slice(0, 0, 0),
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
        let mut op = strided_slice(0, 0, 0);
        op.begin_mask = 1;
        assert_eq!(
            eval(op, tensor1(&[0, 1]), tensor1(&[1]), tensor1(&[1]), tensor1(&[1])),
            Tensor::from(tensor1(&[0]))
        )
    }

    #[test]
    fn eval_shrink_1() {
        let mut op = strided_slice(0, 0, 0);
        op.shrink_axis_mask = 1;
        assert_eq!(
            eval(op, arr2(&[[0]]), tensor1(&[0, 0]), tensor1(&[0, 0]), tensor1(&[1, 1])),
            tensor1::<i32>(&[])
        )
    }

    #[test]
    fn eval_shrink_to_scalar() {
        let mut op = strided_slice(0, 0, 0);
        op.shrink_axis_mask = 1;
        assert_eq!(
            eval(op, tensor1(&[0]), tensor1(&[0]), tensor1(&[0]), tensor1(&[1])),
            tensor0::<i32>(0)
        )
    }

    #[test]
    fn inference_1() {
        let op = strided_slice(5, 7, 0);
        let input = InferenceFact::default().with_datum_type(DatumType::F32);
        let begin = InferenceFact::from(tensor1(&[0i32, 2, 0]));
        let end = InferenceFact::from(tensor1(&[0i32, 0, 0]));
        let strides = InferenceFact::from(tensor1(&[1i32, 1, 1]));
        let any = InferenceFact::default();

        let (input_facts, output_facts, _) = expand(op)
            .infer_facts(tvec![&input, &begin, &end, &strides], tvec![&any], tvec!())
            .unwrap();
        assert_eq!(
            input_facts,
            tvec![
                InferenceFact::default()
                    .with_datum_type(DatumType::F32)
                    .with_shape(shapefactoid![..]),
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
        let op = strided_slice(1, 1, 2);
        let input = InferenceFact::default().with_datum_type(DatumType::F32);
        let begin = InferenceFact::from(tensor1(&[0i32, 0]));
        let end = InferenceFact::from(tensor1(&[0i32, 1]));
        let strides = InferenceFact::from(tensor1(&[1i32, 1]));
        let any = InferenceFact::default();

        let (input_facts, output_facts, _) = expand(op)
            .infer_facts(tvec![&input, &begin, &end, &strides], tvec![&any], tvec!())
            .unwrap();
        assert_eq!(
            input_facts,
            tvec![
                InferenceFact::default()
                    .with_datum_type(DatumType::F32)
                    .with_shape(shapefactoid![..]),
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
        let op = strided_slice(5, 7, 0);
        let input = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, (s() - 2), 16));
        let begin = InferenceFact::from(tensor1(&[0i32, 2, 0]));
        let end = InferenceFact::from(tensor1(&[0i32, 0, 0]));
        let strides = InferenceFact::from(tensor1(&[1i32, 1, 1]));
        let any = InferenceFact::default();

        let (_, output_facts, _) = expand(op)
            .infer_facts(tvec![&input, &begin, &end, &strides], tvec![&any], tvec!())
            .unwrap();

        assert_eq!(
            output_facts,
            tvec![InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, (s() - 4), 16))]
        );
    }

    #[test]
    fn inference_4() {
        let op = strided_slice(5, 7, 0);
        let input = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, (s() - 2), 16));
        let begin = InferenceFact::from(tensor1(&[0i32, 2, 0]));
        let end = InferenceFact::from(tensor1(&[0i32, 0, 0]));
        let strides = InferenceFact::from(tensor1(&[1i32, 1, 1]));
        let any = InferenceFact::default();

        let (_, output_facts, _) = expand(op)
            .infer_facts(tvec![&input, &begin, &end, &strides], tvec![&any], tvec!())
            .unwrap();

        assert_eq!(
            output_facts,
            tvec![InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, (s() - 4), 16))]
        );
    }

    #[test]
    fn prep_1() {
        let op = strided_slice(0, 0, 0);
        assert_eq!(
            op.prepare_one_dim(
                0,
                &4.to_dim(),
                &tensor1(&[-1i64]),
                &tensor1(&[std::i64::MIN]),
                &[-1]
            )
            .unwrap(),
            Dim { begin: 3.to_dim(), end: -1.to_dim(), stride: -1, shrink: false }
        );
    }

    #[test]
    fn prep_pytorch_onnx_bug_workadound() {
        let op = strided_slice(0, 0, 0);
        assert_eq!(
            op.prepare_one_dim(
                0,
                &4.to_dim(),
                &tensor1(&[-1i64]),
                &tensor1(&[std::i64::MIN + 1]),
                &[-1]
            )
            .unwrap(),
            Dim { begin: 3.to_dim(), end: -1.to_dim(), stride: -1, shrink: false }
        );
    }
}
