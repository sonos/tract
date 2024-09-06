use crate::internal::*;
use tract_core::ops::array::StridedSlice;
use tract_itertools::Itertools;

impl InferenceRulesOp for StridedSlice {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(
            inputs,
            3 + self.optional_axes_input.is_some() as usize
                + self.optional_steps_input.is_some() as usize,
        )?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].rank, 1)?;
        s.equals(&inputs[2].rank, 1)?;
        s.equals(&inputs[1].shape[0], &inputs[2].shape[0])?;
        s.equals(
            &outputs[0].rank,
            inputs[0].rank.bex() - self.shrink_axis_mask.count_ones() as i64,
        )?;
        if let Some(axis) = self.optional_axes_input {
            s.equals(&inputs[1].shape, &inputs[axis].shape)?;
        };
        if let Some(step) = self.optional_steps_input {
            s.equals(&inputs[1].shape, &inputs[step].shape)?;
        };
        if let Some(axes_input) = self.optional_axes_input {
            s.given(&inputs[axes_input].value, move |s, axes| {
                let axes = axes.cast_to::<i64>()?.into_owned();
                s.given(&outputs[0].rank, move |s, orank| {
                    let axes = axes
                        .as_slice::<i64>()?
                        .iter()
                        .map(|a| if *a >= 0 { *a } else { *a + orank } as usize)
                        .collect_vec();
                    let mut iaxis = 0;
                    for oaxis in 0..orank as usize {
                        while self.shrink_axis_mask & (1 << iaxis) != 0 {
                            iaxis += 1;
                        }
                        if !axes.contains(&iaxis) {
                            s.equals(&inputs[0].shape[iaxis], &outputs[0].shape[oaxis])?;
                        }
                        iaxis += 1;
                    }
                    Ok(())
                })
            })?;
        }
        s.given(&inputs[0].shape, move |s, input_shape| {
            s.given_all(inputs[1..].iter().map(|i| &i.value), move |s, params| {
                let begin = &params[0];
                let end = &params[1];
                let strides = if let Some(i) = self.optional_steps_input {
                    let t = params[i - 1].cast_to::<i32>()?;
                    t.as_slice::<i32>()?.to_vec()
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

    to_typed!();
    as_op!();
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;
    use tract_core::ops::array::strided_slice::Dim;
    use tract_ndarray::{arr1, arr2, arr3};

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
            tensor1(&[0])
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
        let mut op = strided_slice(5, 7, 0);
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
        let mut op = strided_slice(1, 1, 2);
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
        let table = SymbolScope::default();
        let s = table.new_with_prefix("S").to_dim();
        let mut op = strided_slice(5, 7, 0);
        let input = f32::fact(dims!(1, s.clone() - 2, 16)).into();
        let begin = InferenceFact::from(tensor1(&[0i32, 2, 0]));
        let end = InferenceFact::from(tensor1(&[0i32, 0, 0]));
        let strides = InferenceFact::from(tensor1(&[1i32, 1, 1]));
        let any = InferenceFact::default();

        let (_, output_facts, _) =
            op.infer_facts(tvec![&input, &begin, &end, &strides], tvec![&any], tvec!()).unwrap();

        assert_eq!(output_facts, tvec![f32::fact(dims!(1, s - 4, 16)).into()]);
    }

    #[test]
    fn prep_1() {
        let op = strided_slice(0, 0, 0);
        assert_eq!(
            op.prepare_one_dim(
                0,
                &4.to_dim(),
                &tensor1(&[-1i64]),
                &tensor1(&[i64::MIN]),
                &[-1]
            )
            .unwrap(),
            Dim { begin: 3.to_dim(), end: (-1).to_dim(), stride: -1, shrink: false }
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
                &tensor1(&[i64::MIN + 1]),
                &[-1]
            )
            .unwrap(),
            Dim { begin: 3.to_dim(), end: (-1).to_dim(), stride: -1, shrink: false }
        );
    }
}
