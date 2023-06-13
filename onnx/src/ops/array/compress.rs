use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_hir::internal::*;

pub fn compress(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    Ok((Box::new(Compress::new(node.get_attr_opt("axis")?)), vec![]))
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Compress {
    axis: Option<isize>,
}

impl Compress {
    unsafe fn eval_t<T: Datum>(
        axis: Option<usize>,
        input: &Tensor,
        conds: &[bool],
        output: &mut Tensor,
    ) {
        use tract_ndarray::*;
        let input = input.to_array_view_unchecked::<T>();
        if let Some(ax) = axis {
            for (ixo, ixi) in
                conds.iter().enumerate().filter(|(_, c)| **c).map(|(ix, _)| ix).enumerate()
            {
                output
                    .to_array_view_mut_unchecked::<T>()
                    .index_axis_mut(Axis(ax), ixo)
                    .assign(&input.index_axis(Axis(ax), ixi));
            }
        } else {
            let output = output.as_slice_mut_unchecked::<T>();
            let mut ix = 0;
            for (c, i) in conds.iter().zip(input.iter()) {
                if *c {
                    output[ix] = i.clone();
                    ix += 1;
                }
            }
        }
    }
}

impl Op for Compress {
    fn name(&self) -> Cow<str> {
        "Compress".into()
    }

    not_a_typed_op!();
}

impl EvalOp for Compress {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, conds) = args_2!(inputs);
        let conds = conds.as_slice()?;
        let compressed_dim = conds.iter().filter(|c| **c).count();
        let (shape, axis) = if let Some(axis) = self.axis {
            let axis = if axis < 0 { axis + input.rank() as isize } else { axis } as usize;
            let mut shape: TVec<usize> = input.shape().into();
            shape[axis] = compressed_dim;
            (shape, Some(axis))
        } else {
            (tvec!(compressed_dim), None)
        };
        unsafe {
            let mut output = Tensor::uninitialized_dt(input.datum_type(), &shape)?;
            dispatch_datum_by_size!(Self::eval_t(input.datum_type())(
                axis,
                &input,
                conds,
                &mut output
            ));
            Ok(tvec!(output.into_tvalue()))
        }
    }
}

impl InferenceRulesOp for Compress {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].datum_type, bool::datum_type())?;
        s.equals(&inputs[1].rank, 1)?;
        if let Some(op_axis) = self.axis {
            s.equals(&inputs[0].rank, &outputs[0].rank)?;
            s.given(&inputs[0].rank, move |s, rank| {
                let rank = rank as usize;
                let op_axis = if op_axis < 0 { op_axis + rank as isize } else { op_axis } as usize;
                for axis in 0..rank {
                    if axis != op_axis {
                        s.equals(&inputs[0].shape[axis], &outputs[0].shape[axis])?;
                    }
                }
                Ok(())
            })?;
        } else {
            s.equals(&outputs[0].rank, 1)?;
        }
        Ok(())
    }

    as_op!();
}
