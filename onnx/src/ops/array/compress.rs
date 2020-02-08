use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::internal::*;
use tract_core::infer::*;

pub fn compress(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    Ok((Box::new(Compress::new(node.get_attr_opt("axis")?)), vec![]))
}

#[derive(Debug, Clone, new, Default)]
pub struct Compress {
    axis: Option<usize>,
}

impl Compress {
    fn eval_t<T: Datum>(&self, input: Arc<Tensor>, conds: &[bool]) -> TractResult<Arc<Tensor>> {
        use tract_core::ndarray::*;
        let compressed_dim = conds.iter().filter(|c| **c).count();
        if let Some(ax) = self.axis {
            let input = input.to_array_view::<T>()?;
            let mut shape: TVec<usize> = input.shape().into();
            shape[self.axis.unwrap()] = compressed_dim;
            let mut array: ArrayD<T> = unsafe { T::uninitialized_array(&*shape) };
            for (ixo, ixi) in
                conds.iter().enumerate().filter(|(_, c)| **c).map(|(ix, _)| ix).enumerate()
            {
                array.index_axis_mut(Axis(ax), ixo).assign(&input.index_axis(Axis(ax), ixi));
            }
            Ok(array.into_arc_tensor())
        } else {
            let input = input.as_slice::<T>()?;
            let data: Vec<T> = conds
                .iter()
                .enumerate()
                .filter(|(_, c)| **c)
                .map(|(ix, _)| input[ix].clone())
                .collect();
            Ok(Array::from(data).into_arc_tensor())
        }
    }
}

impl Op for Compress {
    fn name(&self) -> Cow<str> {
        "onnx.Compress".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for Compress {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, cond) = args_2!(inputs);
        let output =
            dispatch_datum!(Self::eval_t(input.datum_type())(self, input, cond.as_slice()?))?;
        Ok(tvec!(output))
    }
}

impl InferenceRulesOp for Compress {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].datum_type, bool::datum_type())?;
        s.equals(&inputs[1].rank, 1)?;
        if let Some(op_axis) = self.axis {
            s.equals(&inputs[0].rank, &outputs[0].rank)?;
            s.given(&inputs[0].rank, move |s, rank| {
                let rank = rank as usize;
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
