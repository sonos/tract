use crate::internal::*;

use super::RmDims;

#[derive(Debug, Clone, new, Default)]
pub struct Squeeze {
    axes: Option<Vec<usize>>,
}

impl Squeeze {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TractResult<TVec<D>> {
        if let Some(ref axes) = self.axes {
            let mut shape: TVec<D> = input.iter().cloned().collect();
            for &axis in axes.iter().rev() {
                if shape.remove(axis) != D::one() {
                    bail!("Attempt to squeeze an axis which dimension in not one");
                }
            }
            Ok(shape)
        } else {
            Ok(input.into_iter().filter(|&d| d != &D::one()).cloned().collect())
        }
    }

    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(&self, input: Arc<Tensor>) -> TractResult<TVec<Arc<Tensor>>> {
        let shape = self.compute_shape(input.shape())?;
        Ok(tvec![input.into_tensor().into_array::<T>()?.into_shape(&*shape)?.into_arc_tensor()])
    }
}

impl Op for Squeeze {
    fn name(&self) -> Cow<str> {
        "Squeeze".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for Squeeze {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for Squeeze {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        if let Some(ref axes) = self.axes {
            s.equals(&outputs[0].rank, (&inputs[0].rank).bex() - axes.len() as i32)?;
        }
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape)?;
            s.equals(&outputs[0].shape, output_shape)
        })
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let axes = if let Some(axes) = &self.axes {
            axes.clone()
        } else {
            let input_fact = target.outlet_fact(input)?;
            input_fact
                .shape
                .iter()
                .enumerate()
                .filter(|(_ix, d)| d == &1.to_dim())
                .map(|(ix, _d)| ix)
                .collect()
        };
        target.wire_node(&*node.name, RmDims::new(axes), [input].as_ref())
    }

    inference_op_as_op!();
}
