use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct PermuteAxes {
    pub axes: Option<Vec<usize>>,
}

tract_linalg::impl_dyn_hash!(PermuteAxes);

impl PermuteAxes {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TractResult<TVec<D>> {
        if let Some(ref axes) = self.axes {
            if input.len() != axes.len() {
                bail!(
                    "Op expects tensor of rank {}, input is actually of rank {}.",
                    axes.len(),
                    input.len()
                );
            }
            let mut new_shape = tvec![D::zero(); input.len()];
            for (ix, &d) in axes.iter().enumerate() {
                new_shape[ix] = input[d].clone();
            }
            Ok(new_shape)
        } else {
            let mut new_shape: TVec<D> = input.iter().cloned().collect();
            new_shape.reverse();
            Ok(new_shape)
        }
    }

    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(&self, input: Arc<Tensor>) -> TractResult<TVec<Arc<Tensor>>> {
        if let Some(ref axes) = self.axes {
            Ok(tvec![input
                .into_tensor()
                .into_array::<T>()?
                .permuted_axes(&**axes)
                .into_arc_tensor()])
        } else {
            Ok(tvec![input.into_tensor().into_array::<T>()?.reversed_axes().into_arc_tensor()])
        }
    }
}

impl Op for PermuteAxes {
    fn name(&self) -> Cow<str> {
        "PermuteAxes".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.axes)])
    }

    op_hir!();
    not_a_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for PermuteAxes {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for PermuteAxes {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape)?;
            s.equals(&outputs[0].shape, output_shape)
        })?;
        if let Some(axes) = &self.axes {
            s.equals(&outputs[0].rank, axes.len() as i32)?;
        }
        Ok(())
    }

    #[allow(unused_variables)]
    fn to_typed(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let fact = target.outlet_fact(mapping[&node.inputs[0]])?;
        if let Some(axes) = &self.axes {
            if fact.rank() != axes.len() {
                bail!(
                    "Op expects tensor of rank {}, input is actually of rank {}.",
                    axes.len(),
                    fact.rank()
                );
            }
            let op = AxisOp::Permute(axes.iter().cloned().collect());
            target.wire_node(&*node.name, op, &[mapping[&node.inputs[0]]])
        } else {
            let axes = (0..fact.rank()).rev().collect();
            let op = AxisOp::Permute(axes);
            target.wire_node(&*node.name, op, &[mapping[&node.inputs[0]]])
        }
    }

    as_op!();
}
