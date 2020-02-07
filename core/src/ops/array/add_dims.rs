use crate::internal::*;
use crate::infer::*;

#[derive(Debug, Clone, new)]
pub struct AddDims {
    pub axes: Vec<usize>,
}

impl AddDims {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        let mut shape: TVec<D> = input.iter().cloned().collect();
        for &axis in &self.axes {
            shape.insert(axis, D::one())
        }
        shape
    }
}

impl Op for AddDims {
    fn name(&self) -> Cow<str> {
        "AddDims".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("Axes: {:?}", self.axes)])
    }

    not_a_typed_op!();
}

impl StatelessOp for AddDims {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let shape = self.compute_shape(input.shape());
        Ok(unsafe { tvec![input.into_tensor().into_shape(&*shape)?.into_arc_tensor()] })
    }
}

impl InferenceRulesOp for AddDims {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, (&inputs[0].rank).bex() + self.axes.len() as i32)?;
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape);
            s.equals(&outputs[0].shape, output_shape)
        })
    }

    #[allow(unused_variables)]
    fn to_typed(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let mut wire = mapping[&node.inputs[0]];
        let mut axes = self.axes.clone();
        axes.sort();
        for axis in axes {
            wire = target.wire_node(
                format!("{}-axis-{}", node.name, axis),
                AxisOp::Add(axis),
                &[wire],
            )?[0];
        }
        Ok(tvec!(wire))
    }

    inference_op_as_op!();
}
