use crate::internal::*;
use crate::infer::*;

use crate::ops::array::MultiBroadcastTo as Typed;

#[derive(Debug, Clone, new, Default)]
pub struct MultiBroadcastTo;

impl Op for MultiBroadcastTo {
    fn name(&self) -> Cow<str> {
        "MultiBroadcastTo".into()
    }

    not_a_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for MultiBroadcastTo {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, dims) = args_2!(inputs);
        let dims: Vec<usize> = dims.to_array_view::<i64>()?.iter().map(|i| *i as usize).collect();
        let dims = crate::broadcast::multi_broadcast(&[&*dims, &*input.shape()])
            .ok_or("incompatible shapes")?;
        dispatch_datum!(Typed::eval_t(input.datum_type())(&*input, &*dims))
    }
}

impl InferenceRulesOp for MultiBroadcastTo {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&inputs[1].rank, 1)?;
        s.given(&inputs[0].shape, move |s, shape| {
            s.given(&inputs[1].value, move |s, dims| {
                let dims = dims.cast_to::<TDim>()?;
                let dims =
                    crate::broadcast::multi_broadcast(&[&*dims.as_slice::<TDim>()?, &*shape])
                        .ok_or("incompatible shapes")
                        .unwrap();
                s.equals(&outputs[0].shape, ShapeFactoid::from(dims))
            })
        })
    }

    fn to_typed(
        &self,
        source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let (Some(input_shape), Some(shape)) = (
            source.outlet_fact(node.inputs[0])?.shape.concretize(),
            source.outlet_fact(node.inputs[1])?.value.concretize(),
        ) {
            let shape = shape.cast_to::<TDim>()?;
            let shape = shape.as_slice::<TDim>()?;
            let dims = crate::broadcast::multi_broadcast(&[&*input_shape, shape])
                .ok_or("incompatible shapes")?;
            let op = Typed::new(dims.into());
            return target.wire_node(&*node.name, op, [mapping[&node.inputs[0]]].as_ref());
        }
        bail!("shape input is variable")
    }

    as_op!();
}


