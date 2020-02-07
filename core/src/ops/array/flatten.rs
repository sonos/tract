use crate::internal::*;
use crate::infer::*;

// TODO: canonicalize as Reshape

#[derive(Debug, Clone, new, Default)]
pub struct Flatten {
    axis: usize,
}

impl Flatten {
    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(
        &self,
        input: Arc<Tensor>,
        shape: (usize, usize),
    ) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec![input.into_tensor().into_array::<T>()?.into_shape(shape)?.into_arc_tensor()])
    }

    fn compute_shape<D: DimLike>(&self, shape: &[D]) -> [D; 2] {
        let shape_0 = shape[..self.axis].iter().fold(D::one(), |acc, v| acc * v);
        let shape_1 = shape[self.axis..].iter().fold(D::one(), |acc, v| acc * v);
        [shape_0, shape_1]
    }
}

impl Op for Flatten {
    fn name(&self) -> Cow<str> {
        "Flatten".into()
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Flatten {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let [shape_0, shape_1] = self.compute_shape(input.shape());
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input, (shape_0, shape_1)))
    }
}

impl InferenceRulesOp for Flatten {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.given(&inputs[0].shape, move |s, shape| {
            let [shape_0, shape_1] = self.compute_shape(&*shape);
            s.equals(&outputs[0].shape, ShapeFactoid::from(vec![shape_0, shape_1]))
        })
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for Flatten {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(
            inputs[0].datum_type,
            self.compute_shape(&*inputs[0].shape.to_tvec()).as_ref(),
        )?))
    }
}
