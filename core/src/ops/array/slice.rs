use crate::internal::*;
use ndarray::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct Slice<D: DimLike + ToDim> {
    axis: usize,
    start: D,
    end: D,
}

impl<D: DimLike + ToDim> Slice<D> {
    fn eval_t<T: Datum>(&self, input: Arc<Tensor>) -> TractResult<Arc<Tensor>> {
        let mut input = input.to_array_view::<T>()?;
        input.slice_axis_inplace(
            Axis(self.axis),
            ::ndarray::Slice::from((self.start.to_integer()?)..(self.end.to_integer()?)),
        );
        Ok(Tensor::from(input.to_owned()).into())
    }
}

impl<D: DimLike + ToDim> Op for Slice<D> {
    fn name(&self) -> Cow<str> {
        "Slice".into()
    }

    fn translation_invariants(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Vec<TranslationInvariant>> {
        let fact = model.outlet_fact(node.inputs[0])?;
        let axes = (0..fact.shape.rank())
            .filter(|&ax| self.axis != ax)
            .map(|axis| TranslationInvariant { axis, period: 1 })
            .collect();
        Ok(axes)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

}

impl<D: DimLike + ToDim> StatelessOp for Slice<D> {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_datum!(Self::eval_t(input.datum_type())(self, input))?))
    }
}

impl<D: DimLike + ToDim> InferenceRulesOp for Slice<D> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.given(&inputs[0].rank, move |s, rank| {
            (0..(rank as usize)).try_for_each(move |axis| {
                if axis == self.axis {
                    s.equals(
                        &outputs[0].shape[axis],
                        (self.end.clone() - &self.start).to_dim(),
                    )
                } else {
                    s.equals(&outputs[0].shape[axis], &inputs[0].shape[axis])
                }
            })
        })?;
        Ok(())
    }

    inference_op_as_op!();
}
