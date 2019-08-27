use crate::internal::*;
use ndarray::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct Slice<D: DimLike + ToDim> {
    pub axis: usize,
    pub start: D,
    pub end: D,
}

impl<D: DimLike + ToDim> Slice<D> {
    fn eval_t<T: Datum>(&self, input: Arc<Tensor>) -> TractResult<Arc<Tensor>> {
        let mut input = input.to_array_view::<T>()?;
        input.slice_axis_inplace(
            Axis(self.axis),
            ::ndarray::Slice::from((self.start.to_integer()?)..(self.end.to_integer()?)),
        );
        if self.start == self.end {
            // dodge a bug in ndarray :/
            unsafe { return Ok(Tensor::from_raw::<T>(input.shape(), &[])?.into()) }
        }
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
                if self.axis == axis {
                    s.equals(&outputs[0].shape[axis], &(self.end.clone() - &self.start).to_dim())
                } else {
                    s.equals(&outputs[0].shape[axis], &inputs[0].shape[axis])
                }
            })
        })?;
        Ok(())
    }

    inference_op_as_op!();
    to_typed!();
}

impl<D: DimLike + ToDim> TypedOp for Slice<D> {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        let mut fact = inputs[0].clone();
        fact.shape.set_dim(self.axis, (self.end.clone() - &self.start).to_dim())?;
        Ok(tvec!(fact))
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let mut fact = target.outlet_fact(input)?.clone();
        let id = if self.axis == fact.axis {
            fact.delay += self.start.to_integer()? as usize;
            fact.dim = (self.end.clone() - &self.start).to_dim();
            target.chain_after(
                input,
                &*node.name,
                crate::ops::identity::Identity::default(),
                tvec!(fact),
            )?
        } else {
            target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?
        };
        Ok(tvec!(OutletId::new(id, 0)))
    }
}
