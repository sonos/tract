use crate::internal::*;
use crate::ops::identity::Identity;
use ndarray::*;

#[derive(Debug, Clone, new, Default)]
pub struct Slice {
    prune: Vec<(usize, usize)>,
}

impl Slice {
    fn eval_t<T: Datum>(&self, input: Arc<Tensor>) -> TractResult<Arc<Tensor>> {
        let input = input.to_array_view::<T>()?;
        let slice_spec: Vec<SliceOrIndex> = self
            .prune
            .iter()
            .map(|&(a, b)| SliceOrIndex::Slice {
                start: a as isize,
                end: if b != 0 { Some(-(b as isize)) } else { None },
                step: 1,
            })
            .collect();
        let slice_info = SliceInfo::<_, IxDyn>::new(slice_spec).unwrap();
        let slice = input.slice(&slice_info.as_ref());
        Ok(slice.to_owned().into_arc_tensor())
    }
}

impl Op for Slice {
    fn name(&self) -> Cow<str> {
        "Slice".into()
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let fact = target.outlet_fact(input)?;
        if self.prune.iter().enumerate().all(|(ax, &(a, b))| ax == fact.axis || (a == 0 && b == 0))
        {
            let (before, after) = self.prune[fact.axis];
            let mut fact = fact.clone();
            fact.delay += before;
            fact.dim -= before.to_dim() + after.to_dim();
            let id = target.chain_after(input, &*node.name, Identity::default(), tvec!(fact))?;
            return Ok(tvec!(OutletId::new(id, 0)));
        }
        bail!("Slice only support pulsify on streaming axis")
    }
}

impl StatelessOp for Slice {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_datum!(Self::eval_t(input.datum_type())(self, input))?))
    }
}

impl InferenceRulesOp for Slice {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        for (ix, &(a, b)) in self.prune.iter().enumerate() {
            s.equals(&inputs[0].shape[ix], outputs[0].shape[ix].bex() + a.to_dim() + b.to_dim())?;
        }
        Ok(())
    }
}
