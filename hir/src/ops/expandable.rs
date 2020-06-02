use crate::internal::*;
use tract_core::internal::*;

pub fn expand<E: Expansion>(e: E) -> Box<dyn InferenceOp> {
    Box::new(Box::new(e) as Box<dyn Expansion>)
}

pub trait Expansion:
    tract_core::dyn_clone::DynClone
    + std::fmt::Debug
    + Send
    + Sync
    + tract_core::downcast_rs::Downcast
    + tract_core::internal::DynHash
{
    fn name(&self) -> &'static str;
    fn op_families(&self) -> &'static [&'static str];
    fn validation(&self) -> Validation {
        Validation::Accurate
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1)
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>>;

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult;
}

tract_core::dyn_clone::clone_trait_object!(Expansion);

impl Hash for Box<dyn Expansion> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        self.dyn_hash(state)
    }
}

tract_linalg::impl_dyn_hash!(Box<dyn Expansion>);

impl Op for Box<dyn Expansion> {
    fn name(&self) -> Cow<str> {
        self.as_ref().name().into()
    }
    fn op_families(&self) -> &'static [&'static str] {
        self.as_ref().op_families()
    }
    not_a_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Box<dyn Expansion> {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut adhoc = TypedModel::default();
        let wires = inputs
            .iter()
            .enumerate()
            .map(|(ix, i)| adhoc.add_source(format!("adhoc-source-{}", ix), TypedFact::from(&**i)))
            .collect::<TractResult<TVec<OutletId>>>()?;
        let wires = self.wire("adhoc", &mut adhoc, &*wires)?;
        adhoc.set_output_outlets(&*wires)?;
        SimplePlan::new(adhoc)?.run(inputs.into_iter().map(|t| t.into_tensor()).collect())
    }
}

impl InferenceRulesOp for Box<dyn Expansion> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        self.as_ref().rules(s, inputs, outputs)
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<Vec<_>>();
        self.wire(&node.name, target, &inputs)
    }

    fn nboutputs(&self) -> TractResult<usize> {
        self.as_ref().nboutputs()
    }

    as_op!();
}
