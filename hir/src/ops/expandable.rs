use crate::internal::*;
use tract_core::internal::*;

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

#[derive(Clone, Debug, new, Hash)]
pub struct Expandable {
    pub expansion: Box<dyn Expansion>,
}

impl Hash for Box<dyn Expansion> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        self.dyn_hash(state)
    }
}

tract_linalg::impl_dyn_hash!(Expandable);

impl Op for Expandable {
    fn name(&self) -> Cow<str> {
        self.expansion.name().into()
    }
    fn op_families(&self) -> &'static [&'static str] {
        self.expansion.op_families()
    }
    not_a_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Expandable {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut adhoc = TypedModel::default();
        let wires = inputs
            .iter()
            .enumerate()
            .map(|(ix, i)| adhoc.add_source(format!("adhoc-source-{}", ix), TypedFact::from(&**i)))
            .collect::<TractResult<TVec<OutletId>>>()?;
        let wires = self.expansion.wire("adhoc", &mut adhoc, &*wires)?;
        adhoc.set_output_outlets(&*wires)?;
        SimplePlan::new(adhoc)?.run(inputs.into_iter().map(|t| t.into_tensor()).collect())
    }
}

impl InferenceRulesOp for Expandable {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        self.expansion.rules(s, inputs, outputs)
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<Vec<_>>();
        self.expansion.wire(&node.name, target, &inputs)
    }

    as_op!();
}
