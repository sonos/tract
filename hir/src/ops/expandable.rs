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
    fn name(&self) -> Cow<str>;
    fn op_families(&self) -> &'static [&'static str];
    fn validation(&self) -> Validation {
        Validation::Accurate
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![])
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

tract_data::impl_dyn_hash!(Box<dyn Expansion>);

impl Op for Box<dyn Expansion> {
    fn name(&self) -> Cow<str> {
        self.as_ref().name().into()
    }
    fn op_families(&self) -> &'static [&'static str] {
        self.as_ref().op_families()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        self.as_ref().info()
    }
    not_a_typed_op!();
}

impl EvalOp for Box<dyn Expansion> {
    fn is_stateless(&self) -> bool {
        true
    }

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
        let outputs = self.wire(&node.name, target, &inputs)?;
        for (ix, o) in outputs.iter().enumerate() {
            let expected = &node.outputs[ix].fact;
            let got = target.outlet_fact(*o)?;
            if expected.clone().unify_with(&InferenceFact::from(got)).is_err() {
                bail!("Output mismatch after rewiring expansion for output #{}: expected {:?} got {:?}", ix, expected, got);
            }
        }
        Ok(outputs)
    }

    fn nboutputs(&self) -> TractResult<usize> {
        self.as_ref().nboutputs()
    }

    as_op!();
}

pub fn inference_wrap<O, R>(op: O, rules: R) -> Box<dyn InferenceOp>
where
    O: TypedOp,
    R: for<'r, 'p, 's> Fn(
            &'s dyn Op,
            &mut Solver<'r>,
            &'p [TensorProxy],
            &'p [TensorProxy],
        ) -> InferenceResult
        + Send
        + Sync
        + 'static,
{
    expand(InferenceWrapper { typed_op: Box::new(op), rules: Arc::new(rules) })
}

#[derive(Clone, new, Educe)]
#[educe(Hash)]
pub struct InferenceWrapper {
    typed_op: Box<dyn TypedOp>,
    #[educe(Hash(ignore))]
    rules: Arc<
        dyn for<'r, 'p, 's> Fn(
                &'s dyn Op,
                &mut Solver<'r>,
                &'p [TensorProxy],
                &'p [TensorProxy],
            ) -> InferenceResult
            + Send
            + Sync
            + 'static,
    >,
}

impl std::fmt::Debug for InferenceWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.typed_op, f)
    }
}

impl Expansion for InferenceWrapper {
    fn name(&self) -> Cow<str> {
        self.typed_op.name()
    }

    fn op_families(&self) -> &'static [&'static str] {
        self.typed_op.op_families()
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        model.wire_node(prefix, &self.typed_op, inputs)
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        (self.rules)(self.typed_op.as_op(), s, inputs, outputs)
    }
}

tract_data::impl_dyn_hash!(InferenceWrapper);
