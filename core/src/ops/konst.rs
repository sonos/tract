use std::fmt::Debug;

use crate::internal::*;

/// A constant's value and, when its storage is not plain, the fact describing it.
pub type MaterializedConst = (Arc<Tensor>, Option<Box<dyn ExoticFact>>);

/// Supplies a constant's value on demand, so a model can be built and pruned before its
/// weights are read.
///
/// The provider is asked for its fact at wiring time and for its tensor only when
/// [`materialize_lazy_consts`] runs. A provider must return the same fact both times.
pub trait LazyConstProvider: Debug + Send + Sync + 'static {
    /// The fact of the value this will produce, without reading it.
    fn output_fact(&self) -> TractResult<TypedFact>;
    /// Read the value, plus the exotic fact it needs if its storage is not plain.
    fn materialize(&self) -> TractResult<MaterializedConst>;
}

/// A constant whose value has not been read yet.
///
/// It exists only between loading and [`materialize_lazy_consts`], so that a pass which
/// discards part of the graph — a shard extraction, say — runs before the discarded
/// weights are ever read. It is stateless so that it cannot block const-folding of its
/// consumers, and deliberately has no value: its output fact carries no `konst`, so rules
/// needing the bytes decline to fire rather than see a wrong one. Evaluating it is an
/// error; materialize first.
#[derive(Debug, Clone)]
pub struct LazyConst(pub Arc<dyn LazyConstProvider>);

/// Two lazy constants are the same when they draw on the same provider; a provider has no
/// value to compare until it is materialized.
impl PartialEq for LazyConst {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(Arc::as_ptr(&self.0) as *const (), Arc::as_ptr(&other.0) as *const ())
    }
}

impl Eq for LazyConst {}

impl std::hash::Hash for LazyConst {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.0) as *const (), state)
    }
}

impl Op for LazyConst {
    fn name(&self) -> StaticName {
        "LazyConst".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.0)])
    }

    op_as_typed_op!();
}

impl EvalOp for LazyConst {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, _inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        bail!("LazyConst {:?} was not materialized before evaluation", self.0)
    }
}

impl TypedOp for LazyConst {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.0.output_fact()?))
    }
}

/// Read every lazy constant still in the model and replace it with a [`Const`]. Returns
/// how many were materialized.
///
/// Run this once the graph has been pruned to what will actually be executed: only the
/// constants that survive are read.
pub fn materialize_lazy_consts(model: &mut TypedModel) -> TractResult<usize> {
    let lazy: Vec<usize> =
        model.nodes().iter().filter(|n| n.op_is::<LazyConst>()).map(|n| n.id).collect();
    let count = lazy.len();
    for id in lazy {
        let op = model.node(id).op_as::<LazyConst>().context("not a LazyConst")?.clone();
        let (tensor, exotic) =
            op.0.materialize()
                .with_context(|| format!("materializing {} ({:?})", model.node(id).name, op.0))?;
        let konst = Const::new_with_opt_exotic_fact(tensor, exotic)?;
        let mut patch = TypedModelPatch::default();
        let wire = patch.wire_node(&model.node(id).name, konst, &[])?[0];
        patch.shunt_outside(model, OutletId::new(id, 0), wire)?;
        patch.apply(model)?;
    }
    if count > 0 {
        model.compact()?;
    }
    Ok(count)
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Const(Arc<Tensor>, Option<Box<dyn ExoticFact>>);

impl Const {
    pub fn new(tensor: Arc<Tensor>) -> TractResult<Const> {
        Self::new_with_opt_exotic_fact(tensor, None)
    }

    pub fn new_with_exotic_fact(
        tensor: Arc<Tensor>,
        fact: Box<dyn ExoticFact>,
    ) -> TractResult<Const> {
        Self::new_with_opt_exotic_fact(tensor, Some(fact))
    }

    pub fn new_with_opt_exotic_fact(
        tensor: Arc<Tensor>,
        fact: Option<Box<dyn ExoticFact>>,
    ) -> TractResult<Const> {
        ensure!(fact.is_some() || tensor.is_plain(), "Exotic tensor requires an exotic_fact");
        Ok(Const(tensor, fact))
    }

    pub fn val(&self) -> &Arc<Tensor> {
        &self.0
    }

    pub fn exotic_fact(&self) -> Option<&dyn ExoticFact> {
        self.1.as_deref()
    }
}

impl Op for Const {
    fn name(&self) -> StaticName {
        "Const".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Const {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, _inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(tvec![Arc::clone(&self.0).into_tvalue()])
    }
}

impl TypedOp for Const {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let fact = if self.1.is_some() {
            // Exotic const tensors (e.g. device-backed) may have storage that
            // cannot produce an ExoticFact (like DeviceTensor). Build the fact
            // from dt/shape and attach the explicit exotic_fact from self.1.
            let mut f = TypedFact::dt_shape(
                self.0.datum_type(),
                ShapeFact::from_dims(self.0.shape().iter().map(TDim::from)),
            );
            f.konst = Some(Arc::clone(&self.0));
            f.exotic_fact.clone_from(&self.1);
            f
        } else {
            // Plain tensor: TryFrom sets uniform, uniform_tdim, exotic_fact from storage.
            TypedFact::try_from(&self.0)?
        };
        Ok(tvec!(fact))
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        Ok(tvec!((Cost::Params(self.0.datum_type().unquantized()), self.0.len().into())))
    }

    fn set_symbols(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        _mapping: &HashMap<OutletId, OutletId>,
        subs: &HashMap<Symbol, TDim>,
    ) -> TractResult<TVec<OutletId>> {
        let op = if self.0.datum_type() == TDim::datum_type() {
            let mut tensor = self.0.clone().into_tensor();
            for d in tensor.try_as_plain_mut()?.as_slice_mut::<TDim>()? {
                *d = d.substitute_all(subs)?;
            }
            Const(tensor.into_arc_tensor(), self.1.clone())
        } else {
            self.clone()
        };
        target.wire_node(&node.name, op, &[])
    }

    fn change_axes(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        anyhow::ensure!(io == InOut::Out(0));
        let mut new_tensor = self.0.clone().into_tensor();
        if change.change_tensor(&mut new_tensor, false).is_ok() {
            let mut sub = Const(new_tensor.into_arc_tensor(), None);
            if self.1.is_some() {
                let my_fact = self.output_facts(&[])?;
                let changed_fact = change.output_facts(&[&my_fact[0]])?;
                sub.1 = changed_fact[0].exotic_fact.clone();
            }
            Ok(Some(AxisChangeConsequence {
                substitute_op: Some(Box::new(sub)),
                wire_changes: tvec!((io, change.clone())),
            }))
        } else {
            Ok(None)
        }
    }
}
