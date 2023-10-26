use std::borrow::Borrow;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::internal::*;
use crate::model::order::eval_order_for_nodes;
use crate::model::{Fact, Graph, OutletId};
use crate::ops::konst::Const;
use crate::ops::FrozenOpState;

#[derive(Default)]
pub struct SessionState {
    pub inputs: HashMap<usize, TValue>,
    pub resolved_symbols: SymbolValues,
    pub tensors: HashMap<String, Tensor>,
    pub cached_mmm_scratch_space: Option<Box<dyn tract_linalg::mmm::ScratchSpace>>,
}

impl Clone for SessionState {
    fn clone(&self) -> Self {
        SessionState {
            inputs: self.inputs.clone(),
            resolved_symbols: self.resolved_symbols.clone(),
            tensors: self.tensors.clone(),
            cached_mmm_scratch_space: None,
        }
    }
}

impl Debug for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SessionState({:?})", self.resolved_symbols)
    }
}

#[derive(Debug, Clone)]
pub struct SimplePlan<F, O, M>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    M: Borrow<Graph<F, O>>,
{
    model: M,
    outputs: Vec<OutletId>,
    order: Vec<usize>,
    flush_lists: Vec<TVec<usize>>,
    has_unresolved_symbols: bool,
    _casper: PhantomData<(F, O)>,
}

impl<F, O, M> SimplePlan<F, O, M>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    M: Borrow<Graph<F, O>>,
{
    /// This contructor returns a plan that will compute all the model default outputs in one pass.
    pub fn new(model: M) -> TractResult<SimplePlan<F, O, M>> {
        let outputs = model.borrow().output_outlets()?.to_vec();
        Self::new_for_outputs(model, &outputs)
    }

    /// This contructor returns a plan that will compute the specified output.
    pub fn new_for_output(model: M, output: OutletId) -> TractResult<SimplePlan<F, O, M>> {
        Self::new_for_outputs_and_deps(model, &[output], &[])
    }

    /// This contructor returns a plan that will compute all specified outputs in one pass.
    pub fn new_for_outputs(model: M, outputs: &[OutletId]) -> TractResult<SimplePlan<F, O, M>> {
        Self::new_for_outputs_and_deps(model, outputs, &[])
    }

    pub fn new_for_outputs_and_deps(
        model: M,
        outputs: &[OutletId],
        deps: &[(usize, usize)],
    ) -> TractResult<SimplePlan<F, O, M>> {
        let inputs = model.borrow().input_outlets()?.iter().map(|n| n.node).collect::<Vec<usize>>();
        let outputs_nodes = outputs.iter().map(|n| n.node).collect::<Vec<usize>>();
        let mut order =
            eval_order_for_nodes(model.borrow().nodes(), &inputs, &outputs_nodes, deps)?;
        order.retain(|node| !model.borrow().node(*node).op_is::<Const>());
        let mut values_needed_until_step = vec![0; model.borrow().nodes().len()];
        for (step, node) in order.iter().enumerate() {
            for i in &model.borrow().node(*node).inputs {
                values_needed_until_step[i.node] = step;
            }
        }
        for o in outputs.iter() {
            values_needed_until_step[o.node] = order.len();
        }
        let mut flush_lists: Vec<TVec<usize>> = vec![tvec!(); order.len() + 1];
        for (node, &flush_at) in values_needed_until_step.iter().enumerate() {
            if flush_at != 0 && !model.borrow().node(node).op_is::<Const>() {
                flush_lists[flush_at].push(node)
            }
        }
        let mut symbols: std::collections::HashSet<Symbol> = Default::default();
        for node in &model.borrow().nodes {
            for output in &node.outputs {
                if let Ok(fact) = output.fact.to_typed_fact() {
                    symbols.extend(fact.shape.iter().flat_map(|d| d.symbols()))
                }
            }
        }
        Ok(SimplePlan {
            model,
            order,
            flush_lists,
            outputs: outputs.to_vec(),
            has_unresolved_symbols: !symbols.is_empty(),
            _casper: PhantomData,
        })
    }

    pub fn order_without_consts(&self) -> &[usize] {
        &self.order
    }

    pub fn run(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut state = SimpleState::new(self)?;
        state.run(inputs)
    }

    pub fn model(&self) -> &Graph<F, O> {
        self.model.borrow()
    }
}

#[derive(Clone, Debug)]
pub struct SimpleState<F, O, M, P>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    M: Borrow<Graph<F, O>>,
    P: Borrow<SimplePlan<F, O, M>>,
{
    plan: P,
    pub states: Vec<Option<Box<dyn OpState>>>,
    pub session_state: SessionState,
    pub values: Vec<Option<TVec<TValue>>>,
    _phantom: PhantomData<(M, F, O)>,
}

impl<F, O, M, P> SimpleState<F, O, M, P>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    M: Borrow<Graph<F, O>>,
    P: Borrow<SimplePlan<F, O, M>> + Clone,
{
    pub fn new(plan: P) -> TractResult<SimpleState<F, O, M, P>> {
        let values = vec![None; plan.borrow().model.borrow().nodes().len()];
        let mut session = SessionState::default();
        let model = plan.borrow().model();
        let states: Vec<Option<Box<dyn OpState>>> = model
            .nodes()
            .iter()
            .map(|n: &Node<F, O>| n.op().state(&mut session, n.id))
            .collect::<TractResult<_>>()?;
        let mut state =
            SimpleState { plan, states, session_state: session, values, _phantom: PhantomData };
        state.populate_consts();
        Ok(state)
    }

    fn populate_consts(&mut self) {
        for node in &self.plan.borrow().model().nodes {
            if let Some(k) = node.op_as::<Const>() {
                self.values[node.id] = Some(tvec!(k.0.clone().into_tvalue()));
            }
        }
    }

    /// Reset wires state.
    pub fn reset_turn(&mut self) -> TractResult<()> {
        for node in &self.plan.borrow().order {
            self.values[*node] = None;
        }
        Ok(())
    }

    /// Reset op inner state.
    pub fn reset_op_states(&mut self) -> TractResult<()> {
        let &mut SimpleState { ref plan, ref mut session_state, ref mut states, .. } = self;
        *states = plan
            .borrow()
            .model()
            .nodes()
            .iter()
            .map(|n| n.op().state(session_state, n.id))
            .collect::<TractResult<_>>()?;
        Ok(())
    }

    pub fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.run_plan_with_eval(inputs, self::eval)
    }

    pub fn exec(&mut self) -> TractResult<()> {
        self.exec_plan_with_eval(self::eval)
    }

    pub fn run_plan_with_eval<Eval, E>(
        &mut self,
        inputs: TVec<TValue>,
        eval: Eval,
    ) -> TractResult<TVec<TValue>>
    where
        Eval: for<'a, 'b, 'c> FnMut(
            &'a mut SessionState,
            Option<&'b mut (dyn OpState + 'static)>,
            &'c Node<F, O>,
            TVec<TValue>,
        ) -> Result<TVec<TValue>, E>,
        E: Into<anyhow::Error> + Send + Sync + 'static,
    {
        self.set_inputs(inputs)?;
        self.exec_plan_with_eval(eval)?;
        let outputs = self.outputs()?;
        self.reset_turn()?;
        Ok(outputs)
    }

    pub fn exec_plan_with_eval<Eval, E>(&mut self, mut eval: Eval) -> TractResult<()>
    where
        Eval: for<'a, 'b, 'c> FnMut(
            &'a mut SessionState,
            Option<&'b mut (dyn OpState + 'static)>,
            &'c Node<F, O>,
            TVec<TValue>,
        ) -> Result<TVec<TValue>, E>,
        E: Into<anyhow::Error> + Send + Sync + 'static,
    {
        {
            let &mut SimpleState {
                ref plan,
                ref mut session_state,
                ref mut states,
                ref mut values,
                ..
            } = self;
            let plan = plan.borrow();
            let model = plan.model();
            for (step, n) in plan.order.iter().enumerate() {
                let node = model.node(*n);
                trace!("Running step {}, node {}", step, node);
                let mut inputs: TVec<TValue> = tvec![];
                for i in &node.inputs {
                    trace!("  use input {:?}", i);
                    let prec_node = model.node(i.node);
                    let prec = values[i.node].as_ref().ok_or_else(|| {
                        format_err!("Computing {}, precursor {} not done:", node, prec_node)
                    })?;
                    inputs.push(prec[i.slot].clone())
                }

                for flush in &plan.flush_lists[step] {
                    trace!("  Ran {} can now flush {}", node, model.node(*flush));
                    values[*flush] = None;
                }

                if cfg!(debug_assertions) {
                    let facts = model.node_input_facts(node.id)?;
                    if facts.len() != inputs.len() {
                        bail!(
                            "Evaluating {}: expected {} inputs, got {}",
                            node,
                            facts.len(),
                            inputs.len()
                        );
                    }
                    for (ix, (v, f)) in inputs.iter().zip(facts.iter()).enumerate() {
                        if !f.matches(v, Some(&session_state.resolved_symbols))? {
                            bail!(
                                "Evaluating {}: input {:?}, expected {:?}, got {:?}",
                                node,
                                ix,
                                f,
                                v
                            );
                        }
                    }
                }

                let vs = eval(session_state, states[node.id].as_deref_mut(), node, inputs)
                    .map_err(|e| e.into())?;

                if plan.has_unresolved_symbols {
                    for (o, v) in node.outputs.iter().zip(vs.iter()) {
                        if let Ok(f) = o.fact.to_typed_fact() {
                            for (dim_abstract, dim_concrete) in f.shape.iter().zip(v.shape()) {
                                Self::resolve(
                                    &mut session_state.resolved_symbols,
                                    &dim_abstract,
                                    *dim_concrete as i64,
                                );
                            }
                        }
                    }
                }
                if cfg!(debug_assertions) {
                    let facts = model.node_output_facts(node.id)?;
                    if facts.len() != vs.len() {
                        bail!(
                            "Evaluating {}: expected {} outputs, got {}",
                            node,
                            facts.len(),
                            vs.len()
                        );
                    }
                    for (ix, (v, f)) in vs.iter().zip(facts.iter()).enumerate() {
                        if node.outputs[ix].successors.len() == 0 {
                            continue;
                        }
                        if !f.matches(v, Some(&session_state.resolved_symbols))? {
                            bail!(
                                "Evaluating {}: output {:?}, expected {:?}, got {:?}",
                                node,
                                ix,
                                f,
                                v
                            );
                        }
                    }
                }

                values[node.id] = Some(vs);
            }
        }
        Ok(())
    }

    pub fn set_inputs(&mut self, inputs: TVec<TValue>) -> TractResult<()> {
        ensure!(
            inputs.len() == self.model().inputs.len(),
            "Wrong number of inputs for model. Expected {} got {}",
            self.model().inputs.len(),
            inputs.len()
        );
        for (ix, t) in inputs.into_iter().enumerate() {
            self.set_input(ix, t)?
        }
        Ok(())
    }

    fn resolve(symbols: &mut SymbolValues, expected: &TDim, provided: i64) {
        match expected {
            TDim::Sym(s) => symbols[s] = Some(provided),
            TDim::MulInt(x, expr) => Self::resolve(symbols, expr, provided / *x),
            _ => (),
        }
    }

    pub fn set_input(&mut self, input: usize, t: TValue) -> TractResult<()> {
        let outlet: OutletId = *self
            .model()
            .input_outlets()?
            .get(input)
            .ok_or_else(|| format_err!("Invalid input id for model ({}).", input))?;
        let SimpleState { plan, session_state, .. } = self;
        let plan = (*plan).borrow();
        let model = plan.model.borrow();
        if let Ok(fact) = model.outlet_fact(outlet)?.to_typed_fact() {
            for (expected, provided) in fact.shape.iter().zip(t.shape()) {
                Self::resolve(&mut session_state.resolved_symbols, &expected, *provided as i64)
            }
        }
        let fact = self.plan.borrow().model().outlet_fact(outlet)?;
        ensure!(
            fact.matches(&t, Some(&self.session_state.resolved_symbols))
            .with_context(|| format!("Setting input {input}"))?,
            "Input at index {} has incorrect dtype or shape (got shape {:?} and dtype {:?}, expected to match fact {:?})",
            input,
            t.shape(),
            t.datum_type(),
            fact
            );
        self.session_state.inputs.insert(outlet.node, t);
        Ok(())
    }

    pub fn output(&self, id: usize) -> TractResult<&TValue> {
        let outlet = self.model().output_outlets()?.get(id).with_context(|| {
            format!(
                "Required output {}, only have {}",
                id,
                self.model().output_outlets().unwrap().len()
            )
        })?;
        let value: &TValue = self
            .values
            .get(outlet.node)
            .context("node id for output beyond node values array")?
            .as_ref()
            .context("node is not an output")?
            .get(outlet.slot)
            .context("slot id too high")?;
        Ok(value)
    }

    pub fn outputs(&mut self) -> TractResult<TVec<TValue>> {
        let SimpleState { ref plan, ref mut values, .. } = self;
        let mut v = tvec![];
        for o in plan.borrow().outputs.iter() {
            let vs = values[o.node].as_mut().ok_or_else(|| {
                format_err!(
                    "Outputs of {:?} are not computed",
                    &plan.borrow().model().nodes()[o.node]
                )
            })?;
            v.push(vs[o.slot].clone())
        }
        Ok(v)
    }

    pub fn set_values(&mut self, id: usize, values: TVec<TValue>) -> TractResult<()> {
        self.values[id] = Some(values);
        Ok(())
    }

    pub fn set_value(&mut self, id: usize, value: TValue) -> TractResult<()> {
        self.set_values(id, tvec!(value))
    }

    pub fn prepare_inputs(&self, node: usize) -> TractResult<TVec<TValue>> {
        let SimpleState { ref plan, ref values, .. } = self;
        let plan = plan.borrow();
        let nodes = plan.model().nodes();
        let node = &nodes[node];
        let mut inputs: TVec<TValue> = tvec![];
        for i in &node.inputs {
            let prec_node = &nodes[i.node];
            let prec = values[i.node].as_ref().ok_or_else(|| {
                format_err!("Computing {}, precursor {} not done.", node, prec_node)
            })?;
            inputs.push(prec[i.slot].clone())
        }
        Ok(inputs)
    }

    pub fn compute_one(&mut self, node: usize) -> TractResult<()> {
        let inputs = self.prepare_inputs(node)?;
        self.compute_one_with_inputs(node, inputs)
    }

    pub fn compute_one_with_inputs(
        &mut self,
        node: usize,
        inputs: TVec<TValue>,
    ) -> TractResult<()> {
        let SimpleState { ref plan, ref mut session_state, ref mut values, .. } = self;
        let plan = plan.borrow();
        let nodes = plan.model().nodes();
        let node = &nodes[node];
        let vs = match self.states[node.id] {
            Some(ref mut state) => state.eval(session_state, node.op(), inputs),
            None => node.op().eval(inputs),
        }
        .with_context(|| format!("Evaluating {node}"))?;
        values[node.id] = Some(vs);
        Ok(())
    }

    pub fn compute_recursively(&mut self, node: usize) -> TractResult<&[TValue]> {
        let values = {
            #[allow(clippy::needless_collect)] // clippy bug ?
            let precs: Vec<usize> =
                self.model().nodes()[node].inputs.iter().map(|i| i.node).collect();
            for i in precs.into_iter() {
                if self.values[i].is_none() {
                    let _ = self.compute_recursively(i)?;
                }
            }
            let mut inputs: TVec<TValue> = tvec![];
            {
                let node = &self.model().nodes()[node];
                for i in &node.inputs {
                    inputs.push(self.values[i.node].as_ref().unwrap()[i.slot].clone())
                }
            }
            let Self { ref mut states, ref mut session_state, ref plan, .. } = self;
            let plan = plan.borrow();
            match states[node] {
                Some(ref mut state) => {
                    state.eval(session_state, plan.model().nodes()[node].op(), inputs)
                }
                None => plan.model().nodes()[node].op().eval(inputs),
            }
            .with_context(|| format!("Evaluating {node:?}"))?
        };
        self.values[node] = Some(values);
        Ok(self.values[node].as_ref().unwrap())
    }

    pub fn take_by_name(&mut self, name: &str) -> TractResult<TVec<Tensor>> {
        let id = self.model().node_by_name(name)?.id;
        Self::take(self, id)
    }

    pub fn take(&mut self, id: usize) -> TractResult<TVec<Tensor>> {
        Ok(self.values[id]
            .take()
            .ok_or_else(|| format_err!("Node is not computed"))?
            .into_iter()
            .map(|v| v.into_tensor())
            .collect())
    }

    pub fn plan(&self) -> &SimplePlan<F, O, M> {
        self.plan.borrow()
    }

    pub fn model(&self) -> &Graph<F, O> {
        self.plan().model()
    }

    pub fn freeze(&self) -> FrozenSimpleState<F, O, M, P> {
        FrozenSimpleState {
            plan: self.plan.clone(),
            inputs: self
                .session_state
                .inputs
                .iter()
                .map(|(ix, t)| (*ix, t.clone().into_tensor()))
                .collect(),
            resolved_symbols: self.session_state.resolved_symbols.clone(),
            tensors: self.session_state.tensors.clone(),
            states: self.states.iter().map(|s| s.as_ref().map(|s| s.freeze())).collect(),
            values: self
                .values
                .iter()
                .enumerate()
                .map(|(ix, t)| {
                    if self.model().nodes[ix].op_is::<Const>() {
                        t.as_ref().map(|t| t.iter().map(|t| t.clone().into_tensor()).collect())
                    } else {
                        None
                    }
                })
                .collect(),
            _phantom: PhantomData,
        }
    }
}

pub fn eval<F, O>(
    session_state: &mut SessionState,
    mut state: Option<&mut (dyn OpState + 'static)>,
    node: &Node<F, O>,
    input: TVec<TValue>,
) -> TractResult<TVec<TValue>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    // eprint!("{node} {input:?}");
    let r = match state {
        Some(ref mut state) => state.eval(session_state, node.op(), input),
        None => node.op().eval(input),
    }
    .with_context(|| format!("Evaluating {node}"));
    // eprintln!(" ==> {r:?}");
    r
}

#[derive(Clone, Debug)]
pub struct FrozenSimpleState<F, O, M, P>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    M: Borrow<Graph<F, O>>,
    P: Borrow<SimplePlan<F, O, M>> + Clone,
{
    plan: P,
    pub inputs: HashMap<usize, Tensor>,
    pub resolved_symbols: SymbolValues,
    pub tensors: HashMap<String, Tensor>,
    pub states: Vec<Option<Box<dyn FrozenOpState>>>,
    pub values: Vec<Option<TVec<Tensor>>>,
    _phantom: PhantomData<(M, F, O)>,
}

impl<F, O, M, P> FrozenSimpleState<F, O, M, P>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    M: Borrow<Graph<F, O>>,
    P: Borrow<SimplePlan<F, O, M>> + Clone,
{
    pub fn unfreeze(&self) -> SimpleState<F, O, M, P> {
        let mut state = SimpleState {
            plan: self.plan.clone(),
            session_state: SessionState {
                inputs: self.inputs.iter().map(|(ix, t)| (*ix, t.clone().into_tvalue())).collect(),
                resolved_symbols: self.resolved_symbols.clone(),
                tensors: self.tensors.clone(),
                cached_mmm_scratch_space: None,
            },
            states: self.states.iter().map(|s| s.as_ref().map(|s| s.unfreeze())).collect(),
            values: self
                .values
                .iter()
                .map(|t| t.as_ref().map(|t| t.iter().map(|t| t.clone().into_tvalue()).collect()))
                .collect(),
            _phantom: PhantomData,
        };
        state.populate_consts();
        state
    }
}

#[cfg(test)]
mod test {
    use super::*;
    fn is_send<T: Send>() {}
    fn is_sync<T: Sync>() {}

    #[test]
    fn type_model_is_sync() {
        is_sync::<TypedModel>();
    }

    #[test]
    fn type_model_is_send() {
        is_send::<TypedModel>();
    }

    #[test]
    fn type_plan_is_send() {
        is_send::<TypedSimplePlan<TypedModel>>();
    }

    #[test]
    fn type_plan_is_sync() {
        is_sync::<TypedSimplePlan<TypedModel>>();
    }

    #[test]
    fn frozen_type_state_is_send() {
        is_send::<TypedFrozenSimpleState<TypedModel, TypedSimplePlan<TypedModel>>>();
    }
}
