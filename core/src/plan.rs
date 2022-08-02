use std::borrow::Borrow;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::internal::*;
use crate::model::order::eval_order_for_nodes;
use crate::model::{Fact, Graph, OutletId};

#[derive(Default)]
pub struct SessionState {
    pub inputs: HashMap<usize, Arc<Tensor>>,
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

#[derive(Debug, Clone, Educe)]
#[educe(Hash)]
pub struct SimplePlan<F, O, M>
where
    F: Fact + Hash + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
    M: Borrow<Graph<F, O>> + Hash,
{
    pub model: M,
    pub outputs: Vec<OutletId>,
    pub order: Vec<usize>,
    pub flush_lists: Vec<TVec<usize>>,
    pub has_unresolved_symbols: bool,
    _casper: PhantomData<(F, O)>,
}

impl<F, O, M> SimplePlan<F, O, M>
where
    F: Fact + Hash + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
    M: Borrow<Graph<F, O>> + Hash,
{
    /// This contructor returns a plan that will compute all the model default outputs in one pass.
    pub fn new(model: M) -> TractResult<SimplePlan<F, O, M>> {
        let outputs = model.borrow().output_outlets()?.iter().cloned().collect::<Vec<OutletId>>();
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
        let order = eval_order_for_nodes(model.borrow().nodes(), &inputs, &outputs_nodes, deps)?;
        let mut values_needed_until_step = vec![0; model.borrow().nodes().len()];
        for step in 0..order.len() {
            for i in &model.borrow().node(order[step]).inputs {
                values_needed_until_step[i.node] = step;
            }
        }
        for o in outputs.iter() {
            values_needed_until_step[o.node] = order.len();
        }
        let mut flush_lists: Vec<TVec<usize>> = vec![tvec!(); order.len() + 1];
        for (node, &flush_at) in values_needed_until_step.iter().enumerate() {
            if flush_at != 0 {
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

    pub fn run(&self, inputs: TVec<Tensor>) -> TractResult<TVec<Arc<Tensor>>> {
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
    F: Fact + Hash + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
    M: Borrow<Graph<F, O>> + Hash,
    P: Borrow<SimplePlan<F, O, M>>,
{
    plan: P,
    pub states: Vec<Option<Box<dyn OpState>>>,
    pub session_state: SessionState,
    pub values: Vec<Option<TVec<Arc<Tensor>>>>,
    _phantom: PhantomData<(M, F, O)>,
}

impl<F, O, M, P> SimpleState<F, O, M, P>
where
    F: Fact + Hash + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
    M: Borrow<Graph<F, O>> + Hash,
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
        Ok(SimpleState { plan, states, session_state: session, values, _phantom: PhantomData })
    }

    /// Reset wires state.
    pub fn reset_wires(&mut self) -> TractResult<()> {
        self.values.iter_mut().for_each(|s| *s = None);
        Ok(())
    }

    /// Reset wires state.
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

    pub fn run(&mut self, inputs: TVec<Tensor>) -> TractResult<TVec<Arc<Tensor>>> {
        self.run_plan_with_eval(inputs, self::eval)
    }

    pub fn exec(&mut self) -> TractResult<TVec<Arc<Tensor>>> {
        self.exec_plan_with_eval(self::eval)
    }

    pub fn run_plan_with_eval<Eval, E>(
        &mut self,
        inputs: TVec<Tensor>,
        eval: Eval,
    ) -> TractResult<TVec<Arc<Tensor>>>
    where
        Eval: for<'a, 'b, 'c> FnMut(
            &'a mut SessionState,
            Option<&'b mut (dyn OpState + 'static)>,
            &'c Node<F, O>,
            TVec<Arc<Tensor>>,
        ) -> Result<TVec<Arc<Tensor>>, E>,
        E: Into<anyhow::Error> + Send + Sync + 'static,
    {
        self.set_inputs(inputs)?;
        self.exec_plan_with_eval(eval)
    }

    pub fn exec_plan_with_eval<Eval, E>(&mut self, mut eval: Eval) -> TractResult<TVec<Arc<Tensor>>>
    where
        Eval: for<'a, 'b, 'c> FnMut(
            &'a mut SessionState,
            Option<&'b mut (dyn OpState + 'static)>,
            &'c Node<F, O>,
            TVec<Arc<Tensor>>,
        ) -> Result<TVec<Arc<Tensor>>, E>,
        E: Into<anyhow::Error> + Send + Sync + 'static,
    {
        let mut result = tvec!();
        {
            let &mut SimpleState {
                ref plan,
                ref mut session_state,
                ref mut states,
                ref mut values,
                ..
            } = self;
            let plan = plan.borrow();
            let model = plan.model().borrow();
            for (step, n) in plan.order.iter().enumerate() {
                let node = model.node(*n);
                trace!("Running step {}, node {}", step, node);
                let mut inputs: TVec<Arc<Tensor>> = tvec![];
                for i in &node.inputs {
                    trace!("  use input {:?}", i);
                    let prec_node = model.node(i.node);
                    let prec = values[i.node].as_ref().ok_or_else(|| {
                        format_err!("Computing {}, precursor {} not done:", node, prec_node)
                    })?;
                    inputs.push(prec[i.slot].clone().into())
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

                let vs =
                    eval(session_state, states[node.id].as_mut().map(|s| &mut **s), node, inputs)
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
            for output in &plan.outputs {
                trace!("Extracting value {:?} ({})", output, model.node(output.node));
                result.push(values[output.node].as_ref().unwrap()[output.slot].clone())
            }
        }
        self.reset_wires()?;
        Ok(result)
    }

    pub fn set_inputs(&mut self, inputs: TVec<Tensor>) -> TractResult<()> {
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
            TDim::Sym(s) => symbols[*s] = Some(provided),
            TDim::MulInt(x, expr) => Self::resolve(symbols, expr, provided / *x),
            _ => (),
        }
    }

    pub fn set_input(&mut self, input: usize, t: Tensor) -> TractResult<()> {
        let outlet: OutletId = *self
            .model()
            .input_outlets()?
            .get(input)
            .ok_or_else(|| format_err!("Invalid input id for model ({}).", input))?;
        let SimpleState { plan, session_state, .. } = self;
        let plan = (&*plan).borrow();
        let model = plan.model.borrow();
        if let Ok(fact) = model.outlet_fact(outlet)?.to_typed_fact() {
            for (expected, provided) in fact.shape.iter().zip(t.shape()) {
                Self::resolve(&mut session_state.resolved_symbols, &expected, *provided as i64)
            }
        }
        self.plan
            .borrow()
            .model()
            .outlet_fact(outlet)?
            .matches(&t, Some(&self.session_state.resolved_symbols))
            .with_context(|| format!("Setting input {}", input))?;
        self.session_state.inputs.insert(outlet.node, t.into());
        Ok(())
    }

    pub fn take_outputs(&mut self) -> TractResult<Vec<Arc<Tensor>>> {
        let SimpleState { ref plan, ref mut values, .. } = self;
        let mut v = vec![];
        for o in plan.borrow().model().output_outlets()?.iter() {
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

    pub fn set_values(&mut self, id: usize, values: TVec<Tensor>) -> TractResult<()> {
        self.values[id] = Some(values.into_iter().map(|t| t.into()).collect());
        Ok(())
    }

    pub fn set_value(&mut self, id: usize, value: Tensor) -> TractResult<()> {
        self.set_values(id, tvec!(value))
    }

    pub fn prepare_inputs(&self, node: usize) -> TractResult<TVec<Arc<Tensor>>> {
        let SimpleState { ref plan, ref values, .. } = self;
        let plan = plan.borrow();
        let nodes = plan.model().nodes();
        let node = &nodes[node];
        let mut inputs: TVec<Arc<Tensor>> = tvec![];
        for i in &node.inputs {
            let prec_node = &nodes[i.node];
            let prec = values[i.node].as_ref().ok_or_else(|| {
                format_err!("Computing {}, precursor {} not done.", node, prec_node)
            })?;
            inputs.push(prec[i.slot].clone().into_tensor().into_arc_tensor())
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
        inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<()> {
        let SimpleState { ref plan, ref mut session_state, ref mut values, .. } = self;
        let plan = plan.borrow();
        let nodes = plan.model().nodes();
        let node = &nodes[node];
        let vs = match self.states[node.id] {
            Some(ref mut state) => state.eval(session_state, node.op(), inputs),
            None => node.op().eval(inputs),
        }
        .with_context(|| format!("Evaluating {}", node))?;
        values[node.id] = Some(vs);
        Ok(())
    }

    pub fn compute_recursively(&mut self, node: usize) -> TractResult<&[Arc<Tensor>]> {
        let values = {
            let precs: Vec<usize> =
                self.model().nodes()[node].inputs.iter().map(|i| i.node).collect();
            for i in precs.into_iter() {
                if self.values[i].is_none() {
                    let _ = self.compute_recursively(i)?;
                }
            }
            let mut inputs: TVec<Arc<Tensor>> = tvec![];
            {
                let node = &self.model().nodes()[node];
                for i in &node.inputs {
                    inputs.push(self.values[i.node].as_ref().unwrap()[i.slot].clone().into())
                }
            }
            let Self { ref mut states, ref mut session_state, ref plan, .. } = self;
            let plan = plan.borrow();
            match states[node] {
                Some(ref mut state) => {
                    state.eval(session_state, plan.borrow().model().nodes()[node].op(), inputs)
                }
                None => plan.borrow().model().nodes()[node].op().eval(inputs),
            }
            .with_context(|| format!("Evaluating {:?}", node))?
        };
        self.values[node] = Some(values);
        Ok(&*self.values[node].as_ref().unwrap())
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
            .map(|v| Arc::try_unwrap(v).unwrap_or_else(|v| (*v).clone()))
            .collect())
    }

    pub fn plan(&self) -> &SimplePlan<F, O, M> {
        &self.plan.borrow()
    }

    pub fn model(&self) -> &Graph<F, O> {
        self.plan().model()
    }
}

pub fn eval<F, O>(
    session_state: &mut SessionState,
    mut state: Option<&mut (dyn OpState + 'static)>,
    node: &Node<F, O>,
    input: TVec<Arc<Tensor>>,
) -> TractResult<TVec<Arc<Tensor>>>
where
    F: Fact + Hash + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
{
    let r = match state {
        Some(ref mut state) => state.eval(session_state, node.op(), input),
        None => node.op().eval(input),
    }
    .with_context(|| format!("Evaluating {}", node));
    r
}
