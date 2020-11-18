use std::borrow::Borrow;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::internal::*;
use crate::model::order::eval_order_for_nodes;
use crate::model::{Fact, Graph, OutletId};

#[derive(Clone, Debug, Default)]
pub struct SessionState {
    pub inputs: HashMap<usize, Arc<Tensor>>,
    pub resolved_symbols: SymbolValues,
    pub tensors: HashMap<String, Tensor>,
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
        Ok(SimplePlan { model, order, outputs: outputs.to_vec(), _casper: PhantomData })
    }

    pub fn run(&self, inputs: TVec<TensorVar>) -> TractResult<TVec<Tensor>> {
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
    pub values: Vec<Option<TVec<(usize, Option<Tensor>)>>>,
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

    pub fn run(&mut self, inputs: TVec<TensorVar>) -> TractResult<TVec<Tensor>> {
        self.run_plan_with_eval(inputs, self::eval)
    }

    #[inline(never)]
    pub fn run_plan_with_eval<Eval, E>(
        &mut self,
        inputs: TVec<TensorVar>,
        mut eval: Eval,
    ) -> TractResult<TVec<Tensor>>
    where
        Eval: for<'a, 'b, 'c> FnMut(
            &'a mut SessionState,
            Option<&'b mut (dyn OpState + 'static)>,
            &'c Node<F, O>,
            TVec<TensorVar>,
        ) -> Result<TVec<Tensor>, E>,
        E: Into<anyhow::Error> + Send + Sync + 'static,
    {
        let mut result: TVec<Tensor> = tvec!();
        unsafe {
            self.set_inputs(inputs.into_iter().map(|t| t.into_tensor()).collect())?;
            let &mut SimpleState {
                ref plan,
                ref mut session_state,
                ref mut states,
                ref mut values,
                ..
            } = self;
            let plan = plan.borrow();
            let model = plan.model().borrow();
            let mut exclusive: TVec<Option<TensorVar>> = tvec!();
            for (step, n) in plan.order.iter().enumerate() {
                let node = model.node(*n);
                trace!("Running step {}, node {}", step, node);
                for i in node.inputs.iter() {
                    let value = values
                        .get_unchecked_mut(i.node)
                        .as_mut()
                        .unwrap()
                        .get_unchecked_mut(i.slot);
                    value.0 -= 1;
                    if value.0 == 0 {
                        exclusive.push(Some(TensorVar::Exclusive(value.1.take().unwrap())))
                    } else {
                        exclusive.push(None)
                    }
                }
                let inputs: TVec<TensorVar> = exclusive
                    .drain(..)
                    .zip(node.inputs.iter())
                    .map(|(v, i)| {
                        v.unwrap_or_else(|| {
                            TensorVar::Borrow(
                                values[i.node].as_ref().unwrap()[i.slot].1.as_ref().unwrap(),
                            )
                        })
                    })
                    .collect();

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
                        if !f.matches(&*v)? {
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
                        if !f.matches(v)? {
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

                values[node.id] = Some(
                    vs.into_iter()
                        .enumerate()
                        .map(|(ix, t)| {
                            let successors = node.outputs[ix].successors.len();
                            let outputs = model
                                .borrow()
                                .outputs
                                .iter()
                                .filter(|o| **o == (node.id, ix).into())
                                .count();
                            (successors + outputs, Some(t))
                        })
                        .collect(),
                );
            }
            for output in &plan.outputs {
                trace!("Extracting value {:?} ({})", output, model.node(output.node));
                result.push(values[output.node].as_ref().unwrap()[output.slot].1.clone().unwrap())
            }
        }
        self.reset_wires()?;
        Ok(result)
    }

    pub fn set_inputs(&mut self, inputs: TVec<Tensor>) -> TractResult<()> {
        for (ix, t) in inputs.into_iter().enumerate() {
            self.set_input(ix, t)?
        }
        Ok(())
    }

    pub fn set_input(&mut self, input: usize, t: Tensor) -> TractResult<()> {
        let outlet: OutletId = *self
            .model()
            .input_outlets()?
            .get(input)
            .ok_or_else(|| format_err!("Invalid input id for model ({}).", input))?;
        self.plan
            .borrow()
            .model()
            .outlet_fact(outlet)?
            .matches(&t)
            .with_context(|| format!("Setting input {}", input))?;
        self.session_state.inputs.insert(outlet.node, t.into());
        Ok(())
    }

    pub fn take_outputs(&mut self) -> TractResult<TVec<Tensor>> {
        let SimpleState { ref plan, ref mut values, .. } = self;
        let mut v = tvec![];
        for o in plan.borrow().model().output_outlets()?.iter() {
            let vs = values[o.node].as_mut().ok_or_else(|| {
                format_err!(
                    "Outputs of {:?} are not computed",
                    &plan.borrow().model().nodes()[o.node]
                )
            })?;
            v.push(vs[o.slot].clone().1.unwrap())
        }
        Ok(v)
    }

    pub fn set_values(&mut self, id: usize, values: TVec<Tensor>) -> TractResult<()> {
        self.values[id] = Some(
            values
                .into_iter()
                .enumerate()
                .map(|(ix, t)| (self.model().node(id).outputs[ix].successors.len(), Some(t)))
                .collect(),
        );
        Ok(())
    }

    pub fn set_value(&mut self, id: usize, value: Tensor) -> TractResult<()> {
        self.set_values(id, tvec!(value))
    }

    pub fn compute_recursively(&mut self, node: usize) -> TractResult<TVec<&Tensor>> {
        let values = {
            let precs: Vec<usize> =
                self.model().nodes()[node].inputs.iter().map(|i| i.node).collect();
            for i in precs.into_iter() {
                if self.values[i].is_none() {
                    let _ = self.compute_recursively(i)?;
                }
            }
            let mut inputs: TVec<TensorVar> = tvec![];
            {
                let node = &self.model().nodes()[node];
                for i in &node.inputs {
                    inputs.push(TensorVar::Borrow(
                        self.values[i.node].as_ref().unwrap()[i.slot].1.as_ref().unwrap(),
                    ))
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
        self.values[node] = Some(values.into_iter().map(|t| (1, Some(t))).collect());
        Ok(self.values[node].as_ref().unwrap().iter().map(|t| t.1.as_ref().unwrap()).collect())
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
            .map(|t| t.1.unwrap())
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
    input: TVec<TensorVar>,
) -> TractResult<TVec<Tensor>>
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
