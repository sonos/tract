use std::borrow::Borrow;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use multithread::Executor;
use tract_data::itertools::Itertools;

use crate::internal::*;
use crate::model::{Fact, Graph, OutletId};
use crate::ops::FrozenOpState;
use crate::ops::konst::Const;

use self::order::{build_flush_list, eval_order_for_nodes, eval_order_opt_ram_for_nodes};

#[derive(Clone, Debug, Default)]
pub struct PlanOptions {
    /// Use the simple ordering instead of the newer memory friendly one
    pub skip_order_opt_ram: bool,

    /// Override default global executor
    pub executor: Option<Executor>,
}

pub struct SessionState {
    pub inputs: HashMap<usize, TValue>,
    pub resolved_symbols: SymbolValues,
    pub scenario: Option<usize>,
    pub tensors: HashMap<String, Tensor>,
    pub cached_mmm_scratch_space: RefCell<Option<Box<dyn tract_linalg::mmm::ScratchSpace>>>,
    pub scratch_extensions: anymap3::Map,
}

impl Default for SessionState {
    fn default() -> Self {
        SessionState {
            inputs: HashMap::default(),
            resolved_symbols: SymbolValues::default(),
            tensors: HashMap::default(),
            scenario: None,
            cached_mmm_scratch_space: None.into(),
            scratch_extensions: anymap3::Map::new(),
        }
    }
}

impl Clone for SessionState {
    fn clone(&self) -> Self {
        SessionState {
            inputs: self.inputs.clone(),
            resolved_symbols: self.resolved_symbols.clone(),
            tensors: self.tensors.clone(),
            scenario: self.scenario,
            cached_mmm_scratch_space: None.into(),
            scratch_extensions: anymap3::Map::new(),
        }
    }
}

pub trait SessionStateHandler: Send + Sync + Debug {
    fn before_plan_eval(&self, session_state: &mut SessionState) -> TractResult<()>;
    fn after_plan_eval(&self, session_state: &mut SessionState) -> TractResult<()>;
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
    inputs: Vec<OutletId>,
    input_facts: Vec<Option<TypedFact>>,
    outputs: Vec<OutletId>,
    output_facts: Vec<Vec<Option<TypedFact>>>,
    order: Vec<usize>,
    flush_lists: Vec<TVec<usize>>,
    has_unresolved_symbols: bool,
    node_has_symbolic_output: Vec<bool>,
    single_symbolic_output_syms: Vec<Vec<Vec<Option<Symbol>>>>,
    symbols: Vec<Symbol>,
    symbolic_output_dims: Vec<Vec<Vec<usize>>>,
    executor: Option<Executor>,
    session_handler: Option<Arc<dyn SessionStateHandler + 'static>>,
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
        Self::build(model, &outputs, &[], &PlanOptions::default())
    }

    /// This contructor returns a plan that will compute all the model default outputs in one pass.
    pub fn new_with_options(model: M, options: &PlanOptions) -> TractResult<SimplePlan<F, O, M>> {
        let outputs = model.borrow().output_outlets()?.to_vec();
        Self::build(model, &outputs, &[], options)
    }

    /// This contructor returns a plan that will compute the specified output.
    #[deprecated]
    pub fn new_for_output(model: M, output: OutletId) -> TractResult<SimplePlan<F, O, M>> {
        Self::build(model, &[output], &[], &PlanOptions::default())
    }

    /// This contructor returns a plan that will compute all specified outputs in one pass.
    #[deprecated]
    pub fn new_for_outputs(model: M, outputs: &[OutletId]) -> TractResult<SimplePlan<F, O, M>> {
        Self::build(model, outputs, &[], &PlanOptions::default())
    }

    pub fn with_session_handler<H: SessionStateHandler + 'static>(
        mut self,
        session_handler: H,
    ) -> Self {
        self.session_handler = Some(Arc::new(session_handler));
        self
    }

    #[deprecated]
    pub fn new_for_outputs_and_deps(
        model: M,
        outputs: &[OutletId],
        deps: &[(usize, usize)],
    ) -> TractResult<SimplePlan<F, O, M>> {
        Self::build(model, outputs, deps, &PlanOptions::default())
    }

    pub fn build(
        model: M,
        outputs: &[OutletId],
        deps: &[(usize, usize)],
        options: &PlanOptions,
    ) -> TractResult<SimplePlan<F, O, M>> {
        let inputs = model.borrow().input_outlets()?.iter().map(|n| n.node).collect::<Vec<usize>>();
        let outputs_nodes = outputs.iter().map(|n| n.node).collect::<Vec<usize>>();
        let mut order = if options.skip_order_opt_ram {
            eval_order_for_nodes(model.borrow().nodes(), &inputs, &outputs_nodes, deps)?
        } else {
            eval_order_opt_ram_for_nodes(model.borrow().nodes(), &inputs, &outputs_nodes, deps)?
        };
        order.retain(|node| !model.borrow().node(*node).op_is::<Const>());
        let flush_lists =
            build_flush_list(model.borrow(), &order, outputs, |n| !n.op_is::<Const>());

        #[allow(clippy::mutable_key_type)]
        let mut symbols: std::collections::HashSet<Symbol> = Default::default();
        let node_count = model.borrow().nodes.len();
        let mut symbolic_output_dims: Vec<Vec<Vec<usize>>> = vec![vec![]; node_count];
        let mut node_has_symbolic_output: Vec<bool> = vec![false; node_count];
        let mut output_facts: Vec<Vec<Option<TypedFact>>> =
            vec![vec![]; model.borrow().nodes.len()];
        let mut single_symbolic_output_syms: Vec<Vec<Vec<Option<Symbol>>>> =
            vec![vec![]; model.borrow().nodes.len()];

        for node in &model.borrow().nodes {
            let mut node_outputs: Vec<Vec<usize>> = Vec::with_capacity(node.outputs.len());
            let mut node_typed_facts: Vec<Option<TypedFact>> =
                Vec::with_capacity(node.outputs.len());
            let mut node_single_syms: Vec<Vec<Option<Symbol>>> =
                Vec::with_capacity(node.outputs.len());
            for output in &node.outputs {
                if let Ok(fact) = output.fact.to_typed_fact() {
                    symbols.extend(fact.shape.iter().flat_map(|d| d.symbols()));

                    let dims_with_symbols = fact
                        .shape
                        .iter()
                        .enumerate()
                        .filter_map(|(ix, d)| if d.symbols().len() > 0 { Some(ix) } else { None })
                        .collect::<Vec<_>>();
                    node_outputs.push(dims_with_symbols);

                    let mut per_dim_syms: Vec<Option<Symbol>> =
                        Vec::with_capacity(fact.shape.iter().count());
                    for d in fact.shape.iter() {
                        let syms = d.symbols();
                        if syms.len() == 1 {
                            let s = syms.iter().next().unwrap().clone();
                            per_dim_syms.push(Some(s));
                        } else {
                            per_dim_syms.push(None);
                        }
                    }
                    node_single_syms.push(per_dim_syms);
                    node_typed_facts.push(Some(fact.into_owned()));
                } else {
                    node_outputs.push(Vec::new());
                    node_single_syms.push(Vec::new());
                    node_typed_facts.push(None);
                }
            }
            let any_symbolic = node_outputs.iter().any(|v| !v.is_empty());
            output_facts[node.id] = node_typed_facts;
            node_has_symbolic_output[node.id] = any_symbolic;
            symbolic_output_dims[node.id] = node_outputs;
            single_symbolic_output_syms[node.id] = node_single_syms;
        }

        let inputs = model.borrow().input_outlets()?.to_vec();
        let input_facts: Vec<Option<TypedFact>> = inputs
            .iter()
            .map(|outlet| {
                model
                    .borrow()
                    .outlet_fact(*outlet)
                    .ok()
                    .and_then(|f| f.to_typed_fact().ok().map(|f| f.into_owned()))
            })
            .collect();

        Ok(SimplePlan {
            model,
            order,
            flush_lists,
            inputs,
            input_facts,
            outputs: outputs.to_vec(),
            output_facts,
            has_unresolved_symbols: !symbols.is_empty(),
            node_has_symbolic_output,
            single_symbolic_output_syms,
            symbols: symbols.into_iter().collect(),
            symbolic_output_dims,
            _casper: PhantomData,
            executor: options.executor.clone(),
            session_handler: None,
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
        let session = SessionState::default();
        let model = plan.borrow().model();
        let states: Vec<Option<Box<dyn OpState>>> = vec![None; model.nodes.len()];
        let mut state =
            SimpleState { plan, states, session_state: session, values, _phantom: PhantomData };
        state.populate_consts();
        state.reset_op_states()?;
        Ok(state)
    }

    pub fn new_from_inputs(plan: P, inputs: TVec<TValue>) -> TractResult<SimpleState<F, O, M, P>> {
        let mut state = SimpleState::new(plan)?;
        state.set_inputs(inputs)?;
        state.resolve_symbols_with_states()?;

        Ok(state)
    }

    fn populate_consts(&mut self) {
        for node in &self.plan.borrow().model().nodes {
            if let Some(k) = node.op_as::<Const>() {
                self.values[node.id] = Some(tvec!(k.val().clone().into_tvalue()));
            }
        }
    }

    /// Reset wires state.
    pub fn reset_turn(&mut self) -> TractResult<()> {
        for node in &self.plan.borrow().order {
            self.values[*node] = None;
        }
        self.session_state.resolved_symbols = SymbolValues::default();
        Ok(())
    }

    /// Reset op inner state.
    pub fn reset_op_states(&mut self) -> TractResult<()> {
        let &mut SimpleState { ref plan, ref mut session_state, ref mut states, .. } = self;
        for (ix, n) in plan.borrow().model().nodes().iter().enumerate() {
            states[ix] =
                if n.op().is_stateless() { None } else { n.op().state(session_state, ix)? };
        }
        Ok(())
    }

    pub fn init_states(&mut self, state_init_tensors: &mut Vec<TValue>) -> TractResult<()> {
        let states_to_init = self
            .states
            .iter_mut()
            .filter_map(Option::as_mut)
            .filter(|s| s.init_tensor_fact().is_some())
            .collect_vec();
        ensure!(
            states_to_init.len() == state_init_tensors.len(),
            "There are {} op to init but got {} tensors",
            states_to_init.len(),
            state_init_tensors.len()
        );
        for state in states_to_init {
            state.load_from(&mut self.session_state, state_init_tensors)?;
        }
        Ok(())
    }

    fn resolve_symbols_with_states(&mut self) -> TractResult<()> {
        for state in self
            .states
            .iter_mut()
            .filter_map(Option::as_mut)
            .filter(|s| s.init_tensor_fact().is_some())
        {
            state.resolve_symbols(&mut self.session_state)?;
        }
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
        self.resolve_symbols_with_states()?;
        self.exec_plan_with_eval(eval)?;
        let outputs = self.outputs()?;
        self.reset_turn()?;
        Ok(outputs)
    }

    pub fn exec_plan_with_eval<Eval, E>(&mut self, eval: Eval) -> TractResult<()>
    where
        Eval: for<'a, 'b, 'c> FnMut(
            &'a mut SessionState,
            Option<&'b mut (dyn OpState + 'static)>,
            &'c Node<F, O>,
            TVec<TValue>,
        ) -> Result<TVec<TValue>, E>,
        E: Into<anyhow::Error> + Send + Sync + 'static,
    {
        if let Some(executor) = self.plan().executor.as_ref() {
            tract_linalg::multithread::multithread_tract_scope(executor.clone(), || {
                self.do_exec_plan_with_eval(eval)
            })
        } else {
            self.do_exec_plan_with_eval(eval)
        }
    }

    fn do_exec_plan_with_eval<Eval, E>(&mut self, mut eval: Eval) -> TractResult<()>
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
            let plan = self.plan.borrow();
            let model = plan.model.borrow();
            plan.session_handler
                .as_ref()
                .map(|it| it.before_plan_eval(&mut self.session_state))
                .transpose()?;

            let mut syms_done = !plan.has_unresolved_symbols
                || plan
                    .symbols
                    .iter()
                    .all(|s| self.session_state.resolved_symbols.get(s).is_some());
            for (step, &n) in plan.order.iter().enumerate() {
                let node = model.node(n);
                trace!("Running step {step}, node {node}");
                let mut inputs: TVec<TValue> = TVec::with_capacity(node.inputs.len());
                for i in &node.inputs {
                    trace!("  use input {i:?}");
                    let prec = self.values[i.node].as_ref().ok_or_else(|| {
                        let prec_node = model.node(i.node);
                        format_err!("Computing {}, precursor {} not done:", node, prec_node)
                    })?;
                    inputs.push(prec[i.slot].clone())
                }

                for flush in &plan.flush_lists[step] {
                    trace!("  Ran {} can now flush {}", node, model.node(*flush));
                    self.values[*flush] = None;
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
                    let syms = &self.session_state.resolved_symbols;
                    for (ix, (v, f)) in inputs.iter().zip(facts.iter()).enumerate() {
                        if !f.matches(v, Some(syms))? {
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

                let vs = eval(
                    &mut self.session_state,
                    self.states[node.id].as_deref_mut(),
                    node,
                    inputs,
                )
                .map_err(|e| e.into())?;

                if !syms_done
                    && plan.has_unresolved_symbols
                    && unsafe { *plan.node_has_symbolic_output.get_unchecked(node.id) }
                {
                    let symbolic = unsafe { plan.symbolic_output_dims.get_unchecked(node.id) };
                    let single_syms =
                        unsafe { plan.single_symbolic_output_syms.get_unchecked(node.id) };
                    let typed_facts = unsafe { plan.output_facts.get_unchecked(node.id) };
                    for out_ix in 0..node.outputs.len() {
                        let dims = unsafe { symbolic.get_unchecked(out_ix) };
                        if dims.is_empty() {
                            continue;
                        }
                        if let Some(f) = unsafe { typed_facts.get_unchecked(out_ix) }.as_ref() {
                            let vshape = unsafe { vs.get_unchecked(out_ix) }.shape();
                            let syms_per_dim = unsafe { single_syms.get_unchecked(out_ix) };
                            for &ix in dims {
                                let dim_abstract = unsafe { f.shape.get_unchecked(ix) };
                                let dim_concrete = unsafe { *vshape.get_unchecked(ix) } as i64;

                                if let Some(sym) =
                                    unsafe { syms_per_dim.get_unchecked(ix) }.as_ref()
                                {
                                    if self.session_state.resolved_symbols.get(sym).is_none() {
                                        let expected =
                                            dim_abstract.eval(&self.session_state.resolved_symbols);
                                        if let Some(v) =
                                            solve_for(sym, &expected, &dim_concrete.to_dim())
                                        {
                                            let val = v.to_i64()?;
                                            self.session_state.resolved_symbols.set(sym, val);
                                            if self.session_state.scenario.is_none() {
                                                let scope = sym.scope().with_context(|| format!(
                                                    "Symbol {sym:?} points to an invalid (dead ?) SymbolScope. Make sure to create symbols using the model-managed SymbolScope."
                                                ))?;
                                                self.session_state.scenario = scope
                                                    .guess_scenario(
                                                        &self.session_state.resolved_symbols,
                                                    )?;
                                            }
                                        } else {
                                            Self::resolve(
                                                &mut self.session_state,
                                                dim_abstract,
                                                dim_concrete,
                                            )?;
                                        }
                                    }
                                } else {
                                    Self::resolve(
                                        &mut self.session_state,
                                        dim_abstract,
                                        dim_concrete,
                                    )?;
                                }
                            }
                        }
                    }
                    if plan
                        .symbols
                        .iter()
                        .all(|s| self.session_state.resolved_symbols.get(s).is_some())
                    {
                        syms_done = true;
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
                    let syms = &self.session_state.resolved_symbols;
                    for (ix, (v, f)) in vs.iter().zip(facts.iter()).enumerate() {
                        if node.outputs[ix].successors.is_empty() {
                            continue;
                        }
                        if !f.matches(v, Some(syms))? {
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

                self.values[node.id] = Some(vs);
            }
            plan.session_handler
                .as_ref()
                .map(|it| it.after_plan_eval(&mut self.session_state))
                .transpose()?;
        }
        Ok(())
    }

    pub fn set_inputs(&mut self, inputs: TVec<TValue>) -> TractResult<()> {
        let expected_count = self.plan.borrow().inputs.len();
        ensure!(
            inputs.len() == expected_count,
            "Wrong number of inputs for model. Expected {} got {}",
            expected_count,
            inputs.len()
        );

        for (ix, t) in inputs.into_iter().enumerate() {
            self.set_input(ix, t)?
        }
        Ok(())
    }

    fn resolve(state: &mut SessionState, expression: &TDim, provided: i64) -> TractResult<()> {
        {
            use crate::internal::TDim::*;
            if let Sym(sym) = expression {
                if state.resolved_symbols.get(sym).is_none() {
                    state.resolved_symbols.set(sym, provided);
                    if state.scenario.is_none() {
                        let scope = sym.scope().with_context(|| format!(
                            "Symbol {sym:?} points to an invalid (dead ?) SymbolScope. Make sure to create symbols using the model-managed SymbolScope."
                        ))?;
                        state.scenario = scope.guess_scenario(&state.resolved_symbols)?;
                    }
                    return Ok(());
                }
            }
        }

        if let Ok(x) = expression.eval_to_i64(&state.resolved_symbols) {
            ensure!(
                x == provided,
                "Clashing resolution for expression. {expression}={x} != {provided}. ({state:?})"
            );
            return Ok(());
        }

        fn single_unresolved_symbol(expr: &TDim, values: &SymbolValues) -> Option<Symbol> {
            use crate::internal::TDim::*;
            fn walk(e: &TDim, values: &SymbolValues, acc: &mut Option<Symbol>) -> bool {
                match e {
                    Val(_) => true,
                    Sym(s) => {
                        if values.get(s).is_some() {
                            true
                        } else {
                            match acc {
                                None => {
                                    *acc = Some(s.clone());
                                    true
                                }
                                Some(seen) if seen == s => true,
                                Some(_) => false,
                            }
                        }
                    }
                    Add(ts) | Mul(ts) | Broadcast(ts) | Min(ts) | Max(ts) => {
                        for t in ts {
                            if !walk(t, values, acc) {
                                return false;
                            }
                        }
                        true
                    }
                    MulInt(_, a) => walk(a, values, acc),
                    Div(a, _) => walk(a, values, acc),
                }
            }
            let mut only: Option<Symbol> = None;
            if walk(expr, values, &mut only) { only } else { None }
        }

        if let Some(sym) = single_unresolved_symbol(expression, &state.resolved_symbols) {
            let expected = expression.eval(&state.resolved_symbols);
            if let Some(v) = solve_for(&sym, &expected, &provided.to_dim()) {
                debug!("Determined symbol {sym}={v}");
                let val = v.to_i64()?;
                state.resolved_symbols.set(&sym, val);
                if state.scenario.is_none() {
                    let scope = sym.scope().with_context(|| format!(
                        "Symbol {sym:?} points to an invalid (dead ?) SymbolScope. Make sure to create symbols using the model-managed SymbolScope."
                    ))?;
                    state.scenario = scope.guess_scenario(&state.resolved_symbols)?;
                }
            }
        }
        Ok(())
    }

    pub fn set_input(&mut self, input: usize, t: TValue) -> TractResult<()> {
        let plan = self.plan.borrow();

        if input >= plan.inputs.len() {
            bail!("Invalid input id for model ({input}).");
        }

        let outlet = unsafe { *plan.inputs.get_unchecked(input) };

        if let Some(fact) = unsafe { plan.input_facts.get_unchecked(input) }.as_ref() {
            let t_shape = t.shape();
            if fact.shape.iter().count() == t_shape.len() {
                for (expected, &provided) in fact.shape.iter().zip(t_shape.iter()) {
                    Self::resolve(&mut self.session_state, expected, provided as i64)?;
                }
            }
        }

        let fact = self.plan.borrow().model().outlet_fact(outlet)?;
        ensure!(
            fact.matches(&t, Some(&self.session_state.resolved_symbols))
                .with_context(|| format!("Setting input {input}"))?,
            "Input at index {input} has incorrect dtype or shape (got {t:?}, expected to match fact {fact:?})",
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
        let &mut SimpleState { ref plan, ref mut values, .. } = self;
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
        let SimpleState { plan, values, .. } = self;
        let plan = plan.borrow();
        let nodes = plan.model().nodes();
        let node = &nodes[node];
        let mut inputs: TVec<TValue> = TVec::with_capacity(node.inputs.len());
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
        let &mut SimpleState {
            ref plan,
            ref mut session_state,
            ref mut values,
            ref mut states,
            ..
        } = self;
        let plan = plan.borrow();
        let nodes = plan.model().nodes();
        let node = &nodes[node];
        let vs = eval(session_state, states[node.id].as_deref_mut(), node, inputs)?;
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
            let mut inputs: TVec<TValue> =
                TVec::with_capacity(self.model().nodes()[node].inputs.len());
            {
                let node = &self.model().nodes()[node];
                for i in &node.inputs {
                    inputs.push(self.values[i.node].as_ref().unwrap()[i.slot].clone())
                }
            }
            let &mut Self { ref mut states, ref mut session_state, ref plan, .. } = self;
            eval(
                session_state,
                states[node].as_deref_mut(),
                &plan.borrow().model().nodes[node],
                inputs,
            )?
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

    #[inline]
    pub fn plan(&self) -> &SimplePlan<F, O, M> {
        self.plan.borrow()
    }

    #[inline]
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
            scenario: self.session_state.scenario,
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
    #[allow(clippy::let_and_return)]
    let r = match state {
        Some(ref mut state) => state.eval(session_state, node.op(), input),
        None => node.op().eval_with_session(node.id, session_state, input),
    }
    .with_context(|| format!("Evaluating {node}"));
    // eprintln!(" ==> {}", r.as_ref().unwrap()[0].dump(true)?);
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
    pub scenario: Option<usize>,
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
                scenario: self.scenario,
                tensors: self.tensors.clone(),
                cached_mmm_scratch_space: None.into(),
                scratch_extensions: anymap3::Map::new(),
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
