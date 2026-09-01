use std::borrow::Borrow;
use std::fmt::{Debug, Display};

use multithread::Executor;

use crate::internal::*;
use crate::model::{Fact, Graph, OutletId};
use crate::ops::konst::Const;
use crate::runtime::RunOptions;

use self::order::{build_flush_list, eval_order_for_nodes, eval_order_opt_ram_for_nodes};

/// Identifies one running state, so an op can key resources it manages itself
/// per session and per node. Unique for the process; a state gets one at
/// construction and keeps it until it is dropped, at which point the plan calls
/// [`EvalOp::drop_session`] on every node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(u64);

static NEXT_SESSION_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl SessionId {
    /// For evaluating an op outside any plan -- const folding, shape inference.
    /// No state is ever built against it, so nothing keys scratch on it.
    pub const NONE: SessionId = SessionId(u64::MAX);

    fn next() -> SessionId {
        SessionId(NEXT_SESSION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

/// Where one stream's state lives inside a state shared by several streams. An
/// index into the lane axis the state's per-lane buffers are sized at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LaneId(pub usize);

/// Which lane each seat of a turn's batch carries: the index is the seat, the
/// value is the lane. `max_lanes` is the extent of the lane axis, fixed for the
/// life of the state, so an op sizing a buffer on first eval knows how wide to
/// make it -- a turn seating one lane of many still needs the full width.
///
/// A lane appears at most once, which is what makes a stream's state sequential:
/// two seats of one turn cannot both advance the same lane. A model with no
/// session-scoped state has nothing to address, so its seating is inert.
///
/// Seats and lanes both index axis 0 -- of the turn's tensors and of a laned
/// state's buffers respectively. A runtime seating more than one lane is what
/// must have checked that axis 0 of every stateful node is the model's batch
/// axis; ops only assert that their input carries one stream per seat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seating {
    lanes: Vec<LaneId>,
    max_lanes: usize,
}

impl Seating {
    pub fn new(max_lanes: usize, lanes: impl IntoIterator<Item = LaneId>) -> TractResult<Seating> {
        let lanes: Vec<LaneId> = lanes.into_iter().collect();
        for (seat, lane) in lanes.iter().enumerate() {
            ensure!(lane.0 < max_lanes, "Seat {seat} takes lane {} of {max_lanes}", lane.0);
            ensure!(!lanes[..seat].contains(lane), "Lane {} takes two seats in one turn", lane.0);
        }
        Ok(Seating { lanes, max_lanes })
    }

    /// The seating of a state one stream owns: one lane wide, that lane seated.
    /// Every turn has a seating, and this is what an unbatched turn is.
    pub fn single() -> Seating {
        Seating { lanes: vec![LaneId(0)], max_lanes: 1 }
    }

    pub fn max_lanes(&self) -> usize {
        self.max_lanes
    }

    pub fn lanes(&self) -> &[LaneId] {
        &self.lanes
    }

    /// Seats filled this turn, i.e. the extent of the batch axis of the tensors
    /// flowing through it.
    pub fn occupancy(&self) -> usize {
        self.lanes.len()
    }
}

/// Resources a [`TurnStateHandler`] installs for the turn and ops read while
/// evaluating, keyed by the stored type. This is the extension point: ops see it
/// read-only, only a handler mutates it.
pub type TurnShared = anymap3::Map<dyn std::any::Any + Send>;

/// Everything an op is given about where and when it is being evaluated. Ops
/// receive it by shared reference: they can read the turn's symbols and the
/// installed shared resources, and they can identify themselves with
/// `(session, node_id)`, but they cannot reach another node's values or rebind a
/// symbol.
#[derive(Debug, Clone, Copy)]
pub struct EvalContext<'a> {
    pub session: SessionId,
    pub node_id: usize,
    pub symbols: &'a SymbolValues,
    pub scenario: Option<usize>,
    pub shared: Option<&'a TurnShared>,
    /// Which lane each seat of this turn carries. A turn one stream owns is
    /// [`Seating::single`], so an op reads the same field either way.
    pub seating: &'a Seating,
}

pub struct TurnState {
    pub resolved_symbols: SymbolValues,
    pub scenario: Option<usize>,
    pub values: Vec<Option<TVec<TValue>>>,
    /// Resources installed for the turn by a [`TurnStateHandler`], reachable by
    /// ops through [`EvalContext::shared`]. Entries outlive the turn --
    /// `reset_turn` does not touch them -- so what a handler installs is reused
    /// turn after turn unless it drops it in `after_plan_eval`.
    pub shared: TurnShared,
    /// Which lane each seat of this turn carries. A laned runtime sets it before
    /// each turn; a state one stream owns keeps [`Seating::single`].
    pub seating: Seating,
}

impl EvalContext<'static> {
    /// Context for evaluating outside any plan -- const folding and shape
    /// inference. What [`EvalOp::eval_out_of_plan`] hands the op: no symbol is
    /// bound and no shared resource is reachable.
    pub fn out_of_plan() -> EvalContext<'static> {
        static SYMBOLS: std::sync::OnceLock<SymbolValues> = std::sync::OnceLock::new();
        static SEATING: std::sync::OnceLock<Seating> = std::sync::OnceLock::new();
        EvalContext {
            session: SessionId::NONE,
            node_id: usize::MAX,
            symbols: SYMBOLS.get_or_init(SymbolValues::default),
            scenario: None,
            shared: None,
            seating: SEATING.get_or_init(Seating::single),
        }
    }
}

impl TurnState {
    pub fn context(&self, session: SessionId, node_id: usize) -> EvalContext<'_> {
        EvalContext {
            session,
            node_id,
            symbols: &self.resolved_symbols,
            scenario: self.scenario,
            shared: Some(&self.shared),
            seating: &self.seating,
        }
    }
}

impl Default for TurnState {
    fn default() -> Self {
        TurnState {
            resolved_symbols: SymbolValues::default(),
            scenario: None,
            values: vec![],
            shared: TurnShared::new(),
            seating: Seating::single(),
        }
    }
}

impl Clone for TurnState {
    fn clone(&self) -> Self {
        TurnState {
            resolved_symbols: self.resolved_symbols.clone(),
            scenario: self.scenario,
            values: vec![],
            shared: TurnShared::new(),
            seating: self.seating.clone(),
        }
    }
}

pub trait TurnStateHandler: Send + Sync + Debug {
    fn before_plan_eval(&self, turn: &mut TurnState) -> TractResult<()>;
    fn after_plan_eval(&self, turn: &mut TurnState) -> TractResult<()>;
}

impl Debug for TurnState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TurnState({:?})", self.resolved_symbols)
    }
}

#[derive(Debug, Clone)]
pub struct SimplePlan<F, O>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    pub(crate) model: Arc<Graph<F, O>>,
    outputs: Vec<OutletId>,
    order: Vec<usize>,
    flush_lists: Vec<TVec<usize>>,
    has_unresolved_symbols: bool,
    symbols: Vec<Symbol>,
    executor: Option<Executor>,
    turn_handler: Option<Arc<dyn TurnStateHandler + 'static>>,
}

impl<F, O> SimplePlan<F, O>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    /// This contructor returns a plan that will compute all the model default outputs in one pass.
    pub fn new(model: impl Into<Arc<Graph<F, O>>>) -> TractResult<Arc<SimplePlan<F, O>>> {
        let model = model.into();
        Self::build(model, &RunOptions::default()).map(Arc::new)
    }

    /// This contructor returns a plan that will compute all the model default outputs in one pass.
    pub fn new_with_options(
        model: impl Into<Arc<Graph<F, O>>>,
        options: &RunOptions,
    ) -> TractResult<Arc<SimplePlan<F, O>>> {
        let model = model.into();
        Self::build(model, options).map(Arc::new)
    }

    /// This contructor returns a plan that will compute the specified output.
    #[deprecated]
    pub fn new_for_output(
        model: Graph<F, O>,
        output: OutletId,
    ) -> TractResult<Arc<SimplePlan<F, O>>> {
        #[allow(deprecated)]
        Self::build_with_outputs_and_deps(model, &[output], &[], &RunOptions::default())
            .map(Arc::new)
    }

    /// This contructor returns a plan that will compute all specified outputs in one pass.
    #[deprecated]
    pub fn new_for_outputs(
        model: impl Into<Arc<Graph<F, O>>>,
        outputs: &[OutletId],
    ) -> TractResult<Arc<SimplePlan<F, O>>> {
        #[allow(deprecated)]
        Self::build_with_outputs_and_deps(model, outputs, &[], &RunOptions::default()).map(Arc::new)
    }

    pub fn with_turn_handler<H: TurnStateHandler + 'static>(mut self, turn_handler: H) -> Self {
        self.turn_handler = Some(Arc::new(turn_handler));
        self
    }

    #[deprecated]
    pub fn new_for_outputs_and_deps(
        model: impl Into<Arc<Graph<F, O>>>,
        outputs: &[OutletId],
        deps: &[(usize, usize)],
    ) -> TractResult<Arc<SimplePlan<F, O>>> {
        #[allow(deprecated)]
        Self::build_with_outputs_and_deps(model, outputs, deps, &RunOptions::default())
            .map(Arc::new)
    }

    pub fn build(
        model: impl Into<Arc<Graph<F, O>>>,
        options: &RunOptions,
    ) -> TractResult<SimplePlan<F, O>> {
        let model = model.into();
        let outputs = model.outputs.clone();
        #[allow(deprecated)]
        Self::build_with_outputs_and_deps(model, &outputs, &[], options)
    }

    #[deprecated]
    pub fn build_with_outputs_and_deps(
        model: impl Into<Arc<Graph<F, O>>>,
        outputs: &[OutletId],
        deps: &[(usize, usize)],
        options: &RunOptions,
    ) -> TractResult<SimplePlan<F, O>> {
        let model = model.into();
        let inputs = model.input_outlets()?.iter().map(|n| n.node).collect::<Vec<usize>>();
        let outputs_nodes = outputs.iter().map(|n| n.node).collect::<Vec<usize>>();
        let mut order = if options.skip_order_opt_ram {
            eval_order_for_nodes(model.nodes(), &inputs, &outputs_nodes, deps)?
        } else {
            eval_order_opt_ram_for_nodes(model.nodes(), &inputs, &outputs_nodes, deps)?
        };
        order.retain(|node| !model.node(*node).op_is::<Const>());
        let flush_lists = build_flush_list(&*model, &order, outputs, |n| !n.op_is::<Const>());

        #[allow(clippy::mutable_key_type)]
        let mut symbols: std::collections::HashSet<Symbol> = Default::default();
        for node in &model.nodes {
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
            symbols: symbols.into_iter().collect(),
            executor: options.executor.clone(),
            turn_handler: None,
        })
    }

    pub fn order_without_consts(&self) -> &[usize] {
        &self.order
    }

    pub fn run(self: &Arc<Self>, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut state = self.spawn()?;
        state.run(inputs)
    }

    pub fn model(&self) -> &Graph<F, O> {
        self.model.borrow()
    }

    pub fn spawn(self: &Arc<Self>) -> TractResult<SimpleState<F, O>> {
        SimpleState::new(self)
    }
}

#[derive(Debug)]
pub struct SimpleState<F, O>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    pub(crate) plan: Arc<SimplePlan<F, O>>,
    pub op_states: Vec<Option<Box<dyn OpState>>>,
    pub turn_state: TurnState,
    session: SessionId,
}

/// A clone is a distinct session: it gets its own [`SessionId`], so whatever the
/// ops key on `(session, node_id)` stays separate from the original's.
impl<F, O> Clone for SimpleState<F, O>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn clone(&self) -> Self {
        SimpleState {
            plan: self.plan.clone(),
            op_states: self.op_states.clone(),
            turn_state: self.turn_state.clone(),
            session: SessionId::next(),
        }
    }
}

impl<F, O> Drop for SimpleState<F, O>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn drop(&mut self) {
        for (ix, node) in self.plan.model.nodes.iter().enumerate() {
            node.op().drop_session(self.session, ix);
        }
    }
}

impl<F, O> SimpleState<F, O>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    pub fn new(plan: &Arc<SimplePlan<F, O>>) -> TractResult<SimpleState<F, O>> {
        let plan = Arc::clone(plan);
        let turn = TurnState::default();
        let model = plan.model();
        let states: Vec<Option<Box<dyn OpState>>> = vec![None; model.nodes.len()];
        let mut state =
            SimpleState { plan, op_states: states, turn_state: turn, session: SessionId::next() };
        state.reset_op_states()?;
        Ok(state)
    }

    pub fn new_from_inputs(
        plan: &Arc<SimplePlan<F, O>>,
        inputs: TVec<TValue>,
    ) -> TractResult<SimpleState<F, O>> {
        let mut state = SimpleState::new(plan)?;
        state.set_inputs(inputs)?;
        state.resolve_symbols_with_states()?;

        Ok(state)
    }

    fn ready_turn(&mut self) {
        if self.turn_state.values.len() == 0 {
            self.turn_state.values = vec![None; self.plan.model.nodes().len()];
            for node in &self.plan.model.nodes {
                if let Some(k) = node.op_as::<Const>() {
                    self.turn_state.values[node.id] = Some(tvec!(k.val().clone().into_tvalue()));
                }
            }
        }
    }
    /// Reset wires state.
    pub fn reset_turn(&mut self) -> TractResult<()> {
        self.reset_turn_keep_symbols();
        self.turn_state.resolved_symbols = SymbolValues::default();
        Ok(())
    }

    /// Like [`reset_turn`] but keeps the resolved symbols (and scenario). Used by
    /// `Scan`/`Loop` bodies, whose shapes are constant across iterations: it lets
    /// the body resolve its symbols once and skip the per-iteration re-resolution
    /// the full `reset_turn` + `run` cycle would otherwise force.
    pub(crate) fn reset_turn_keep_symbols(&mut self) {
        for node in &self.plan.order {
            self.turn_state.values[*node] = None;
        }
    }

    /// Clear resolved symbols (and scenario) without touching node values. Used at
    /// the start of a fresh `Scan` evaluation, since the body state persists across
    /// outer calls and a previous call may have left stale symbol resolutions.
    pub(crate) fn clear_resolved_symbols(&mut self) {
        self.turn_state.resolved_symbols = SymbolValues::default();
        self.turn_state.scenario = None;
    }

    /// Seat the lanes carrying the coming turn's streams, one lane per row of
    /// axis 0 of its tensors.
    pub fn seat(&mut self, seating: Seating) {
        self.turn_state.seating = seating;
    }

    /// Drop the session state `lanes` hold, handing them to new streams.
    pub fn reset_lanes(&mut self, lanes: &[LaneId]) -> TractResult<()> {
        for op_state in self.op_states.iter_mut().flatten() {
            op_state.reset_lanes(lanes)?;
        }
        Ok(())
    }

    /// Reset op inner state.
    fn reset_op_states(&mut self) -> TractResult<()> {
        let &mut SimpleState {
            ref plan, ref turn_state, op_states: ref mut states, session, ..
        } = self;
        for (ix, n) in plan.model.nodes.iter().enumerate() {
            states[ix] = n.op().state(&turn_state.context(session, ix))?;
        }
        Ok(())
    }

    pub(crate) fn resolve_symbols_with_states(&mut self) -> TractResult<()> {
        for state in self
            .op_states
            .iter_mut()
            .filter_map(Option::as_mut)
            .filter(|s| s.has_init_tensor_fact())
        {
            state.resolve_symbols(&mut self.turn_state)?;
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
            &'a EvalContext<'a>,
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
            &'a EvalContext<'a>,
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
            &'a EvalContext<'a>,
            Option<&'b mut (dyn OpState + 'static)>,
            &'c Node<F, O>,
            TVec<TValue>,
        ) -> Result<TVec<TValue>, E>,
        E: Into<anyhow::Error> + Send + Sync + 'static,
    {
        {
            self.ready_turn();
            self.plan
                .turn_handler
                .as_ref()
                .map(|it| it.before_plan_eval(&mut self.turn_state))
                .transpose()?;

            let mut syms_done = !self.plan.has_unresolved_symbols
                || self
                    .plan
                    .symbols
                    .iter()
                    .all(|s| self.turn_state.resolved_symbols.get(s).is_some());

            for (step, n) in self.plan.order.iter().enumerate() {
                let node = self.plan.model.node(*n);
                trace!("Running step {step}, node {node}");
                let mut inputs: TVec<TValue> = tvec![];
                for i in &node.inputs {
                    trace!("  use input {i:?}");
                    let prec_node = self.plan.model.node(i.node);
                    let prec = self.turn_state.values[i.node].as_ref().ok_or_else(|| {
                        format_err!("Computing {}, precursor {} not done:", node, prec_node)
                    })?;
                    inputs.push(prec[i.slot].clone())
                }
                for flush in &self.plan.flush_lists[step] {
                    trace!("  Ran {} can now flush {}", node, self.plan.model.node(*flush));
                    self.turn_state.values[*flush] = None;
                }

                if cfg!(debug_assertions) {
                    let facts = self.plan.model.node_input_facts(node.id)?;
                    if facts.len() != inputs.len() {
                        bail!(
                            "Evaluating {}: expected {} inputs, got {}",
                            node,
                            facts.len(),
                            inputs.len()
                        );
                    }
                    for (ix, (v, f)) in inputs.iter().zip(facts.iter()).enumerate() {
                        if !f.matches(v, Some(&self.turn_state.resolved_symbols))? {
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

                // A node with no precursors whose value is already set is a model
                // input: `set_inputs` wrote it and `reset_turn` clears everything
                // else, so hand it in rather than have the op reach for it. After
                // the checks above, which are stated against the node's declared
                // inputs -- a source declares none.
                if node.inputs.is_empty()
                    && let Some(preset) = self.turn_state.values[*n].as_ref()
                {
                    inputs = preset.clone();
                }

                let ctx = self.turn_state.context(self.session, node.id);
                let vs = eval(&ctx, self.op_states[node.id].as_deref_mut(), node, inputs)
                    .map_err(|e| e.into())?;

                if !syms_done && self.plan.has_unresolved_symbols {
                    for (o, v) in node.outputs.iter().zip(vs.iter()) {
                        if let Ok(f) = o.fact.to_typed_fact() {
                            for (dim_abstract, dim_concrete) in f.shape.iter().zip(v.shape()) {
                                Self::resolve(
                                    &mut self.turn_state,
                                    dim_abstract,
                                    *dim_concrete as i64,
                                )?;
                            }
                        }
                    }
                    if self
                        .plan
                        .symbols
                        .iter()
                        .all(|s| self.turn_state.resolved_symbols.get(s).is_some())
                    {
                        syms_done = true;
                    }
                }
                if cfg!(debug_assertions) {
                    let facts = self.plan.model.node_output_facts(node.id)?;
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
                        if !f.matches(v, Some(&self.turn_state.resolved_symbols))? {
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

                self.turn_state.values[node.id] = Some(vs);
            }
            self.plan
                .turn_handler
                .as_ref()
                .map(|it| it.after_plan_eval(&mut self.turn_state))
                .transpose()?;
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

    /// Like [`set_inputs`] but drains the caller's buffer (leaving it empty with
    /// its capacity intact) instead of consuming it, so a repeated caller (a
    /// `Scan` body loop) can reuse one allocation across iterations.
    pub(crate) fn set_inputs_drain(&mut self, inputs: &mut TVec<TValue>) -> TractResult<()> {
        ensure!(
            inputs.len() == self.model().inputs.len(),
            "Wrong number of inputs for model. Expected {} got {}",
            self.model().inputs.len(),
            inputs.len()
        );
        for (ix, t) in inputs.drain(..).enumerate() {
            self.set_input(ix, t)?
        }
        Ok(())
    }

    fn resolve(state: &mut TurnState, expression: &TDim, provided: i64) -> TractResult<()> {
        if let TDim::Sym(sym) = expression
            && state.resolved_symbols.get(sym).is_none()
        {
            state.resolved_symbols.set(sym, provided);
            if state.scenario.is_none() {
                let scope = sym.scope().with_context(|| {
                    format!(
                        "Symbol {sym:?} points to an invalid (dead ?) SymbolScope. \
                         Make sure to create symbols using the model-managed SymbolScope."
                    )
                })?;
                state.scenario = scope.guess_scenario(&state.resolved_symbols)?;
            }
            return Ok(());
        }
        let expected = expression.eval(&state.resolved_symbols);
        if let Some(x) = expected.as_i64()
            && x != provided
        {
            bail!("Clashing resolution for expression. {expression}={x} != {provided}. ({state:?})")
        }
        if expected.symbols().len() == 1 {
            let sym = expected.symbols().into_iter().next().unwrap();
            if let Some(v) = solve_for(&sym, &expected, &provided.to_dim()) {
                debug!("Determined symbol {sym}={v}");
                state.resolved_symbols.set(&sym, v.to_i64().unwrap());
            }
            if state.scenario.is_none() {
                let scope = sym
                    .scope()
                    .with_context(|| format!("Symbol {sym:?} points to an invalid (dead ?) SymbolScope. Make sure to create symbols using the model-managed SymbolScope."))?;
                state.scenario = scope.guess_scenario(&state.resolved_symbols)?;
            }
        }
        Ok(())
    }

    pub fn set_input(&mut self, input: usize, t: TValue) -> TractResult<()> {
        let outlet: OutletId = *self
            .model()
            .input_outlets()?
            .get(input)
            .with_context(|| format!("Invalid input id for model ({input})."))?;
        if let Ok(fact) = self.plan.model.outlet_fact(outlet)?.to_typed_fact() {
            for (expected, provided) in fact.shape.iter().zip(t.shape()) {
                Self::resolve(&mut self.turn_state, expected, *provided as i64)?;
            }
        }
        let fact = self.plan.model.outlet_fact(outlet)?;
        ensure!(
            fact.matches(&t, Some(&self.turn_state.resolved_symbols))
                .with_context(|| format!("Setting input {input}"))?,
            "Input at index {input} has incorrect dtype or shape (got {t:?}, expected to match fact {fact:?})",
        );
        self.ready_turn();
        self.turn_state.values[outlet.node] = Some(tvec!(t));
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
            .turn_state
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
        let &mut SimpleState { ref plan, ref mut turn_state, .. } = self;
        let mut v = tvec![];
        for o in plan.outputs.iter() {
            let vs = turn_state.values[o.node].as_mut().ok_or_else(|| {
                format_err!("Outputs of {:?} are not computed", plan.model.nodes()[o.node])
            })?;
            v.push(vs[o.slot].clone())
        }
        Ok(v)
    }

    pub fn set_values(&mut self, id: usize, values: TVec<TValue>) -> TractResult<()> {
        self.turn_state.values[id] = Some(values);
        Ok(())
    }

    pub fn set_value(&mut self, id: usize, value: TValue) -> TractResult<()> {
        self.set_values(id, tvec!(value))
    }

    pub fn prepare_inputs(&self, node: usize) -> TractResult<TVec<TValue>> {
        let SimpleState { plan, turn_state, .. } = self;
        let nodes = plan.model.nodes();
        let node = &nodes[node];
        let mut inputs: TVec<TValue> = tvec![];
        for i in &node.inputs {
            let prec_node = &nodes[i.node];
            let prec = turn_state.values[i.node].as_ref().ok_or_else(|| {
                format_err!("Computing {}, precursor {} not done.", node, prec_node)
            })?;
            inputs.push(prec[i.slot].clone())
        }
        if node.inputs.is_empty()
            && let Some(preset) = turn_state.values[node.id].as_ref()
        {
            inputs = preset.clone();
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
            ref mut turn_state,
            op_states: ref mut states,
            session,
            ..
        } = self;
        let nodes = plan.model.nodes();
        let node = &nodes[node];
        let ctx = turn_state.context(session, node.id);
        let vs = eval(&ctx, states[node.id].as_deref_mut(), node, inputs)?;
        turn_state.values[node.id] = Some(vs);
        Ok(())
    }

    pub fn compute_recursively(&mut self, node: usize) -> TractResult<&[TValue]> {
        let values = {
            #[allow(clippy::needless_collect)] // clippy bug ?
            let precs: Vec<usize> =
                self.model().nodes()[node].inputs.iter().map(|i| i.node).collect();
            for i in precs.into_iter() {
                if self.turn_state.values[i].is_none() {
                    let _ = self.compute_recursively(i)?;
                }
            }
            let mut inputs: TVec<TValue> = tvec![];
            {
                let node = &self.model().nodes()[node];
                for i in &node.inputs {
                    inputs.push(self.turn_state.values[i.node].as_ref().unwrap()[i.slot].clone())
                }
                if node.inputs.is_empty()
                    && let Some(preset) = self.turn_state.values[node.id].as_ref()
                {
                    inputs = preset.clone();
                }
            }
            let &mut Self {
                op_states: ref mut states,
                turn_state: ref mut turn,
                ref plan,
                session,
                ..
            } = self;
            let ctx = turn.context(session, node);
            eval(&ctx, states[node].as_deref_mut(), &plan.model().nodes[node], inputs)?
        };
        self.turn_state.values[node] = Some(values);
        Ok(self.turn_state.values[node].as_ref().unwrap())
    }

    pub fn take_by_name(&mut self, name: &str) -> TractResult<TVec<Tensor>> {
        let id = self.model().node_by_name(name)?.id;
        Self::take(self, id)
    }

    pub fn take(&mut self, id: usize) -> TractResult<TVec<Tensor>> {
        Ok(self.turn_state.values[id]
            .take()
            .ok_or_else(|| format_err!("Node is not computed"))?
            .into_iter()
            .map(|v| v.into_tensor())
            .collect())
    }

    pub fn plan(&self) -> &Arc<SimplePlan<F, O>> {
        &self.plan
    }

    pub fn model(&self) -> &Graph<F, O> {
        &self.plan.model
    }
}

pub fn eval<F, O>(
    ctx: &EvalContext,
    mut state: Option<&mut (dyn OpState + 'static)>,
    node: &Node<F, O>,
    input: TVec<TValue>,
) -> TractResult<TVec<TValue>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    match state {
        Some(ref mut state) => state.eval(ctx, node.op(), input),
        None => node.op().eval(ctx, input),
    }
    .with_context(|| format!("Evaluating {node}"))
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
        is_send::<TypedSimplePlan>();
    }

    #[test]
    fn type_plan_is_sync() {
        is_sync::<TypedSimplePlan>();
    }

    #[test]
    fn type_state_is_send() {
        is_send::<TypedSimpleState>();
    }
}
