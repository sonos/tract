use crate::memory::MetalMemSchema;
use crate::MetalMemoryPool;
use std::borrow::Borrow;
use tract_core::internal::*;

pub struct MetalPlanState<M, P>
where
    M: Borrow<Graph<TypedFact, Box<dyn TypedOp>>>,
    P: Borrow<TypedSimplePlan<M>> + Clone,
{
    pub mem_schema: MetalMemSchema,
    pub state: TypedSimpleState<M, P>,
}

impl<M, P> MetalPlanState<M, P>
where
    M: Borrow<Graph<TypedFact, Box<dyn TypedOp>>>,
    P: Borrow<SimplePlan<TypedFact, Box<dyn TypedOp>, M>> + Clone,
{
    pub fn new(plan: P, memory_hint: &SymbolValues) -> TractResult<Self> {
        let state = TypedSimpleState::new(plan)?;
        let mem_schema = MetalMemSchema::build(
            state.plan().model(),
            state.plan().order_without_consts(),
            memory_hint,
        )?;
        Ok(Self { state, mem_schema })
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
            &'c TypedNode,
            TVec<TValue>,
        ) -> Result<TVec<TValue>, E>,
        E: Into<anyhow::Error> + Send + Sync + 'static,
    {
        self.state.session_state = SessionState::default();
        self.state.set_inputs(inputs)?;

        let resolved_mem_schema = self
            .mem_schema
            .resolve(self.model().nodes().len(), &self.state.session_state.resolved_symbols)?;

        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let memory_pool = MetalMemoryPool::from_schema(context, resolved_mem_schema)?;
                let (_, outputs) = context.execute_in_mem_pool(memory_pool, || {
                    self.state.exec_plan_with_eval(eval)?;
                    let outputs = self.state.outputs()?;
                    self.state.reset_turn()?;
                    Ok(outputs)
                })?;
                Ok(outputs)
            })
        })
    }

    pub fn model(&self) -> &TypedModel {
        self.state.plan().model()
    }

    pub fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.run_plan_with_eval(inputs, tract_core::plan::eval)
    }

    /// Reset wires state.
    pub fn reset_turn(&mut self) -> TractResult<()> {
        self.state.reset_turn()
    }

    /// Reset op inner state.
    pub fn reset_op_states(&mut self) -> TractResult<()> {
        self.state.reset_op_states()
    }
}
