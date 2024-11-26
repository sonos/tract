use crate::memory::MetalMemSchema;
use crate::MetalMemoryPool;
use std::borrow::Borrow;
use tract_core::internal::*;

#[derive(Debug, Clone)]
pub struct MetalSessionHandler {
    mem_schema: MetalMemSchema,
}

impl MetalSessionHandler {
    pub fn from_plan<M, P>(plan: P, memory_hint: &SymbolValues) -> TractResult<Self>
    where
        M: Borrow<Graph<TypedFact, Box<dyn TypedOp>>>,
        P: Borrow<TypedSimplePlan<M>> + Clone,
    {
        let mem_schema = MetalMemSchema::build(
            plan.borrow().model(),
            plan.borrow().order_without_consts(),
            memory_hint,
        )?;
        Ok(Self { mem_schema })
    }
}

impl SessionStateHandler for MetalSessionHandler {
    fn before_plan_eval(&self, session_state: &mut SessionState) -> TractResult<()> {
        let resolved_mem_schema = self.mem_schema.resolve(&session_state.resolved_symbols)?;
        let memory_pool = objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT
                .with_borrow(|context| MetalMemoryPool::from_schema(context, resolved_mem_schema))
        })?;

        session_state.scratch_extensions.insert(memory_pool);
        ensure!(session_state.scratch_extensions.get::<MetalMemoryPool>().is_some());
        Ok(())
    }

    fn after_plan_eval(&self, session_state: &mut SessionState) -> TractResult<()> {
        session_state.scratch_extensions.remove::<MetalMemoryPool>();
        Ok(())
    }
}

pub fn get_metal_mem_pool(session: &SessionState) -> Option<&MetalMemoryPool> {
    session.scratch_extensions.get::<MetalMemoryPool>()
}
