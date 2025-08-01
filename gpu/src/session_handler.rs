use crate::memory::DeviceMemSchema;
use crate::memory::DeviceMemoryPool;
use crate::tensor::DeviceTensor;
use std::borrow::Borrow;
use tract_core::internal::*;

#[derive(Debug, Clone)]
pub struct DeviceSessionHandler {
    pub mem_schema: DeviceMemSchema,
}

impl DeviceSessionHandler {
    pub fn from_plan<M, P>(plan: P, memory_hint: &SymbolValues) -> TractResult<Self>
    where
        M: Borrow<Graph<TypedFact, Box<dyn TypedOp>>>,
        P: Borrow<TypedSimplePlan<M>> + Clone,
    {
        let mem_schema = DeviceMemSchema::build(
            plan.borrow().model(),
            plan.borrow().order_without_consts(),
            memory_hint,
        )?;
        Ok(Self { mem_schema })
    }
}

impl SessionStateHandler for DeviceSessionHandler {
    fn before_plan_eval(&self, session_state: &mut SessionState) -> TractResult<()> {
        let resolved_mem_schema = self.mem_schema.resolve(&session_state.resolved_symbols)?;
        let memory_pool = DeviceMemoryPool::from_schema(resolved_mem_schema)?;

        session_state.scratch_extensions.insert(memory_pool);
        ensure!(session_state.scratch_extensions.get::<DeviceMemoryPool>().is_some());
        Ok(())
    }

    fn after_plan_eval(&self, session_state: &mut SessionState) -> TractResult<()> {
        session_state.scratch_extensions.remove::<DeviceMemoryPool>();
        Ok(())
    }
}

pub fn make_tensor_for_node(
    session: &SessionState,
    node_id: usize,
    dt: DatumType,
    shape: &[usize],
) -> TractResult<DeviceTensor> {
    session
        .scratch_extensions
        .get::<DeviceMemoryPool>()
        .map(|mem| mem.tensor_for_node(node_id, dt, shape))
        .unwrap_or_else(|| unsafe { DeviceTensor::uninitialized_dt(dt, shape) })
}
