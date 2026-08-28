use crate::memory::DeviceMemSchema;
use crate::memory::DeviceMemoryPool;
use crate::tensor::DeviceTensor;
use tract_core::internal::*;

#[derive(Debug, Clone)]
pub struct DeviceTurnHandler {
    pub mem_schema: DeviceMemSchema,
}

impl DeviceTurnHandler {
    pub fn from_plan(plan: &TypedSimplePlan, memory_hint: &SymbolValues) -> TractResult<Self> {
        let mem_schema =
            DeviceMemSchema::build(plan.model(), plan.order_without_consts(), memory_hint)?;
        Ok(Self { mem_schema })
    }
}

impl TurnStateHandler for DeviceTurnHandler {
    fn before_plan_eval(&self, turn: &mut TurnState) -> TractResult<()> {
        let resolved_mem_schema = self.mem_schema.resolve(&turn.resolved_symbols)?;
        let memory_pool = DeviceMemoryPool::from_schema(resolved_mem_schema)?;

        turn.scratch_extensions.insert(memory_pool);
        ensure!(turn.scratch_extensions.get::<DeviceMemoryPool>().is_some());
        Ok(())
    }

    fn after_plan_eval(&self, turn: &mut TurnState) -> TractResult<()> {
        turn.scratch_extensions.remove::<DeviceMemoryPool>();
        Ok(())
    }
}

pub fn make_tensor_for_node(
    turn: &TurnState,
    node_id: usize,
    dt: DatumType,
    shape: &[usize],
) -> TractResult<DeviceTensor> {
    turn.scratch_extensions
        .get::<DeviceMemoryPool>()
        .map(|mem| mem.tensor_for_node(node_id, dt, shape))
        .unwrap_or_else(|| DeviceTensor::uninitialized_dt(dt, shape))
}

pub fn make_scalar_exotic_tensor_for_node(
    turn: &TurnState,
    node_id: usize,
    dt: DatumType,
    exotic_fact: Box<dyn ExoticFact>,
) -> TractResult<DeviceTensor> {
    match turn.scratch_extensions.get::<DeviceMemoryPool>() {
        Some(mem) => mem.scalar_exotic_tensor_for_node(node_id, dt, exotic_fact),
        None => DeviceTensor::uninitialized_exotic(exotic_fact),
    }
}
