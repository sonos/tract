use crate::memory::ArenaStorageCache;
use crate::memory::DeviceMemSchema;
use crate::memory::DeviceMemoryPool;
use crate::tensor::DeviceTensor;
use tract_core::internal::*;

tract_core::declare_knob!(
    scopable TRACT_GPU_RETAIN_ARENA,
    bool,
    false,
    "Keep a state's arena storage allocated between turns instead of allocating it on every turn."
);

#[derive(Debug, Clone)]
pub struct DeviceTurnHandler {
    pub mem_schema: DeviceMemSchema,
    /// Read from [`TRACT_GPU_RETAIN_ARENA`] when the handler is built.
    pub retain_arena: bool,
}

impl DeviceTurnHandler {
    pub fn from_plan(plan: &TypedSimplePlan, memory_hint: &SymbolValues) -> TractResult<Self> {
        let mem_schema =
            DeviceMemSchema::build(plan.model(), plan.order_without_consts(), memory_hint)?;
        Ok(Self { mem_schema, retain_arena: TRACT_GPU_RETAIN_ARENA.get() })
    }

    /// The arena a runtime installs on `plan`, sized by [`schema_hints`]. wgpu
    /// does not call it: its dispatches cannot bind two regions of one arena
    /// buffer as a read and a write, so it gives each partition a buffer of its
    /// own instead.
    pub fn for_plan(
        plan: &TypedSimplePlan,
        hints: Option<&SymbolValues>,
    ) -> TractResult<Option<Self>> {
        let Some(hints) = schema_hints(plan, hints) else { return Ok(None) };
        Self::from_plan(plan, &hints).context("While sizing memory arena. Missing hint ?").map(Some)
    }
}

/// What a memory schema for `plan` is sized by: `hints` when given, and none
/// when every shape in the model is concrete. A symbolic model with no hints
/// gets no schema, and its device tensors are allocated one by one.
pub fn schema_hints(plan: &TypedSimplePlan, hints: Option<&SymbolValues>) -> Option<SymbolValues> {
    match hints {
        Some(hints) => Some(hints.clone()),
        None if plan
            .model()
            .nodes
            .iter()
            .all(|n| n.outputs.iter().all(|o| o.fact.shape.is_concrete())) =>
        {
            Some(SymbolValues::default())
        }
        None => None,
    }
}

impl TurnStateHandler for DeviceTurnHandler {
    fn before_plan_eval(&self, turn: &mut TurnState) -> TractResult<()> {
        let resolved_mem_schema = self.mem_schema.resolve(&turn.resolved_symbols)?;
        let memory_pool = if self.retain_arena {
            if turn.shared.get::<ArenaStorageCache>().is_none() {
                turn.shared.insert(ArenaStorageCache::default());
            }
            let cache = turn.shared.get::<ArenaStorageCache>().unwrap();
            DeviceMemoryPool::from_schema_with_cache(resolved_mem_schema, cache)?
        } else {
            DeviceMemoryPool::from_schema(resolved_mem_schema)?
        };

        turn.shared.insert(memory_pool);
        ensure!(turn.shared.get::<DeviceMemoryPool>().is_some());
        Ok(())
    }

    fn after_plan_eval(&self, turn: &mut TurnState) -> TractResult<()> {
        turn.shared.remove::<DeviceMemoryPool>();
        Ok(())
    }
}

pub fn make_tensor_for_node(
    ctx: &EvalContext,
    dt: DatumType,
    shape: &[usize],
) -> TractResult<DeviceTensor> {
    ctx.shared
        .and_then(|s| s.get::<DeviceMemoryPool>())
        .map(|mem| mem.tensor_for_node(ctx.node_id, dt, shape))
        .unwrap_or_else(|| DeviceTensor::uninitialized_dt(dt, shape))
}

/// Like [`make_tensor_for_node`] but for one output slot of a multi-output
/// node: the memory schema reserves one arena region per device output.
pub fn make_tensor_for_node_output(
    ctx: &EvalContext,
    slot: usize,
    dt: DatumType,
    shape: &[usize],
) -> TractResult<DeviceTensor> {
    ctx.shared
        .and_then(|s| s.get::<DeviceMemoryPool>())
        .map(|mem| mem.tensor_for_node_output(ctx.node_id, slot, dt, shape))
        .unwrap_or_else(|| DeviceTensor::uninitialized_dt(dt, shape))
}

pub fn make_scalar_exotic_tensor_for_node(
    ctx: &EvalContext,
    dt: DatumType,
    exotic_fact: Box<dyn ExoticFact>,
) -> TractResult<DeviceTensor> {
    match ctx.shared.and_then(|s| s.get::<DeviceMemoryPool>()) {
        Some(mem) => mem.scalar_exotic_tensor_for_node(ctx.node_id, dt, exotic_fact),
        None => DeviceTensor::uninitialized_exotic(exotic_fact),
    }
}
