use crate::memory::DeviceMemSchema;
use crate::memory::DeviceMemoryPool;
use crate::tensor::DeviceTensor;
use tract_core::internal::*;

#[derive(Debug, Clone)]
pub struct DeviceSessionHandler {
    pub mem_schema: DeviceMemSchema,
    /// Set once the arena-resolution failure has been reported, so a decode
    /// loop does not log the same error on every step.
    arena_failure_logged: Arc<std::sync::atomic::AtomicBool>,
}

impl DeviceSessionHandler {
    pub fn from_plan(plan: &TypedSimplePlan, memory_hint: &SymbolValues) -> TractResult<Self> {
        let mem_schema =
            DeviceMemSchema::build(plan.model(), plan.order_without_consts(), memory_hint)?;
        Ok(Self { mem_schema, arena_failure_logged: Arc::new(false.into()) })
    }
}

impl SessionStateHandler for DeviceSessionHandler {
    fn before_plan_eval(&self, session_state: &mut TurnState) -> TractResult<()> {
        // A schema that cannot be resolved yet (e.g. a symbol only known
        // mid-eval) is not a correctness error: ops fall back to per-node
        // allocation when no memory pool is installed. It is a heavy
        // performance loss though, so report it loudly (once per session),
        // and hard-error under TRACT_GPU_STRICT_ARENA=1.
        let resolved_mem_schema = match self.mem_schema.resolve(&session_state.resolved_symbols) {
            Ok(schema) => schema,
            Err(e) => {
                if std::env::var("TRACT_GPU_STRICT_ARENA").is_ok_and(|v| v == "1") {
                    return Err(e.context(
                        "TRACT_GPU_STRICT_ARENA=1: device memory arena resolution failed",
                    ));
                }
                if !self.arena_failure_logged.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    log::error!(
                        "Device memory arena disabled for this session, ops fall back to \
                         per-node allocations (expect a performance loss): {e:?}"
                    );
                }
                return Ok(());
            }
        };
        // Arena observability: per-evaluation resolved size. Measured on
        // qwen3.5-35B @11k context (2026-08-14) that a context-bucketed
        // schema ladder would shave only ~0.4% off the peak arena (the
        // largest transients sort first under any packing hint), so the
        // single hint-built schema stays; this log is what measured it.
        if std::env::var("TRACT_GPU_LOG_ARENA").is_ok_and(|v| v == "1") {
            log::info!("device memory arena size: {} bytes", resolved_mem_schema.memory_size);
        }
        // The storage cache lives in the session scratch slot, which survives
        // both plan evaluations and freeze/unfreeze cycles (the tract API
        // freezes the state between every call), so consecutive evaluations
        // reuse one storage allocation.
        let cache = match &session_state.session_scratch {
            Some(scratch) => scratch.clone(),
            None => {
                let cache: Arc<dyn std::any::Any + Send + Sync> =
                    Arc::new(crate::memory::ArenaStorageCache::default());
                session_state.session_scratch = Some(cache.clone());
                cache
            }
        };
        let memory_pool = match cache.downcast_ref::<crate::memory::ArenaStorageCache>() {
            Some(cache) => DeviceMemoryPool::from_schema_with_cache(resolved_mem_schema, cache)?,
            // Someone else owns the scratch slot: run with a per-evaluation
            // storage rather than fighting over it.
            None => DeviceMemoryPool::from_schema(resolved_mem_schema)?,
        };

        session_state.scratch_extensions.insert(memory_pool);
        ensure!(session_state.scratch_extensions.get::<DeviceMemoryPool>().is_some());
        Ok(())
    }

    fn after_plan_eval(&self, session_state: &mut TurnState) -> TractResult<()> {
        session_state.scratch_extensions.remove::<DeviceMemoryPool>();
        Ok(())
    }
}

pub fn make_tensor_for_node(
    session: &TurnState,
    node_id: usize,
    dt: DatumType,
    shape: &[usize],
) -> TractResult<DeviceTensor> {
    session
        .scratch_extensions
        .get::<DeviceMemoryPool>()
        .map(|mem| mem.tensor_for_node(node_id, dt, shape))
        .unwrap_or_else(|| DeviceTensor::uninitialized_dt(dt, shape))
}

/// Like [`make_tensor_for_node`] but for one output slot of a multi-output
/// node: the memory schema reserves one arena region per device output.
pub fn make_tensor_for_node_output(
    session: &TurnState,
    node_id: usize,
    slot: usize,
    dt: DatumType,
    shape: &[usize],
) -> TractResult<DeviceTensor> {
    session
        .scratch_extensions
        .get::<DeviceMemoryPool>()
        .map(|mem| mem.tensor_for_node_output(node_id, slot, dt, shape))
        .unwrap_or_else(|| DeviceTensor::uninitialized_dt(dt, shape))
}

pub fn make_scalar_exotic_tensor_for_node(
    session: &TurnState,
    node_id: usize,
    dt: DatumType,
    exotic_fact: Box<dyn ExoticFact>,
) -> TractResult<DeviceTensor> {
    match session.scratch_extensions.get::<DeviceMemoryPool>() {
        Some(mem) => mem.scalar_exotic_tensor_for_node(node_id, dt, exotic_fact),
        None => DeviceTensor::uninitialized_exotic(exotic_fact),
    }
}
