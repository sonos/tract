use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tract_core::internal::*;
use tract_gpu::memory::{DeviceMemSchema, DeviceMemoryPool, SchemaOptions};
use tract_gpu::turn_handler::schema_hints;

use crate::context::with_wgpu_queue;

/// Plans a model's device memory as tract-gpu's arena does, but gives each partition a buffer of
/// its own: a dispatch cannot bind two regions of one buffer as a read and a write. A node's
/// inputs are alive while it runs, so they never share a partition with its outputs. The buffers
/// stay with the thread for the plan's next turn, and the bind groups built on them with it.
#[derive(Debug)]
pub struct WgpuTurnHandler {
    schema: DeviceMemSchema,
    /// Names the plan's buffers in the queue, which lets them go once the plan is dropped.
    token: Arc<u64>,
}

impl WgpuTurnHandler {
    pub fn for_plan(
        plan: &TypedSimplePlan,
        hints: Option<&SymbolValues>,
    ) -> TractResult<Option<Self>> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let Some(hints) = schema_hints(plan, hints) else { return Ok(None) };
        let options = SchemaOptions {
            plan_host_synced: false,
            exclude_outputs: true,
            min_shared_size: 1 << 16,
        };
        let schema =
            DeviceMemSchema::build_with(plan.model(), plan.order_without_consts(), &hints, options)
                .context("While planning device memory. Missing hint ?")?;
        Ok(Some(Self { schema, token: Arc::new(NEXT.fetch_add(1, Ordering::Relaxed)) }))
    }
}

impl TurnStateHandler for WgpuTurnHandler {
    fn before_plan_eval(&self, turn: &mut TurnState) -> TractResult<()> {
        let resolved = self.schema.resolve_partitions(&turn.resolved_symbols)?;
        let buffers = with_wgpu_queue(|q| Ok(q.partition_buffers(&self.token, &resolved.sizes)))?;
        turn.shared.insert(DeviceMemoryPool::from_partitions(resolved, buffers)?);
        Ok(())
    }

    fn after_plan_eval(&self, turn: &mut TurnState) -> TractResult<()> {
        turn.shared.remove::<DeviceMemoryPool>();
        Ok(())
    }
}
