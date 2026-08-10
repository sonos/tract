use crate::device::get_context;
use crate::memory::DeviceResolvedMemSchema;
use crate::tensor::DeviceArenaView;
use crate::tensor::DeviceTensor;
use crate::tensor::OwnedDeviceTensor;

use tract_core::internal::*;

#[derive(Debug)]
pub struct DeviceMemoryPool {
    storage: Arc<Box<dyn OwnedDeviceTensor>>,
    resolved_schema: DeviceResolvedMemSchema,
}

/// Session-lived arena storage, cached in the session state across plan
/// evaluations. Without it every evaluation allocates (and wires) a fresh
/// multi-hundred-MB device buffer: over a chunked long-context prefill that
/// is gigabytes of alloc/free churn, enough to push the process into the
/// compressor and stall the next evaluations on driver re-residency.
///
/// Growth allocates 25% headroom so a prefill whose per-chunk arena grows
/// with the past-context length reallocates a handful of times instead of
/// once per chunk; when demand drops to half the cached size or less (the
/// prefill -> decode transition), the storage shrinks back to the demand.
///
/// Safety: the cached storage is only reused when this cache holds the sole
/// reference to it. Anything still alive from an earlier evaluation (an
/// escaped output view, an in-flight command buffer, a concurrently running
/// clone of the state) keeps its own `Arc`, which forces a fresh allocation
/// here instead of clobbering memory someone still reads.
#[derive(Debug, Default)]
pub struct ArenaStorageCache {
    storage: std::sync::Mutex<Option<(Arc<Box<dyn OwnedDeviceTensor>>, usize)>>,
}

impl DeviceMemoryPool {
    /// Arena allocations are rounded up to this granularity so consecutive
    /// steps of a growing-context decode request the SAME buffer size: one
    /// storage allocation then serves many steps instead of reallocating and
    /// wiring a fresh multi-MB device buffer per token (an IOGPU kernel trap
    /// on alloc and free, per step, growing with context).
    const ARENA_SIZE_BUCKET: usize = 16 * 1024 * 1024;

    fn bucketed(size: usize) -> usize {
        if size > Self::ARENA_SIZE_BUCKET {
            size.next_multiple_of(Self::ARENA_SIZE_BUCKET)
        } else {
            size
        }
    }

    pub fn from_schema_with_cache(
        resolved_schema: DeviceResolvedMemSchema,
        cache: &ArenaStorageCache,
    ) -> TractResult<Self> {
        let needed = Self::bucketed(resolved_schema.memory_size);
        let mut cached = cache.storage.lock().map_err(|e| anyhow!("{e:?}"))?;
        let storage = match &*cached {
            Some((storage, size))
                if *size >= needed
                    && *size <= needed.saturating_mul(2)
                    && Arc::strong_count(storage) == 1 =>
            {
                Arc::clone(storage)
            }
            _ => {
                // Headroom only helps the growth path; a shrink goes straight
                // to the demanded size.
                let size = if cached.as_ref().is_some_and(|(_, size)| *size < needed) {
                    Self::bucketed(needed.saturating_mul(5) / 4)
                } else {
                    needed
                };
                if std::env::var_os("TRACT_GPU_LOG_ARENA").is_some() {
                    eprintln!(
                        "arena alloc: {:.1} MB (demand {:.1} MB, cached {:.1} MB{})",
                        size as f64 / (1024.0 * 1024.0),
                        needed as f64 / (1024.0 * 1024.0),
                        cached.as_ref().map(|(_, s)| *s).unwrap_or(0) as f64 / (1024.0 * 1024.0),
                        if cached.as_ref().is_some_and(|(s, _)| Arc::strong_count(s) > 1) {
                            ", busy"
                        } else {
                            ""
                        },
                    );
                }
                let storage = Arc::new(
                    get_context()?.uninitialized_device_tensor(&[size], DatumType::U8)?,
                );
                *cached = Some((Arc::clone(&storage), size));
                storage
            }
        };
        Ok(Self { storage, resolved_schema })
    }

    pub fn from_schema(resolved_schema: DeviceResolvedMemSchema) -> TractResult<Self> {
        Self::from_schema_with_cache(resolved_schema, &ArenaStorageCache::default())
    }

    pub fn tensor_for_node(
        &self,
        node_id: usize,
        dt: DatumType,
        shape: &[usize],
    ) -> TractResult<DeviceTensor> {
        self.resolved_schema.offsets_by_node[node_id]
            .as_ref()
            .map(|offsets| {
                ensure!(
                    offsets.len() == 1 && offsets[0].len() == 1,
                    "'tensor_for_node' is for mono-output nodes only"
                );
                Ok(DeviceArenaView {
                    arena: Arc::clone(&self.storage),
                    dt,
                    len: shape.iter().product(),
                    shape: shape.into(),
                    strides: Tensor::natural_strides(shape),
                    offset_bytes: offsets[0][0],
                    exotic_fact: None,
                }
                .into())
            })
            .unwrap_or_else(|| DeviceTensor::uninitialized_dt(dt, shape))
    }

    /// Per-output variant of [`Self::tensor_for_node`] for multi-output nodes:
    /// each output slot has its own arena region in the schema.
    pub fn tensor_for_node_output(
        &self,
        node_id: usize,
        slot: usize,
        dt: DatumType,
        shape: &[usize],
    ) -> TractResult<DeviceTensor> {
        match self.resolved_schema.offsets_by_node[node_id].as_ref() {
            Some(offsets) if slot < offsets.len() && offsets[slot].len() == 1 => {
                Ok(DeviceArenaView {
                    arena: Arc::clone(&self.storage),
                    dt,
                    len: shape.iter().product(),
                    shape: shape.into(),
                    strides: Tensor::natural_strides(shape),
                    offset_bytes: offsets[slot][0],
                    exotic_fact: None,
                }
                .into())
            }
            _ => DeviceTensor::uninitialized_dt(dt, shape),
        }
    }

    pub fn scalar_exotic_tensor_for_node(
        &self,
        node_id: usize,
        dt: DatumType,
        exotic_fact: Box<dyn ExoticFact>,
    ) -> TractResult<DeviceTensor> {
        match self.resolved_schema.offsets_by_node[node_id].as_ref() {
            Some(offsets) => {
                ensure!(
                    offsets.len() == 1 && offsets[0].len() == 2,
                    "'scalar_exotic_tensor_for_node' is for mono-output nodes only"
                );
                Ok(DeviceArenaView {
                    arena: Arc::clone(&self.storage),
                    dt,
                    len: 1,
                    shape: tvec!(),
                    strides: tvec!(),
                    offset_bytes: offsets[0][1],
                    exotic_fact: Some(exotic_fact.clone()),
                }
                .into())
            }
            None => DeviceTensor::uninitialized_exotic(exotic_fact),
        }
    }
}
