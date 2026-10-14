use crate::device::get_context;
use crate::memory::DeviceResolvedMemSchema;
use crate::tensor::DeviceTensor;
use crate::tensor::DeviceView;
use crate::tensor::OwnedDeviceTensor;

use tract_core::internal::*;

#[derive(Debug)]
pub struct DeviceMemoryPool {
    storage: Arc<Box<dyn OwnedDeviceTensor>>,
    resolved_schema: DeviceResolvedMemSchema,
}

/// Reusable arena storage held in a turn's shared resources.
///
/// Storage is reused only when the cache is its sole owner. Live arena views
/// retain their backing allocation and force a new allocation on the next turn.
#[derive(Debug, Default)]
pub(crate) struct ArenaStorageCache {
    storage: std::sync::Mutex<Option<ArenaStorage>>,
}

#[derive(Debug)]
struct ArenaStorage {
    tensor: Arc<Box<dyn OwnedDeviceTensor>>,
    capacity: usize,
}

impl DeviceMemoryPool {
    /// Reuses storage when it has no live views and capacity is within twice
    /// the resolved demand. Growth reserves 25% headroom; shrinking below half
    /// the cached capacity releases it.
    pub(crate) fn from_schema_with_cache(
        resolved_schema: DeviceResolvedMemSchema,
        cache: &ArenaStorageCache,
    ) -> TractResult<Self> {
        let needed = resolved_schema.memory_size;
        let mut cached = cache.storage.lock().map_err(|e| anyhow!("{e:?}"))?;
        let storage = match &*cached {
            Some(storage)
                if storage.capacity >= needed
                    && storage.capacity <= needed.saturating_mul(2)
                    && Arc::strong_count(&storage.tensor) == 1 =>
            {
                Arc::clone(&storage.tensor)
            }
            _ => {
                let capacity = if cached.as_ref().is_some_and(|s| s.capacity < needed) {
                    needed.saturating_add(needed / 4)
                } else {
                    needed
                };
                let tensor = Arc::new(
                    get_context()?.uninitialized_device_tensor(&[capacity], DatumType::U8)?,
                );
                *cached = Some(ArenaStorage { tensor: Arc::clone(&tensor), capacity });
                tensor
            }
        };
        Ok(Self { storage, resolved_schema })
    }

    pub fn from_schema(resolved_schema: DeviceResolvedMemSchema) -> TractResult<Self> {
        Ok(Self {
            storage: Arc::new(
                get_context()?
                    .uninitialized_device_tensor(&[resolved_schema.memory_size], DatumType::U8)?,
            ),
            resolved_schema,
        })
    }

    pub fn tensor_for_node(
        &self,
        node_id: usize,
        dt: DatumType,
        shape: &[usize],
    ) -> TractResult<DeviceTensor> {
        if let Some(offsets) = self.resolved_schema.offsets_by_node[node_id].as_ref() {
            ensure!(offsets.len() == 1, "'tensor_for_node' is for mono-output nodes only");
        }
        self.tensor_for_node_output(node_id, 0, dt, shape)
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
            Some(offsets) if slot < offsets.len() && offsets[slot].len() == 1 => Ok(DeviceView {
                buffer: Arc::clone(&self.storage),
                dt,
                len: shape.iter().product(),
                shape: shape.into(),
                strides: Tensor::natural_strides(shape),
                offset_bytes: offsets[slot][0],
                exotic_fact: None,
            }
            .into()),
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
                Ok(DeviceView {
                    buffer: Arc::clone(&self.storage),
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
