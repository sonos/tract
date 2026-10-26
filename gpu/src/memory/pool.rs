use crate::device::get_context;
use crate::memory::DeviceResolvedMemSchema;
use crate::memory::DeviceResolvedPartitions;
use crate::tensor::DeviceTensor;
use crate::tensor::DeviceView;
use crate::tensor::OwnedDeviceTensor;

use tract_core::internal::*;

#[derive(Debug)]
pub struct DeviceMemoryPool {
    layout: Layout,
}

#[derive(Debug)]
enum Layout {
    /// One buffer, each node output at its offset.
    Arena { storage: Arc<Box<dyn OwnedDeviceTensor>>, resolved_schema: DeviceResolvedMemSchema },
    /// A buffer per partition, each node output at the start of its partition's.
    Partitions { buffers: Vec<Arc<Box<dyn OwnedDeviceTensor>>>, resolved: DeviceResolvedPartitions },
}

/// Reusable arena storage held in a turn's shared resources.
///
/// Storage is reused only when the cache is its sole owner. Live arena views
/// retain their backing allocation and force a new allocation on the next turn.
#[derive(Debug, Default)]
pub struct ArenaStorageCache {
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
        Ok(Self { layout: Layout::Arena { storage, resolved_schema } })
    }

    pub fn from_schema(resolved_schema: DeviceResolvedMemSchema) -> TractResult<Self> {
        let storage = Arc::new(
            get_context()?
                .uninitialized_device_tensor(&[resolved_schema.memory_size], DatumType::U8)?,
        );
        Ok(Self { layout: Layout::Arena { storage, resolved_schema } })
    }

    /// A pool over `buffers`, one per partition of `resolved` and at least as large.
    pub fn from_partitions(
        resolved: DeviceResolvedPartitions,
        buffers: Vec<Arc<Box<dyn OwnedDeviceTensor>>>,
    ) -> TractResult<Self> {
        ensure!(buffers.len() == resolved.sizes.len());
        for (buffer, size) in buffers.iter().zip(&resolved.sizes) {
            ensure!(buffer.len() * buffer.datum_type().size_of() >= *size);
        }
        Ok(Self { layout: Layout::Partitions { buffers, resolved } })
    }

    /// For each output slot of `node_id`, how many buffers the schema planned for it.
    fn planned(&self, node_id: usize) -> Option<TVec<usize>> {
        let by_node = match &self.layout {
            Layout::Arena { resolved_schema, .. } => &resolved_schema.offsets_by_node,
            Layout::Partitions { resolved, .. } => &resolved.partitions_by_node,
        };
        by_node[node_id].as_ref().map(|slots| slots.iter().map(|it| it.len()).collect())
    }

    /// The storage and byte offset of buffer `ix` of output `slot` of `node_id`.
    fn region(
        &self,
        node_id: usize,
        slot: usize,
        ix: usize,
    ) -> Option<(&Arc<Box<dyn OwnedDeviceTensor>>, usize)> {
        match &self.layout {
            Layout::Arena { storage, resolved_schema } => {
                let offsets = resolved_schema.offsets_by_node[node_id].as_ref()?.get(slot)?;
                Some((storage, *offsets.get(ix)?))
            }
            Layout::Partitions { buffers, resolved } => {
                let partitions = resolved.partitions_by_node[node_id].as_ref()?.get(slot)?;
                Some((&buffers[*partitions.get(ix)?], 0))
            }
        }
    }

    pub fn tensor_for_node(
        &self,
        node_id: usize,
        dt: DatumType,
        shape: &[usize],
    ) -> TractResult<DeviceTensor> {
        if let Some(slots) = self.planned(node_id) {
            ensure!(slots.len() == 1, "'tensor_for_node' is for mono-output nodes only");
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
        match self.planned(node_id).and_then(|slots| slots.get(slot).copied()) {
            Some(1) => {
                let (storage, offset_bytes) = self.region(node_id, slot, 0).unwrap();
                Ok(DeviceView {
                    buffer: Arc::clone(storage),
                    dt,
                    len: shape.iter().product(),
                    shape: shape.into(),
                    strides: Tensor::natural_strides(shape),
                    offset_bytes,
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
        match self.planned(node_id) {
            Some(slots) => {
                ensure!(
                    slots.len() == 1 && slots[0] == 2,
                    "'scalar_exotic_tensor_for_node' is for mono-output nodes only"
                );
                let (storage, offset_bytes) = self.region(node_id, 0, 1).unwrap();
                Ok(DeviceView {
                    buffer: Arc::clone(storage),
                    dt,
                    len: 1,
                    shape: tvec!(),
                    strides: tvec!(),
                    offset_bytes,
                    exotic_fact: Some(exotic_fact.clone()),
                }
                .into())
            }
            None => DeviceTensor::uninitialized_exotic(exotic_fact),
        }
    }
}
