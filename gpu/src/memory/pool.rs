use crate::device::GpuDevice;
use crate::memory::DeviceResolvedMemSchema;
use crate::tensor::DeviceTensor;
use crate::tensor::IntoGpu;
use crate::tensor::{DeviceArenaStorage, DeviceArenaView};
use anyhow::Result;
use std::cell::RefCell;
use std::collections::HashSet;
use tract_core::internal::*;

#[derive(Debug)]
pub struct DeviceMemoryPool {
    storage: Arc<DeviceArenaStorage>,
    resolved_schema: DeviceResolvedMemSchema,
    node_seen: RefCell<HashSet<usize>>,
}

impl DeviceMemoryPool {
    pub fn from_schema(
        context: Box<dyn GpuDevice>,
        resolved_schema: DeviceResolvedMemSchema,
    ) -> Result<Self> {
        let storage =
            Arc::new(DeviceArenaStorage::with_capacity(context, resolved_schema.memory_size)?);

        Ok(Self { storage, resolved_schema, node_seen: RefCell::new(HashSet::new()) })
    }

    pub fn tensor_for_node(
        &self,
        node_id: usize,
        dt: DatumType,
        shape: &[usize],
    ) -> Result<DeviceTensor> {
        ensure!(!self.node_seen.borrow().contains(&node_id), "Tensor for node {:?} was already requested. Maybe the memory pool was not reset properly.", node_id);
        self.resolved_schema.offsets_by_node[node_id]
            .map(|offset| {
                // self.node_seen.borrow_mut().insert(node_id);
                Ok(DeviceArenaView {
                    arena: Arc::clone(&self.storage),
                    dt,
                    len: shape.iter().product(),
                    shape: shape.into(),
                    strides: Tensor::natural_strides(shape),
                    offset_bytes: offset,
                }
                .into())
            })
            .unwrap_or_else(|| unsafe { Tensor::uninitialized_dt(dt, shape)?.into_gpu() })
    }

    pub fn reset(&self) {
        self.node_seen.borrow_mut().clear();
    }
}
