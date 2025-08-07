use crate::device::get_context;
use crate::memory::DeviceResolvedMemSchema;
use crate::tensor::DeviceArenaView;
use crate::tensor::DeviceTensor;
use crate::tensor::OwnedDeviceTensor;

use std::cell::RefCell;
use std::collections::HashSet;
use tract_core::internal::*;

#[derive(Debug)]
pub struct DeviceMemoryPool {
    storage: Arc<Box<dyn OwnedDeviceTensor>>,
    resolved_schema: DeviceResolvedMemSchema,
    node_seen: RefCell<HashSet<usize>>,
}

impl DeviceMemoryPool {
    pub fn from_schema(resolved_schema: DeviceResolvedMemSchema) -> TractResult<Self> {
        Ok(Self {
            storage: Arc::new(
                get_context()?
                    .uninitialized_device_tensor(&[resolved_schema.memory_size], DatumType::U8)?,
            ),
            resolved_schema,
            node_seen: RefCell::new(HashSet::new()),
        })
    }

    pub fn tensor_for_node(
        &self,
        node_id: usize,
        dt: DatumType,
        shape: &[usize],
    ) -> TractResult<DeviceTensor> {
        ensure!(
            !self.node_seen.borrow().contains(&node_id),
            "Tensor for node {:?} was already requested. Maybe the memory pool was not reset properly.",
            node_id
        );
        self.resolved_schema.offsets_by_node[node_id]
            .map(|offset| {
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
            .unwrap_or_else(|| DeviceTensor::uninitialized_dt(dt, shape))
    }

    pub fn reset(&self) {
        self.node_seen.borrow_mut().clear();
    }
}
