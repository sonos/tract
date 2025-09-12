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

impl DeviceMemoryPool {
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
        let offsets: Vec<Vec<usize>> = self
            .resolved_schema
            .offsets_by_node
            .iter()
            .filter(|(k, _)| k.node == node_id)
            .map(|(_, v)| v.clone())
            .collect();

        ensure!(dt != DatumType::Opaque, "Use opaque_tensor for node instead");
        ensure!(
            offsets.is_empty() || (offsets.len() == 1 && offsets[0].len() == 1),
            "'tensor_for_node' is for mono-output nodes only"
        );

        offsets
            .first()
            .map(|offset| {
                Ok(DeviceArenaView {
                    arena: Arc::clone(&self.storage),
                    dt,
                    len: shape.iter().product(),
                    shape: shape.into(),
                    strides: Tensor::natural_strides(shape),
                    offset_bytes: offset[0],
                    opaque_fact: None,
                }
                .into())
            })
            .unwrap_or_else(|| DeviceTensor::uninitialized_dt(dt, shape))
    }

    pub fn scalar_opaque_tensor_for_node(&self, node_id: usize) -> TractResult<DeviceTensor> {
        let offsets: Vec<Vec<usize>> = self
            .resolved_schema
            .offsets_by_node
            .iter()
            .filter(|(k, _)| k.node == node_id)
            .map(|(_, v)| v.clone())
            .collect();

        ensure!(
            offsets.is_empty() || (offsets.len() == 1 && offsets[0].len() == 2),
            "'scalar_opaque_tensor_for_node' is for mono-output nodes only"
        );

        offsets
            .first()
            .map(|offset| {
                Ok(DeviceArenaView {
                    arena: Arc::clone(&self.storage),
                    dt: DatumType::Opaque,
                    len: 1,
                    shape: tvec!(),
                    strides: tvec!(),
                    offset_bytes: offset[1],
                    opaque_fact: self.resolved_schema.opaque_facts[node_id][0].clone(),
                }
                .into())
            })
            .unwrap_or_else(|| {
                DeviceTensor::uninitialized_opaque(
                    self.resolved_schema.opaque_facts[node_id][0].clone().unwrap().as_ref(),
                )
            })
    }
}
