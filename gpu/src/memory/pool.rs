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
        ensure!(dt != DatumType::Opaque, "Use opaque_tensor for node instead");
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
                    opaque_fact: None,
                }
                .into())
            })
            .unwrap_or_else(|| DeviceTensor::uninitialized_dt(dt, shape))
    }

    pub fn scalar_opaque_tensor_for_node(
        &self,
        node_id: usize,
        opaque_fact: Box<dyn OpaqueFact>,
    ) -> TractResult<DeviceTensor> {
        match self.resolved_schema.offsets_by_node[node_id].as_ref() {
            Some(offsets) => {
                ensure!(
                    offsets.len() == 1 && offsets[0].len() == 2,
                    "'scalar_opaque_tensor_for_node' is for mono-output nodes only"
                );
                Ok(DeviceArenaView {
                    arena: Arc::clone(&self.storage),
                    dt: DatumType::Opaque,
                    len: 1,
                    shape: tvec!(),
                    strides: tvec!(),
                    offset_bytes: offsets[0][1],
                    opaque_fact: Some(opaque_fact.clone()),
                }
                .into())
            }
            None => DeviceTensor::uninitialized_opaque(opaque_fact),
        }
    }
}
