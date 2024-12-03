use crate::memory::MetalResolvedMemSchema;
use crate::tensor::{MetalArenaStorage, MetalArenaView};
use crate::{IntoMetal, MetalContext, MetalTensor};
use anyhow::Result;
use std::cell::RefCell;
use std::collections::HashSet;
use tract_core::internal::*;

#[derive(Debug)]
pub struct MetalMemoryPool {
    storage: Arc<MetalArenaStorage>,
    alignment: usize,
    resolved_schema: MetalResolvedMemSchema,
    node_seen: RefCell<HashSet<usize>>,
}

impl MetalMemoryPool {
    pub fn from_schema(
        context: &MetalContext,
        resolved_schema: MetalResolvedMemSchema,
    ) -> Result<Self> {
        let alignment = std::mem::size_of::<usize>();
        let storage = Arc::new(MetalArenaStorage::with_capacity(
            context,
            resolved_schema.memory_size,
            alignment,
        )?);

        Ok(Self { storage, alignment, resolved_schema, node_seen: RefCell::new(HashSet::new()) })
    }

    pub fn tensor_for_node(
        &self,
        node_id: usize,
        dt: DatumType,
        shape: &[usize],
    ) -> Result<MetalTensor> {
        ensure!(!self.node_seen.borrow().contains(&node_id), "Tensor for node {:?} was already requested. Maybe the memory pool was not reset properly.", node_id);
        let alignment = dt.alignment();
        (self.alignment % alignment == 0)
            .then(|| self.resolved_schema.offsets_by_node[node_id])
            .flatten()
            .map(|offset| {
                // self.node_seen.borrow_mut().insert(node_id);
                Ok(MetalArenaView {
                    arena: Arc::clone(&self.storage),
                    dt,
                    len: shape.iter().product(),
                    shape: shape.into(),
                    strides: Tensor::natural_strides(shape),
                    offset_bytes: offset,
                }
                .into())
            })
            .unwrap_or_else(|| unsafe { Tensor::uninitialized_dt(dt, shape)?.into_metal() })
    }

    pub fn reset(&self) {
        self.node_seen.borrow_mut().clear();
    }
}
