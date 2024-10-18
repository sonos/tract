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
        // ensure!(!self.node_seen.borrow().contains(&node_id), "Tensor for node {:?} was already requested. Maybe the memory pool was not reset properly.", node_id);
        let alignment = dt.alignment();
        (self.alignment % alignment == 0)
            .then(|| self.resolved_schema.offsets_by_node[node_id])
            .map(|offset| {
                // self.node_seen.borrow_mut().insert(node_id);
                Ok(MetalArenaView {
                    arena: Arc::clone(&self.storage),
                    dt,
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

// #[derive(Debug)]
// pub struct MetalArena {
//     storage: Arc<MetalArenaStorage>,
//     cursor: AtomicUsize,
//     capacity: usize,
//     alignment: usize,
// }

// impl MetalArena {
//     pub fn with_capacity(context: &MetalContext, capacity: usize) -> TractResult<Self> {
//         let alignment = std::mem::size_of::<usize>();
//         let tensor = unsafe {
//             Tensor::uninitialized_aligned_dt(DatumType::U8, &[capacity], alignment).with_context(
//                 || anyhow!("Error while allocating a tensor of {:?} bytes", capacity),
//             )?
//         };
//         let buffer = context.device().new_buffer_with_bytes_no_copy(
//             tensor.as_bytes().as_ptr() as *const core::ffi::c_void,
//             capacity as _,
//             MTLResourceOptions::StorageModeShared,
//             None,
//         );
//         Ok(Self {
//             storage: Arc::new(MetalArenaStorage { tensor, metal: buffer }),
//             cursor: AtomicUsize::new(0),
//             capacity,
//             alignment,
//         })
//     }

//     pub fn free_capacity(&self) -> usize {
//         let cursor = self.cursor.load(Ordering::SeqCst);
//         self.capacity - cursor
//     }

//     pub fn capacity(&self) -> usize {
//         self.capacity
//     }

//     pub fn used_capacity(&self) -> usize {
//         self.cursor.load(Ordering::SeqCst)
//     }

//     pub fn view_uninitialized_dt(&self, dt: DatumType, shape: &[usize]) -> Option<MetalArenaView> {
//         // Check if we can reset the cursor of the arena for next
//         // view.
//         self.try_reset();

//         let alignment = dt.alignment();
//         if self.alignment % alignment != 0 {
//             return None;
//         }
//         let size = dt.size_of() * shape.iter().product::<usize>();

//         let cursor = self.cursor.load(Ordering::SeqCst);

//         let start = if cursor % alignment != 0 {
//             cursor + (alignment - cursor % alignment)
//         } else {
//             cursor
//         };

//         let end = start + size;
//         if self.capacity < end {
//             return None;
//         }

//         self.cursor.store(end, Ordering::SeqCst);

//         Some(MetalArenaView {
//             arena: Arc::clone(&self.storage),
//             dt,
//             shape: shape.into(),
//             strides: Tensor::natural_strides(shape),
//             offset_bytes: start,
//         })
//     }
// }
