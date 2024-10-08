use crate::MetalContext;
use crate::tensor::MetalArenaView;
use anyhow::{anyhow, Context};
use core::sync::atomic::AtomicUsize;
use metal::{ MTLResourceOptions, Buffer };
use std::sync::atomic::Ordering;
use tract_core::internal::*;

#[derive(Debug)]
pub struct MetalArena {
    storage: Arc<MetalArenaStorage>,
    cursor: AtomicUsize,
    capacity: usize,
    alignment: usize,
}

impl MetalArena {
    pub fn with_capacity(context: &MetalContext, capacity: usize) -> TractResult<Self> {
        let alignment = std::mem::size_of::<usize>();
        let tensor = unsafe {
            Tensor::uninitialized_aligned_dt(DatumType::U8, &[capacity], alignment).with_context(|| {
                anyhow!("Error while allocating a tensor of {:?} bytes", capacity)
            })?
        };
        let buffer = context.device().new_buffer_with_bytes_no_copy(
            tensor.as_bytes().as_ptr() as *const core::ffi::c_void,
            capacity as _,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        Ok(Self {
            storage: Arc::new(MetalArenaStorage { tensor, metal: buffer }),
            cursor: AtomicUsize::new(0),
            capacity,
            alignment,
        })
    }

    pub fn free_capacity(&self) -> usize {
        let cursor = self.cursor.load(Ordering::SeqCst);
        self.capacity - cursor
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn used_capacity(&self) -> usize {
        self.cursor.load(Ordering::SeqCst)
    }

    pub fn view_uninitialized_dt(&self, dt: DatumType, shape: &[usize]) -> Option<MetalArenaView> {
        // Check if we can reset the cursor of the arena for next
        // view.
        self.try_reset();

        let alignment = dt.alignment();
        if self.alignment % alignment != 0 {
            return None;
        }
        let size = dt.size_of() * shape.iter().product::<usize>();

        let cursor = self.cursor.load(Ordering::SeqCst);

        let start = if cursor % alignment != 0 {
            cursor + (alignment - cursor % alignment)
        } else {
            cursor
        };

        let end = start + size;
        if self.capacity < end {
            return None;
        }

        self.cursor.store(end, Ordering::SeqCst);

        Some(MetalArenaView {
            arena: Arc::clone(&self.storage),
            dt,
            shape: shape.into(),
            strides: Tensor::natural_strides(shape),
            offset_bytes: start,
        })
    }

    pub fn try_reset(&self) {
        let cursor = self.cursor.load(Ordering::SeqCst);
        if Arc::strong_count(&self.storage) == 1 && cursor != 0 {
            self.cursor.store(0, Ordering::SeqCst)
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetalArenaStorage {
    tensor: Tensor,
    metal: Buffer,
}

impl MetalArenaStorage {
    /// Get underlying inner metal buffer.
    pub fn metal(&self) -> &Buffer {
        &self.metal
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }
}

impl Hash for MetalArenaStorage {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tensor.hash(state)
    }
}
