use crate::tensor::MetalArenaView;
use anyhow::{anyhow, Context};
use core::sync::atomic::AtomicUsize;
use metal::Buffer;
use metal::Device;
use metal::MTLResourceOptions;
use std::sync::atomic::Ordering;
use tract_core::internal::*;

#[derive(Debug)]
pub struct MetalArena {
    storage: Arc<MetalArenaStorage>,
    cursor: AtomicUsize,
    capacity: usize,
}

impl MetalArena {
    pub fn with_capacity(device: &Device, capacity: usize) -> TractResult<Self> {
        let tensor = unsafe {
            Tensor::uninitialized_dt(DatumType::U8, &[capacity]).with_context(|| {
                anyhow!("Error while allocating a tensor of {:?} bytes", capacity)
            })?
        };
        let buffer = device.new_buffer_with_bytes_no_copy(
            tensor.as_bytes().as_ptr() as *const core::ffi::c_void,
            capacity as _,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        Ok(Self {
            storage: Arc::new(MetalArenaStorage { tensor, metal: buffer }),
            cursor: AtomicUsize::new(0),
            capacity,
        })
    }

    pub fn view_uninitialized_dt(&self, dt: DatumType, shape: &[usize]) -> Option<MetalArenaView> {
        // Check if we can reset the cursor of the arena for next
        // view.
        self.try_reset();

        let alignment = dt.alignment();
        let size = dt.size_of() * shape.into_iter().product::<usize>();

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
