use crate::MetalContext;
use metal::Buffer;
use metal::MTLResourceOptions;
use num_traits::AsPrimitive;
use std::fmt::Display;
use tract_core::internal::*;

#[derive(Debug, Clone)]
pub struct MetalArenaStorage {
    tensor: Tensor,
    metal: Buffer,
}

impl MetalArenaStorage {
    pub fn with_capacity(
        context: &MetalContext,
        capacity: usize,
        alignment: usize,
    ) -> TractResult<Self> {
        let tensor = unsafe {
            Tensor::uninitialized_aligned_dt(DatumType::U8, &[capacity], alignment).with_context(
                || anyhow!("Error while allocating a tensor of {:?} bytes", capacity),
            )?
        };
        let buffer = context.device().new_buffer_with_bytes_no_copy(
            tensor.as_bytes().as_ptr() as *const core::ffi::c_void,
            capacity as _,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        Ok(MetalArenaStorage { tensor, metal: buffer })
    }
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

#[derive(Debug, Clone, Hash)]
pub struct MetalArenaView {
    pub(crate) arena: Arc<MetalArenaStorage>,
    pub(crate) dt: DatumType,
    pub(crate) len: usize,
    pub(crate) shape: TVec<usize>,
    pub(crate) strides: TVec<isize>,
    pub(crate) offset_bytes: usize,
}

impl MetalArenaView {
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    /// Get the datum type of the tensor.
    #[inline]
    pub fn datum_type(&self) -> DatumType {
        self.dt
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.strides.as_slice()
    }

    /// Get underlying inner metal buffer.
    pub fn metal(&self) -> &Buffer {
        self.arena.metal()
    }

    /// Get underlying inner metal buffer offset
    pub fn metal_offset<I: Copy + 'static>(&self) -> I
    where
        usize: AsPrimitive<I>,
    {
        self.offset_bytes.as_()
    }

    /// Get the number of values in the tensor.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.arena.tensor().as_bytes()
            [self.offset_bytes..self.offset_bytes + self.len() * self.dt.size_of()]
    }

    #[inline]
    pub fn view(&self) -> TensorView<'_> {
        unsafe {
            TensorView::from_bytes(
                self.arena.tensor(),
                self.offset_bytes as _,
                self.shape.as_slice(),
                self.strides.as_slice(),
            )
        }
    }

    /// Reshaped tensor with given shape.
    pub fn reshaped(&self, shape: impl Into<TVec<usize>>) -> TractResult<Self> {
        let shape = shape.into();
        if self.len() != shape.iter().product::<usize>() {
            bail!("Invalid reshape {:?} to {:?}", self.shape(), shape);
        }
        if shape.as_slice() != self.shape() {
            Ok(Self {
                arena: Arc::clone(&self.arena),
                dt: self.dt,
                len: self.len,
                strides: Tensor::natural_strides(&shape),
                shape,
                offset_bytes: self.offset_bytes,
            })
        } else {
            Ok(self.clone())
        }
    }
}

impl Display for MetalArenaView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let content =
            self.clone().into_tensor().dump(false).unwrap_or_else(|e| format!("Error : {e:?}"));
        write!(f, "MetalArenaView: {{ {content} }}")
    }
}

impl IntoTensor for MetalArenaView {
    fn into_tensor(self) -> Tensor {
        unsafe {
            Tensor::from_raw_dt(self.dt, &self.shape, self.as_bytes())
                .expect("Could not transform a MetalArenaView to tensor")
        }
    }
}
