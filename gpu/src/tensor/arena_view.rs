use num_traits::AsPrimitive;
use std::ffi::c_void;
use std::fmt::Display;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuantFact, BlockQuantStorage};

use crate::device::{DeviceBuffer, get_context};
use crate::utils::check_strides_validity;

use super::OwnedDeviceTensor;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DeviceArenaView {
    pub(crate) arena: Arc<Box<dyn OwnedDeviceTensor>>,
    pub(crate) dt: DatumType,
    pub(crate) len: usize,
    pub(crate) shape: TVec<usize>,
    pub(crate) strides: TVec<isize>,
    pub(crate) offset_bytes: usize,
    pub(crate) exotic_fact: Option<Box<dyn ExoticFact>>,
}

impl DeviceArenaView {
    /// Build a view over any owned device tensor. `shape`/`strides` are in
    /// elements of `dt`; `offset_bytes` from the buffer start. The backing
    /// tensor stays alive as long as any view of it does.
    pub fn from_owned(
        arena: Arc<Box<dyn OwnedDeviceTensor>>,
        dt: DatumType,
        shape: TVec<usize>,
        strides: TVec<isize>,
        offset_bytes: usize,
    ) -> TractResult<Self> {
        // Unlike arena slots, these views may be non-dense (e.g. the valid
        // region of a capacity buffer); validate bounds, not density.
        ensure!(shape.len() == strides.len());
        ensure!(strides.iter().all(|&s| s >= 0), "negative strides unsupported");
        let max_index: usize = shape
            .iter()
            .zip(strides.iter())
            .map(|(&d, &s)| d.saturating_sub(1) * s as usize)
            .sum();
        let needed = offset_bytes + (max_index + 1) * dt.size_of();
        let arena_bytes = arena.len() * arena.datum_type().size_of();
        ensure!(
            shape.iter().product::<usize>() == 0 || needed <= arena_bytes,
            "view out of bounds: needs {needed} bytes, arena has {arena_bytes}"
        );
        let len = shape.iter().product();
        Ok(DeviceArenaView { arena, dt, len, shape, strides, offset_bytes, exotic_fact: None })
    }

    /// Metadata-only slice keeping `[start, end)` along `axis`: same arena,
    /// same strides, adjusted shape and byte offset. No bytes move, and the
    /// backing buffer stays alive through the new view's Arc, so other views
    /// of the arena (e.g. longer KV-cache snapshots) remain valid.
    pub fn sliced(&self, axis: usize, start: usize, end: usize) -> TractResult<Self> {
        ensure!(self.exotic_fact.is_none(), "cannot slice a view with an exotic fact");
        ensure!(axis < self.shape.len(), "axis {axis} out of rank {}", self.shape.len());
        ensure!(
            start <= end && end <= self.shape[axis],
            "invalid slice [{start}, {end}) on axis {axis} of len {}",
            self.shape[axis]
        );
        let mut shape = self.shape.clone();
        shape[axis] = end - start;
        let offset_bytes =
            self.offset_bytes + start * self.strides[axis] as usize * self.dt.size_of();
        Self::from_owned(self.arena.clone(), self.dt, shape, self.strides.clone(), offset_bytes)
    }

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

    /// Get underlying inner device buffer.
    pub fn device_buffer(&self) -> &dyn DeviceBuffer {
        self.arena.device_buffer()
    }

    pub fn device_buffer_ptr(&self) -> *const c_void {
        self.arena.device_buffer().ptr()
    }

    /// Get underlying inner device buffer offset
    pub fn buffer_offset<I: Copy + 'static>(&self) -> I
    where
        usize: AsPrimitive<I>,
    {
        self.offset_bytes.as_()
    }

    pub fn exotic_fact(&self) -> Option<&dyn ExoticFact> {
        self.exotic_fact.as_deref()
    }

    /// Get the number of values in the tensor.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_bytes(&self) -> Vec<u8> {
        let len = if let Some(of) = &self.exotic_fact {
            of.mem_size().as_i64().unwrap() as usize
        } else {
            self.len() * self.dt.size_of()
        };
        if self.is_dense() {
            return self.arena.get_bytes_slice(self.offset_bytes, len);
        }
        // Non-dense view: gather row by row (contiguous rows in one slice,
        // element-wise when the last axis is strided too, e.g. transposed
        // KV-cache layouts).
        let esize = self.dt.size_of();
        let rank = self.shape.len();
        let row = self.shape[rank - 1];
        let last_stride = self.strides[rank - 1] as usize;
        let outer: usize = self.shape[..rank - 1].iter().product();
        let mut out = Vec::with_capacity(len);
        for r in 0..outer {
            let mut rem = r;
            let mut offset = self.offset_bytes;
            for ax in (0..rank - 1).rev() {
                let ix = rem % self.shape[ax];
                rem /= self.shape[ax];
                offset += ix * self.strides[ax] as usize * esize;
            }
            if last_stride == 1 {
                out.extend_from_slice(&self.arena.get_bytes_slice(offset, row * esize));
            } else {
                for i in 0..row {
                    out.extend_from_slice(
                        &self.arena.get_bytes_slice(offset + i * last_stride * esize, esize),
                    );
                }
            }
        }
        out
    }

    /// True when the view's strides are the natural (packed) strides.
    pub fn is_dense(&self) -> bool {
        let mut expect = 1isize;
        for (d, s) in self.shape.iter().zip(self.strides.iter()).rev() {
            if *d != 1 && *s != expect {
                return false;
            }
            expect *= *d as isize;
        }
        true
    }

    /// Reshaped tensor with given shape.
    pub fn reshaped(&self, shape: impl Into<TVec<usize>>) -> TractResult<Self> {
        ensure!(self.exotic_fact.is_none(), "Can't reshape exotic tensor");
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
                exotic_fact: None,
            })
        } else {
            Ok(self.clone())
        }
    }

    pub fn restrided(&self, strides: impl Into<TVec<isize>>) -> TractResult<Self> {
        ensure!(self.exotic_fact.is_none(), "Can't restride exotic tensor");
        let strides = strides.into();
        check_strides_validity(self.shape().into(), strides.clone())?;

        if strides.as_slice() != self.strides() {
            Ok(Self {
                arena: Arc::clone(&self.arena),
                dt: self.dt,
                len: self.len,
                strides,
                shape: self.shape.clone(),
                offset_bytes: self.offset_bytes,
                exotic_fact: None,
            })
        } else {
            Ok(self.clone())
        }
    }

    pub fn to_host(&self) -> TractResult<Tensor> {
        get_context()?.synchronize()?;
        let content = self.as_bytes();
        unsafe {
            if let Some(bqf) =
                self.exotic_fact.as_ref().and_then(|of| of.downcast_ref::<BlockQuantFact>())
            {
                Ok(BlockQuantStorage::new(
                    bqf.format.clone(),
                    bqf.m(),
                    bqf.k(),
                    Arc::new(Blob::from_bytes(&content)?),
                )?
                .into_tensor_with_shape(self.dt, bqf.shape()))
            } else {
                Tensor::from_raw_dt(self.dt, &self.shape, &content)
            }
        }
    }
}

impl Display for DeviceArenaView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let content = self
            .clone()
            .to_host()
            .unwrap()
            .dump(false)
            .unwrap_or_else(|e| format!("Error : {e:?}"));
        write!(f, "DeviceArenaView: {{ {content} }}")
    }
}
