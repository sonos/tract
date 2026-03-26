use std::fmt;
use std::sync::Arc;

use tract_data::internal::*;

use super::BlockQuant;
use super::BlockQuantFact;

/// Concrete tensor storage for block-quantized weights.
///
/// Stores a single contiguous `Arc<Blob>` of quantized data along with the
/// block-quant format. Shape lives on the tensor, not here.
pub struct BlockQuantStorage {
    format: Box<dyn BlockQuant>,
    data: Arc<Blob>,
}

impl BlockQuantStorage {
    fn expected_bytes(format: &dyn BlockQuant, m: usize, k: usize) -> usize {
        m * k / format.block_len() * format.block_bytes()
    }

    pub fn new(
        format: Box<dyn BlockQuant>,
        m: usize,
        k: usize,
        data: Arc<Blob>,
    ) -> TractResult<Self> {
        let expected = Self::expected_bytes(&*format, m, k);
        ensure!(
            data.len() == expected,
            "BlockQuantStorage::new: blob length {} does not match expected {} (m={}, k={}, format={})",
            data.len(),
            expected,
            m,
            k,
            format,
        );
        Ok(Self { format, data })
    }

    pub fn format(&self) -> &dyn BlockQuant {
        &*self.format
    }

    /// Returns the single contiguous blob.
    pub fn value(&self) -> &Arc<Blob> {
        &self.data
    }

    /// Converts this storage into a `Tensor` with the given shape.
    ///
    /// `dt` is the logical element type (e.g. f32, f16) — the type these
    /// weights represent when dequantized.
    pub fn into_tensor_with_shape(self, dt: DatumType, shape: &[usize]) -> Tensor {
        Tensor::from_storage(dt, shape, self)
    }
}

/// Returns a byte slice for a single group within contiguous block-quant data.
pub fn block_quant_slice<'a>(
    data: &'a [u8],
    format: &dyn BlockQuant,
    m_per_group: usize,
    k: usize,
    g: usize,
) -> &'a [u8] {
    let row_bytes = k / format.block_len() * format.block_bytes();
    let group_bytes = m_per_group * row_bytes;
    let start = g * group_bytes;
    &data[start..start + group_bytes]
}

impl Clone for BlockQuantStorage {
    fn clone(&self) -> Self {
        Self { format: self.format.clone(), data: self.data.clone() }
    }
}

impl fmt::Debug for BlockQuantStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlockQuantStorage({}, bytes={})", self.format, self.data.len())
    }
}

impl fmt::Display for BlockQuantStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlockQuantStorage({}, bytes={})", self.format, self.data.len())
    }
}

impl TensorStorage for BlockQuantStorage {
    fn byte_len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn deep_clone(&self) -> Box<dyn TensorStorage> {
        Box::new(self.clone())
    }

    fn as_plain(&self) -> Option<&PlainStorage> {
        None
    }

    fn as_plain_mut(&mut self) -> Option<&mut PlainStorage> {
        None
    }

    fn into_plain(self: Box<Self>) -> Option<PlainStorage> {
        None
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write_u8(1);
        self.format.dyn_hash(state);
        state.write(self.data.as_bytes());
    }

    fn same_as(&self, other: &dyn TensorStorage) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            self.format.same_as(&*other.format) && self.data.as_bytes() == other.data.as_bytes()
        } else {
            false
        }
    }

    fn exotic_fact(&self, shape: &[usize]) -> TractResult<Option<Box<dyn ExoticFact>>> {
        Ok(Some(Box::new(BlockQuantFact::new(dyn_clone::clone_box(&*self.format), shape.into()))))
    }
}
