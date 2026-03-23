use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use tract_data::internal::*;

use super::{BlockQuant, BlockQuantFact};

/// Concrete tensor storage for block-quantized weights.
///
/// Stores a single contiguous `Arc<Blob>` of quantized data along with the
/// block-quant format and logical m×k dimensions.  The G (group) dimension
/// is purely a tensor-shape concern — storage knows nothing about groups.
pub struct BlockQuantStorage {
    format: Box<dyn BlockQuant>,
    m: usize,
    k: usize,
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
        Ok(Self { format, m, k, data })
    }

    pub fn format(&self) -> &dyn BlockQuant {
        &*self.format
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the single contiguous blob.
    pub fn value(&self) -> &Arc<Blob> {
        &self.data
    }

    /// Converts this storage into a rank-3 `Tensor` with shape `[1, M, K]`.
    pub fn into_tensor(self) -> Tensor {
        Tensor::from_storage(DatumType::Opaque, &[1, self.m, self.k], self)
    }

    /// Converts this storage into a `Tensor` with the given shape.
    ///
    /// The shape's product of dimensions must be consistent with the stored data.
    pub fn into_tensor_with_shape(self, shape: &[usize]) -> Tensor {
        Tensor::from_storage(DatumType::Opaque, shape, self)
    }

    /// Reconstructs a `BlockQuantFact` from this storage's metadata.
    pub fn to_block_quant_fact(&self) -> BlockQuantFact {
        BlockQuantFact::new(self.format.clone(), tvec!(1, self.m, self.k))
    }

    /// Returns a clone with updated m and k dimensions, preserving format and data blob.
    pub fn with_shape(&self, m: usize, k: usize) -> TractResult<Self> {
        let expected = Self::expected_bytes(&*self.format, m, k);
        ensure!(
            self.data.len() == expected,
            "BlockQuantStorage::with_shape: blob length {} does not match expected {} (m={}, k={}, format={})",
            self.data.len(),
            expected,
            m,
            k,
            self.format,
        );
        Ok(Self { format: self.format.clone(), m, k, data: self.data.clone() })
    }

    /// Returns a byte slice for a single group within the contiguous data.
    ///
    /// The caller provides the group index and total number of groups.
    /// `self.m` must be the total M across all groups (i.e. `num_groups * m_per_group`).
    pub fn group_slice(&self, g: usize, num_groups: usize) -> &[u8] {
        let rows_per_group = self.m / num_groups;
        let row_bytes = self.k / self.format.block_len() * self.format.block_bytes();
        let group_bytes = rows_per_group * row_bytes;
        let start = g * group_bytes;
        &self.data[start..start + group_bytes]
    }
}

impl Clone for BlockQuantStorage {
    fn clone(&self) -> Self {
        Self { format: self.format.clone(), m: self.m, k: self.k, data: self.data.clone() }
    }
}

impl fmt::Debug for BlockQuantStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BlockQuantStorage({}, m={}, k={}, bytes={})",
            self.format,
            self.m,
            self.k,
            self.data.len()
        )
    }
}

impl fmt::Display for BlockQuantStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BlockQuantStorage({}, m={}, k={}, bytes={})",
            self.format,
            self.m,
            self.k,
            self.data.len()
        )
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

    fn as_dense(&self) -> Option<&DenseStorage> {
        None
    }

    fn as_dense_mut(&mut self) -> Option<&mut DenseStorage> {
        None
    }

    fn into_dense(self: Box<Self>) -> Option<DenseStorage> {
        None
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write_u8(1);
        self.format.dyn_hash(state);
        self.m.hash(&mut HashWrapper(state));
        self.k.hash(&mut HashWrapper(state));
        state.write(self.data.as_bytes());
    }

    fn same_as(&self, other: &dyn TensorStorage) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            self.format.same_as(&*other.format)
                && self.m == other.m
                && self.k == other.k
                && self.data.as_bytes() == other.data.as_bytes()
        } else {
            false
        }
    }
}

/// Adapter to bridge `&mut dyn Hasher` into `impl Hasher` for the `Hash` trait.
struct HashWrapper<'a>(&'a mut dyn std::hash::Hasher);

impl std::hash::Hasher for HashWrapper<'_> {
    fn finish(&self) -> u64 {
        self.0.finish()
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
}
