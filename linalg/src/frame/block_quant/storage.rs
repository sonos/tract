use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use tract_data::internal::*;

use super::{BlockQuant, BlockQuantFact};

/// Concrete tensor storage for block-quantized weights.
///
/// Stores one or more groups of quantized data as `Arc<Blob>` along with the
/// block-quant format and logical m×k dimensions.  Multi-group tensors arise
/// from `SplitGroupBlockQuant` which partitions the m dimension.
pub struct BlockQuantStorage {
    format: Box<dyn BlockQuant>,
    m: usize,
    k: usize,
    groups: Vec<Arc<Blob>>,
}

impl BlockQuantStorage {
    fn expected_group_bytes(format: &dyn BlockQuant, m: usize, k: usize) -> usize {
        m * k / format.block_len() * format.block_bytes()
    }

    pub fn new(
        format: Box<dyn BlockQuant>,
        m: usize,
        k: usize,
        value: Arc<Blob>,
    ) -> TractResult<Self> {
        let expected = Self::expected_group_bytes(&*format, m, k);
        ensure!(
            value.len() == expected,
            "BlockQuantStorage::new: blob length {} does not match expected {} (m={}, k={}, format={})",
            value.len(),
            expected,
            m,
            k,
            format,
        );
        Ok(Self { format, m, k, groups: vec![value] })
    }

    pub fn new_multi_group(
        format: Box<dyn BlockQuant>,
        m: usize,
        k: usize,
        groups: Vec<Arc<Blob>>,
    ) -> TractResult<Self> {
        let expected = Self::expected_group_bytes(&*format, m, k);
        for (i, g) in groups.iter().enumerate() {
            ensure!(
                g.len() == expected,
                "BlockQuantStorage::new_multi_group: group {} blob length {} does not match expected {} (m={}, k={}, format={})",
                i,
                g.len(),
                expected,
                m,
                k,
                format,
            );
        }
        Ok(Self { format, m, k, groups })
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

    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    pub fn group_blob(&self, i: usize) -> &Arc<Blob> {
        &self.groups[i]
    }

    pub fn groups(&self) -> &[Arc<Blob>] {
        &self.groups
    }

    /// Returns the single blob for a non-multi-group storage.
    ///
    /// # Panics
    /// Panics if this storage has more than one group.
    pub fn value(&self) -> &Arc<Blob> {
        assert_eq!(self.groups.len(), 1, "value() called on multi-group BlockQuantStorage");
        &self.groups[0]
    }

    /// Converts this storage into a rank-0 `Tensor` with `DatumType::Opaque`.
    pub fn into_tensor(self) -> Tensor {
        Tensor::from_storage(DatumType::Opaque, &[], self)
    }

    /// Reconstructs a `BlockQuantFact` from this storage's metadata.
    pub fn to_block_quant_fact(&self) -> BlockQuantFact {
        BlockQuantFact::new(self.format.clone(), tvec!(self.m * self.num_groups(), self.k))
    }

    /// Returns a clone with updated m and k dimensions, preserving format and group blobs.
    pub fn with_shape(&self, m: usize, k: usize) -> TractResult<Self> {
        let expected = Self::expected_group_bytes(&*self.format, m, k);
        for (i, g) in self.groups.iter().enumerate() {
            ensure!(
                g.len() == expected,
                "BlockQuantStorage::with_shape: group {} blob length {} does not match expected {} (m={}, k={}, format={})",
                i,
                g.len(),
                expected,
                m,
                k,
                self.format,
            );
        }
        Ok(Self { format: self.format.clone(), m, k, groups: self.groups.clone() })
    }

    /// Splits a single-group storage into `num_groups` by partitioning the m dimension.
    ///
    /// Each group gets `m / num_groups` rows. Panics if not single-group or m not divisible.
    pub fn split_m(&self, num_groups: usize) -> TractResult<Self> {
        assert_eq!(self.groups.len(), 1, "split_m requires single-group storage");
        ensure!(
            self.m % num_groups == 0,
            "m={} not divisible by num_groups={}",
            self.m,
            num_groups
        );
        let rows_per_group = self.m / num_groups;
        let row_bytes = self.k / self.format.block_len() * self.format.block_bytes();
        let group_bytes = rows_per_group * row_bytes;
        let blob = &self.groups[0];
        let groups = (0..num_groups)
            .map(|g| {
                let start = g * group_bytes;
                let mut new_blob =
                    unsafe { Blob::new_for_size_and_align(group_bytes, vector_size()) };
                new_blob.copy_from_slice(&blob[start..start + group_bytes]);
                Arc::new(new_blob)
            })
            .collect();
        let result = Self { format: self.format.clone(), m: rows_per_group, k: self.k, groups };
        let expected = Self::expected_group_bytes(&*result.format, result.m, result.k);
        for (i, g) in result.groups.iter().enumerate() {
            ensure!(
                g.len() == expected,
                "BlockQuantStorage::split_m: group {} blob length {} does not match expected {} (m={}, k={})",
                i,
                g.len(),
                expected,
                result.m,
                result.k,
            );
        }
        Ok(result)
    }
}

impl Clone for BlockQuantStorage {
    fn clone(&self) -> Self {
        Self { format: self.format.clone(), m: self.m, k: self.k, groups: self.groups.clone() }
    }
}

impl fmt::Debug for BlockQuantStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BlockQuantStorage({}, m={}, k={}, groups={})",
            self.format,
            self.m,
            self.k,
            self.groups.len()
        )
    }
}

impl fmt::Display for BlockQuantStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BlockQuantStorage({}, m={}, k={}, groups={})",
            self.format,
            self.m,
            self.k,
            self.groups.len()
        )
    }
}

impl TensorStorage for BlockQuantStorage {
    fn byte_len(&self) -> usize {
        self.groups.iter().map(|g| g.len()).sum()
    }

    fn is_empty(&self) -> bool {
        self.groups.is_empty() || self.groups.iter().all(|g| g.is_empty())
    }

    fn deep_clone(&self) -> Box<dyn TensorStorage> {
        Box::new(self.clone())
    }

    fn same_as(&self, other: &dyn TensorStorage) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            self.format.same_as(&*other.format)
                && self.m == other.m
                && self.k == other.k
                && self.groups.len() == other.groups.len()
                && self.groups.iter().zip(&other.groups).all(|(a, b)| Arc::ptr_eq(a, b))
        } else {
            false
        }
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
        self.format.dyn_hash(state);
        self.m.hash(&mut HashWrapper(state));
        self.k.hash(&mut HashWrapper(state));
        for g in &self.groups {
            state.write(g.as_bytes());
        }
    }

    fn eq_storage(&self, other: &dyn TensorStorage) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            self.format.same_as(&*other.format)
                && self.m == other.m
                && self.k == other.k
                && self.groups.len() == other.groups.len()
                && self.groups.iter().zip(&other.groups).all(|(a, b)| a.as_bytes() == b.as_bytes())
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
