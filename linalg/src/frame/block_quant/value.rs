use std::ops::Range;
use std::sync::Arc;

use super::{BlockQuant, PackedBlockQuantFormat};
use tract_data::internal::*;
use tract_data::TVec;

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Hash)]
pub struct BlockQuantFact {
    pub format: Box<dyn BlockQuant>,
    shape: TVec<usize>,
}
impl BlockQuantFact {
    pub fn new(format: Box<dyn BlockQuant>, shape: TVec<usize>) -> Self {
        Self { format, shape }
    }

    pub fn m(&self) -> usize {
        self.shape[0]
    }

    pub fn k(&self) -> usize {
        self.shape.iter().skip(1).product()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl std::fmt::Debug for BlockQuantFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:?})", self.format, self.shape)
    }
}

impl OpaqueFact for BlockQuantFact {
    fn mem_size(&self) -> TDim {
        (self.shape.iter().product::<usize>() / self.format.block_len() * self.format.block_bytes())
            .to_dim()
    }

    fn same_as(&self, other: &dyn OpaqueFact) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| o == self)
    }
}

impl PartialEq for BlockQuantFact {
    fn eq(&self, other: &Self) -> bool {
        self.format.same_as(&*other.format) && self.shape == other.shape
    }
}

#[derive(Clone, Hash)]
pub struct BlockQuantValue {
    pub fact: BlockQuantFact,
    pub value: Arc<Blob>,
}

impl BlockQuantValue {
    pub fn split_rows(&self, range: Range<usize>) -> TractResult<BlockQuantValue> {
        let row_bytes =
            self.fact.k() / self.fact.format.block_len() * self.fact.format.block_bytes();
        let mut value =
            unsafe { Blob::new_for_size_and_align(range.len() * row_bytes, vector_size()) };
        value.copy_from_slice(&self.value[range.start * row_bytes..][..range.len() * row_bytes]);
        let mut shape = self.fact.shape.clone();
        shape[0] = range.len();
        Ok(BlockQuantValue {
            fact: BlockQuantFact { format: self.fact.format.clone(), shape },
            value: Arc::new(value),
        })
    }
}

impl OpaquePayload for BlockQuantValue {
    fn same_as(&self, other: &dyn OpaquePayload) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| o.fact == self.fact && o.value == self.value)
    }
}

impl std::fmt::Debug for BlockQuantValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {:?}", self.fact, self.value)
    }
}

impl std::fmt::Display for BlockQuantValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Clone, Hash, PartialEq)]
pub struct PackedBlockQuantFact {
    pub format: PackedBlockQuantFormat,
    pub shape: TVec<usize>,
}

impl std::fmt::Debug for PackedBlockQuantFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:?})", self.format, self.shape)
    }
}

impl OpaqueFact for PackedBlockQuantFact {
    fn mem_size(&self) -> TDim {
        (self.shape.iter().product::<usize>() / self.format.bq.block_len()
            * self.format.bq.block_bytes())
        .to_dim()
    }
    fn same_as(&self, other: &dyn OpaqueFact) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| o == self)
    }
}
