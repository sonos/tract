use super::{BlockQuant, PackedBlockQuantFormat};
use tract_data::internal::*;
use tract_data::TVec;

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Hash)]
pub struct BlockQuantFact {
    pub format: Box<dyn BlockQuant>,
    pub shape: TVec<usize>,
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
    pub value: Blob,
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
