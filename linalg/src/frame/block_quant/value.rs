use super::{BlockQuant, PackedBlockQuantFormat};
use tract_data::TVec;
use tract_data::internal::*;

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
    fn same_as(&self, other: &dyn OpaqueFact) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| o == self)
    }

    fn buffer_sizes(&self) -> TVec<TDim> {
        tvec!(
            (self.shape.iter().product::<usize>() / self.format.block_len()
                * self.format.block_bytes())
            .to_dim()
        )
    }
}

impl PartialEq for BlockQuantFact {
    fn eq(&self, other: &Self) -> bool {
        self.format.same_as(&*other.format) && self.shape == other.shape
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
    fn same_as(&self, other: &dyn OpaqueFact) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| o == self)
    }

    fn buffer_sizes(&self) -> TVec<TDim> {
        tvec!(
            (self.shape.iter().product::<usize>() / self.format.bq.block_len()
                * self.format.bq.block_bytes())
            .to_dim()
        )
    }
}
