use super::BlockQuant;
use tract_data::internal::*;
use tract_data::TVec;

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
}

#[derive(Clone, Hash)]
pub struct BlockQuantValue {
    pub fact: BlockQuantFact,
    pub value: Blob,
}

impl OpaquePayload for BlockQuantValue {}

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
