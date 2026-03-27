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

    /// Product of all leading dims except the last two (M, K).
    /// For rank <= 2, returns 1.
    pub fn num_groups(&self) -> usize {
        if self.shape.len() <= 2 { 1 } else { self.shape[..self.shape.len() - 2].iter().product() }
    }

    /// Product of all dims except the last (K). This is the flat M
    /// dimension (groups * m_per_group).
    pub fn m(&self) -> usize {
        self.shape[..self.shape.len() - 1].iter().product()
    }

    /// Last dimension.
    pub fn k(&self) -> usize {
        *self.shape.last().unwrap()
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

impl ExoticFact for BlockQuantFact {
    fn buffer_sizes(&self) -> TVec<TDim> {
        let total = self.m() * self.k() / self.format.block_len() * self.format.block_bytes();
        tvec!(total.to_dim())
    }
}

impl PartialEq for BlockQuantFact {
    fn eq(&self, other: &Self) -> bool {
        *self.format == *other.format && self.shape == other.shape
    }
}
impl Eq for BlockQuantFact {}

#[derive(Clone, Hash, PartialEq)]
pub struct PackedBlockQuantFact {
    pub format: PackedBlockQuantFormat,
    pub shape: TVec<usize>,
}
impl Eq for PackedBlockQuantFact {}

impl std::fmt::Debug for PackedBlockQuantFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({:?})", self.format, self.shape)
    }
}

impl ExoticFact for PackedBlockQuantFact {
    fn buffer_sizes(&self) -> TVec<TDim> {
        tvec!(
            (self.shape.iter().product::<usize>() / self.format.bq.block_len()
                * self.format.bq.block_bytes())
            .to_dim()
        )
    }
}
