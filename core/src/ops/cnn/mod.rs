use crate::internal::*;

pub mod conv;
pub mod deconv;
mod maxpool;
mod padding;
mod patch_axis;
mod patches;
pub mod pools;
mod sumpool;

pub use self::conv::{ConvUnary, KernelFormat};
pub use self::deconv::DeconvUnary;
pub use self::maxpool::MaxPool;
pub use self::padding::PaddingSpec;
pub use self::patch_axis::PatchAxis;
pub use self::patches::{Patch, PatchSpec};
pub use self::pools::PoolSpec;
pub use self::sumpool::SumPool;

pub trait ResolveSymbolsTo<Concrete> {
    fn resolve(&self, input_full_shape: &[usize]) -> TractResult<Concrete>;
}

#[derive(Debug, Clone, Hash, PartialEq)]
pub enum GeometryBound<Symbolic, Concrete> {
    Symbolic(Symbolic),
    Concrete(Concrete),
}

impl<S: ResolveSymbolsTo<C>, C: Clone> GeometryBound<S, C> {
    pub fn is_concrete(&self) -> bool {
        match self {
            GeometryBound::Concrete { .. } => true,
            GeometryBound::Symbolic { .. } => false,
        }
    }

    pub fn into_concrete(self, input_full_shape: &[usize]) -> TractResult<Self> {
        match self {
            Self::Symbolic(sym) => Ok(Self::Concrete(sym.resolve(input_full_shape)?)),
            Self::Concrete(conc) => Ok(Self::Concrete(conc)),
        }
    }

    pub fn to_concrete(&self, input_full_shape: &[usize]) -> TractResult<Cow<C>> {
        match self {
            Self::Symbolic(sym) => Ok(Cow::Owned(sym.resolve(input_full_shape)?)),
            Self::Concrete(conc) => Ok(Cow::Borrowed(conc)),
        }
    }

    pub fn optimize(self, input_full_shape: &[TDim]) -> TractResult<Self> {
        if self.is_concrete() {
            return Ok(self)
        }
        if let Ok(input_full_shape) =
            input_full_shape.iter().map(|x| x.to_usize()).collect::<TractResult<TVec<usize>>>()
        {
            self.into_concrete(&input_full_shape)
        } else {
            Ok(self)
        }
    }
}

impl<S, C> From<S> for GeometryBound<S, C> {
    fn from(s: S) -> Self {
        GeometryBound::Symbolic(s)
    }
}
