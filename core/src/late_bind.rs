use crate::prelude::TractResult;
use std::borrow::Cow;

pub trait ResolveTo<Concrete> {
    type Param: ?Sized;
    fn resolve(&self, param: &Self::Param) -> TractResult<Concrete>;
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum GeometryBound<Symbolic, Concrete> {
    Symbolic(Symbolic),
    Concrete(Concrete),
}

impl<S: ResolveTo<C>, C: Clone> GeometryBound<S, C> {
    pub fn is_concrete(&self) -> bool {
        match self {
            GeometryBound::Concrete { .. } => true,
            GeometryBound::Symbolic { .. } => false,
        }
    }

    pub fn into_concrete(self, param: &S::Param) -> TractResult<Self> {
        match self {
            Self::Symbolic(sym) => Ok(Self::Concrete(sym.resolve(param)?)),
            Self::Concrete(conc) => Ok(Self::Concrete(conc)),
        }
    }

    pub fn to_concrete(&self, param: &S::Param) -> TractResult<Cow<C>> {
        match self {
            Self::Symbolic(sym) => Ok(Cow::Owned(sym.resolve(param)?)),
            Self::Concrete(conc) => Ok(Cow::Borrowed(conc)),
        }
    }

    pub fn as_concrete(&self) -> Option<&C> {
        if let Self::Concrete(conc) = self {
            Some(conc)
        } else {
            None
        }
    }

    pub fn optimize_if(self, param: Option<&S::Param>) -> TractResult<Self> {
        if let Some(param) = param {
            self.into_concrete(param)
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
