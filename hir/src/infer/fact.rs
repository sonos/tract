use std::convert::TryFrom;
use std::fmt;
use std::sync::Arc;

use super::factoid::*;
use crate::internal::*;

/// Partial information about a tensor.
///
/// The task of the analyser is to tag every edge in the graph with information
/// about the tensors that flow through it - specifically their datum_type, their
/// shape and possibly their value. During the analysis, however, we might only
/// know some of that information (say, for instance, that an edge only carries
/// tensors of rank 4, but without knowing their precise dimension).
///
/// This is where tensor facts come in: they hold partial information about the
/// datum_type, shape and value of tensors that might flow through an edge of the
/// graph. The analyser will first tag each edge with a fact, starting with the
/// most general one and specializing it at each iteration. Eventually, it will
/// reach a fixed point that - hopefully - holds enough information.
#[derive(Clone, PartialEq, Eq, Default, Hash)]
pub struct InferenceFact {
    pub datum_type: TypeFactoid,
    pub shape: ShapeFactoid,
    pub value: ValueFact,
}

impl InferenceFact {
    /// Constructs the most general tensor fact possible.
    pub fn new() -> InferenceFact {
        InferenceFact::default()
    }

    pub fn any() -> InferenceFact {
        InferenceFact::default()
    }

    pub fn dt(dt: DatumType) -> InferenceFact {
        InferenceFact::default().with_datum_type(dt)
    }

    pub fn dt_shape<S: Into<ShapeFactoid>>(dt: DatumType, shape: S) -> InferenceFact {
        InferenceFact::dt(dt).with_shape(shape)
    }

    pub fn shape<S: Into<ShapeFactoid>>(shape: S) -> InferenceFact {
        InferenceFact::default().with_shape(shape)
    }

    pub fn with_datum_type(self, dt: DatumType) -> InferenceFact {
        InferenceFact { datum_type: dt.into(), ..self }
    }

    pub fn without_datum_type(self) -> InferenceFact {
        InferenceFact { datum_type: TypeFactoid::Any, ..self }
    }

    pub fn with_shape<S: Into<ShapeFactoid>>(self, shape: S) -> InferenceFact {
        InferenceFact { shape: shape.into(), ..self }
    }

    pub fn format_dt_shape(&self) -> String {
        if !self.shape.open && self.shape.dims.len() == 0 {
            self.datum_type
                .concretize()
                .map(|dt| format!("{dt:?}"))
                .unwrap_or_else(|| "?".to_string())
        } else {
            format!(
                "{:?},{}",
                self.shape,
                self.datum_type
                    .concretize()
                    .map(|dt| format!("{dt:?}"))
                    .unwrap_or_else(|| "?".to_string())
            )
        }
    }

    pub fn dt_shape_from_tensor(t: &Tensor) -> InferenceFact {
        InferenceFact::dt_shape(t.datum_type(), t.shape())
    }

    pub fn without_value(self) -> InferenceFact {
        InferenceFact { value: GenericFactoid::Any, ..self }
    }
}

impl Factoid for InferenceFact {
    type Concrete = Arc<Tensor>;

    /// Tries to transform the fact into a concrete value.
    fn concretize(&self) -> Option<Self::Concrete> {
        self.value.concretize()
    }

    /// Tries to unify the fact with another fact of the same type.
    fn unify(&self, other: &Self) -> TractResult<Self> {
        let tensor = InferenceFact {
            datum_type: self.datum_type.unify(&other.datum_type)?,
            shape: self.shape.unify(&other.shape)?,
            value: self.value.unify(&other.value)?,
        };

        trace!("Unifying {self:?} with {other:?} into {tensor:?}.");

        Ok(tensor)
    }
}

impl fmt::Debug for InferenceFact {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        if let Some(t) = self.value.concretize() {
            write!(formatter, "{t:?}")
        } else {
            write!(formatter, "{}", self.format_dt_shape())
        }
    }
}

use crate::infer::factoid::Factoid;

impl Fact for InferenceFact {
    fn to_typed_fact(&self) -> TractResult<Cow<TypedFact>> {
        Ok(Cow::Owned(TypedFact::try_from(self)?))
    }

    fn matches(&self, t: &Tensor, _symbols: Option<&SymbolValues>) -> TractResult<bool> {
        if let Some(dt) = self.datum_type() {
            if t.datum_type() != dt {
                return Ok(false);
            }
        }
        if let Some(shape) = self.shape.concretize() {
            if *ShapeFact::from(t.shape()) != *shape {
                return Ok(false);
            }
        }
        if let Some(value) = self.value.concretize() {
            if &*value != t {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn same_as(&self, other: &dyn Fact) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            self.unify(other).is_ok()
        } else {
            false
        }
    }

    fn compatible_with(&self, other: &dyn Fact) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            self.unify(other).is_ok()
        } else {
            false
        }
    }

    fn datum_type(&self) -> Option<DatumType> {
        self.datum_type.concretize()
    }
}

impl TryFrom<&InferenceFact> for TypedFact {
    type Error = TractError;
    fn try_from(fact: &InferenceFact) -> TractResult<TypedFact> {
        if let (Some(datum_type), Some(shape)) =
            (fact.datum_type.concretize(), fact.shape.concretize())
        {
            let shape = ShapeFact::from_dims(shape);
            let konst = fact.value.concretize();
            let uniform = konst.as_ref().and_then(|k| k.as_uniform()).map(Arc::new);
            Ok(TypedFact { datum_type, shape, konst, uniform, opaque_fact: None })
        } else {
            bail!("Can not make a TypedFact out of {:?}", fact)
        }
    }
}

impl<'a> From<&'a InferenceFact> for InferenceFact {
    fn from(t: &'a InferenceFact) -> InferenceFact {
        t.clone()
    }
}

impl<'a> From<&'a TypedFact> for InferenceFact {
    fn from(t: &'a TypedFact) -> InferenceFact {
        let mut fact = InferenceFact::dt_shape(t.datum_type, t.shape.iter());
        if let Some(k) = &t.konst {
            fact.value = Arc::clone(k).into();
        }
        fact
    }
}

impl From<TypedFact> for InferenceFact {
    fn from(t: TypedFact) -> InferenceFact {
        InferenceFact::from(&t)
    }
}

impl<'a> From<&'a Arc<Tensor>> for InferenceFact {
    fn from(t: &'a Arc<Tensor>) -> InferenceFact {
        InferenceFact::from(&TypedFact::from(Arc::clone(t)))
    }
}

impl From<Arc<Tensor>> for InferenceFact {
    fn from(t: Arc<Tensor>) -> InferenceFact {
        InferenceFact::from(&TypedFact::from(t))
    }
}

impl From<Tensor> for InferenceFact {
    fn from(t: Tensor) -> InferenceFact {
        let mut fact = InferenceFact::dt_shape(t.datum_type(), t.shape());
        fact.value = t.into_arc_tensor().into();
        fact
    }
}
