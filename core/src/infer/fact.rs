use std::convert::TryFrom;
use std::fmt;
use std::sync::Arc;

use super::factoid::*;
use crate::prelude::*;
use crate::errors::*;
use crate::model::{ Fact, TypedFact, ShapeFact, StreamFact };
use crate::tensor::Tensor;
use crate::datum::DatumType;
use crate::dim::TDim;
use crate::dim::ToDim;

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
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Clone, PartialEq, Default)]
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

    pub fn with_shape<S: Into<ShapeFactoid>>(self, shape: S) -> InferenceFact {
        InferenceFact { shape: shape.into(), ..self }
    }

    pub fn with_streaming_shape<S: IntoIterator<Item = Option<usize>>>(
        self,
        shape: S,
    ) -> InferenceFact {
        let shape: ShapeFactoid = shape
            .into_iter()
            .map(|d| d.map(|d| (d as isize).to_dim()).unwrap_or(TDim::s()))
            .collect();
        self.with_shape(shape)
    }

    pub fn stream_info(&self) -> TractResult<Option<StreamFact>> {
        self.shape.stream_info()
    }

    pub fn format_dt_shape(&self) -> String {
        if !self.shape.open && self.shape.dims.len() == 0 {
            format!(
                "{}",
                self.datum_type
                    .concretize()
                    .map(|dt| format!("{:?}", dt))
                    .unwrap_or("?".to_string())
            )
        } else {
            format!(
                "{:?}x{}",
                self.shape,
                self.datum_type
                    .concretize()
                    .map(|dt| format!("{:?}", dt))
                    .unwrap_or("?".to_string())
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

        trace!("Unifying {:?} with {:?} into {:?}.", self, other, tensor);

        Ok(tensor)
    }
}

impl<V: Into<Arc<Tensor>>> From<V> for InferenceFact {
    fn from(v: V) -> InferenceFact {
        let v: Arc<Tensor> = v.into();
        InferenceFact {
            datum_type: GenericFactoid::Only(v.datum_type()),
            shape: ShapeFactoid::from(v.shape()),
            value: GenericFactoid::Only(v),
        }
    }
}

impl fmt::Debug for InferenceFact {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        if let Some(t) = self.value.concretize() {
            write!(formatter, "{:?}", t)
        } else {
            write!(formatter, "{}", self.format_dt_shape())
        }
    }
}


use crate::infer::factoid::Factoid;

impl Fact for InferenceFact {
    fn to_typed_fact(&self) -> TractResult<TypedFact> {
        TypedFact::try_from(self)
    }

    fn matches(&self, t: &Tensor) -> TractResult<bool> {
        Ok(self.unify(&InferenceFact::from(t)).is_ok())
    }
}

impl<'a> TryFrom<&'a InferenceFact> for TypedFact {
    type Error = TractError;
    fn try_from(fact: &InferenceFact) -> TractResult<TypedFact> {
        if let (Some(datum_type), Some(shape)) =
            (fact.datum_type.concretize(), fact.shape.concretize())
        {
            let shape = ShapeFact::from_dims(shape)?;
            Ok(TypedFact { datum_type, shape, konst: fact.value.concretize() })
        } else {
            bail!("Can not make a TypedFact out of {:?}", fact)
        }
    }
}


impl<'a> From<&'a Tensor> for InferenceFact {
    fn from(t: &'a Tensor) -> InferenceFact {
        InferenceFact::from(t.clone())
    }
}

impl<'a> From<&'a InferenceFact> for InferenceFact {
    fn from(t: &'a InferenceFact) -> InferenceFact {
        t.clone()
    }
}

impl<'a> From<&'a TypedFact> for InferenceFact {
    fn from(t: &'a TypedFact) -> InferenceFact {
        if let Some(v) = &t.konst {
            v.clone().into_tensor().into()
        } else {
            InferenceFact::dt_shape(t.datum_type, t.shape.iter())
        }
    }
}
