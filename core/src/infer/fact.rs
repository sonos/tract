use crate::errors::*;
use crate::model::{ Fact, TypedFact, ShapeFact };
use crate::tensor::Tensor;
use std::convert::TryFrom;

pub use crate::analyser::types::InferenceFact;

use crate::analyser::types::Factoid;

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
            use crate::tensor::IntoTensor;
            v.clone().into_tensor().into()
        } else {
            InferenceFact::dt_shape(t.datum_type, t.shape.iter())
        }
    }
}
