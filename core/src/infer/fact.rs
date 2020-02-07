use crate::errors::*;
use crate::model::{ Fact, TypedFact, ShapeInfo };
use crate::tensor::Tensor;
use std::convert::TryFrom;

pub use crate::analyser::types::InferenceFact;

use crate::analyser::types::Factoid;

impl Fact for InferenceFact {
    fn to_tensor_fact(&self) -> InferenceFact {
        self.clone()
    }
}

impl<'a> TryFrom<&'a InferenceFact> for TypedFact {
    type Error = TractError;
    fn try_from(fact: &InferenceFact) -> TractResult<TypedFact> {
        if let (Some(datum_type), Some(shape)) =
            (fact.datum_type.concretize(), fact.shape.concretize())
        {
            let shape = ShapeInfo::from_dims(shape)?;
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

