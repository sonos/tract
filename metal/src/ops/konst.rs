use crate::IntoMetal;
use crate::MetalFact;
use crate::MetalTensor;
use tract_core::internal::*;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct MetalConst(pub Arc<Tensor>, pub TypedFact);

impl MetalConst {
    pub fn new(c: Arc<Tensor>) -> TractResult<Option<Self>> {
        if !MetalTensor::is_supported_dt(c.datum_type()) {
            return Ok(None);
        }
        let fact = TypedFact::dt_scalar(DatumType::Opaque)
            .with_opaque_fact(MetalFact(Arc::clone(&c).into()));
        Ok(Some(Self(c.into_metal()?.into_opaque_tensor().into_arc_tensor(), fact)))
    }
}

impl Op for MetalConst {
    fn name(&self) -> Cow<str> {
        "MetalConst".into()
    }

    op_as_typed_op!();
    impl_op_same_as!();
}

impl EvalOp for MetalConst {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, _inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(tvec![self.0.clone().into_tvalue()])
    }
}

impl TypedOp for MetalConst {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.1.clone()))
    }
}
