use tract_linalg::frame::block_quant::BlockQuant;

use crate::internal::*;

#[derive(Debug, Clone, Hash)]
pub struct DeBlockQuant {
    bq: Box<dyn BlockQuant>,
    fact: TypedFact,
}

impl Op for DeBlockQuant {
    fn name(&self) -> Cow<str> {
        "DeBlockQuant".into()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        other
            .downcast_ref::<Self>()
            .map(|other| other.bq.same_as(&*self.bq) && other.fact == self.fact)
            .unwrap_or(false)
    }

    op_as_typed_op!();
}

impl EvalOp for DeBlockQuant {
    fn is_stateless(&self) -> bool {
        false
    }

    fn eval(&self, _inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        unreachable!()
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(self.clone())))
    }
}

impl OpState for DeBlockQuant {
    fn eval(
        &mut self,
        _session: &mut SessionState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let blob = input.to_scalar::<Blob>()?;
        Ok(tvec!(self.bq.dequant_f32(&blob)?.into_tvalue()))
    }
}

trivial_op_state_freeeze!(DeBlockQuant);

impl TypedOp for DeBlockQuant {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.fact.clone()))
    }
    as_op!();
}
