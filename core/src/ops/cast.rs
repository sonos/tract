use crate::internal::*;

pub fn cast(to: DatumType) -> Cast {
    Cast { to }
}

#[derive(Debug, Clone, new, Hash)]
pub struct Cast {
    pub to: DatumType,
}

impl Cast {
    fn do_eval(&self, input: &Tensor, symbols: &SymbolValues) -> TractResult<TVec<TValue>> {
        if input.datum_type() == self.to {
            Ok(tvec!(input.clone().into_tvalue()))
        } else if input.datum_type() == TDim::datum_type() {
            unsafe {
                let mut tmp = Tensor::uninitialized_dt(i64::datum_type(), input.shape())?;
                for (dim, i) in
                    tract_itertools::izip!(input.as_slice::<TDim>()?, tmp.as_slice_mut::<i64>()?)
                {
                    *i = dim.eval(symbols).to_i64()?
                }
                Ok(tvec!(tmp.cast_to_dt(self.to)?.into_owned().into_tvalue()))
            }
        } else {
            Ok(tvec!(input.cast_to_dt(self.to)?.into_owned().into_tvalue()))
        }
    }
}

impl Op for Cast {
    fn name(&self) -> Cow<str> {
        "Cast".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Cast {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.do_eval(&inputs[0], &Default::default())
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(self.clone())))
    }
}

impl OpState for Cast {
    fn eval(
        &mut self,
        session: &mut SessionState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        self.do_eval(&inputs[0], &session.resolved_symbols)
    }
}

trivial_op_state_freeeze!(Cast);
impl TypedOp for Cast {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.to.fact(inputs[0].shape.clone())))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if model.outlet_fact(node.inputs[0])?.datum_type == self.to {
            TypedModelPatch::shunt_one_op(model, node)
        } else {
            Ok(None)
        }
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
    }

    as_op!();
}
