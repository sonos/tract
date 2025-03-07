use crate::internal::*;

pub fn cast(to: DatumType) -> Cast {
    Cast { to }
}

pub fn wire_cast(
    prefix: impl AsRef<str>,
    target: &mut TypedModel,
    inputs: &[OutletId],
    operating_datum_type: DatumType,
) -> TractResult<TVec<OutletId>> {
    let prefix = prefix.as_ref();
    let mut wires = tvec!();
    for mut wire in inputs.iter().copied() {
        if target.outlet_fact(wire)?.datum_type != operating_datum_type {
            wire = target.wire_node(
                target.unique_name(format!("{prefix}.cast")),
                crate::ops::cast::cast(operating_datum_type),
                &[wire],
            )?[0];
        }
        wires.push(wire);
    }
    Ok(wires)
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct Cast {
    pub to: DatumType,
}

impl Op for Cast {
    fn name(&self) -> Cow<str> {
        "Cast".into()
    }

    op_as_typed_op!();
    impl_op_same_as!();
}

impl EvalOp for Cast {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        state: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        if input.datum_type() == self.to {
            Ok(tvec!(input))
        } else if input.datum_type() == TDim::datum_type() {
            let mut tmp = Tensor::zero_dt(i64::datum_type(), input.shape())?;
            for (dim, i) in
                tract_itertools::izip!(input.as_slice::<TDim>()?, tmp.as_slice_mut::<i64>()?)
            {
                *i = dim.eval(&state.resolved_symbols).to_i64()?
            }
            Ok(tvec!(tmp.cast_to_dt(self.to)?.into_owned().into_tvalue()))
        } else {
            Ok(tvec!(input.cast_to_dt(self.to)?.into_owned().into_tvalue()))
        }
    }
}

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

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        _model: &TypedModel,
        node: &TypedNode,
        _prefix: &str,
        inputs: &[OutletId],
        _output_axis: usize,
        _start: &TDim,
        _end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        patch.wire_node(&node.name, &node.op, inputs).map(Some)
    }

    as_op!();
}
