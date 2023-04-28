use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;

pub fn mat_mul_integer(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut options = crate::model::optional_inputs(node).skip(2);
    let op = MatMulInteger::new(options.next().unwrap(), options.next().unwrap());
    Ok((expand(op), vec![]))
}

#[derive(Debug, Clone, new, Hash)]
struct MatMulInteger {
    pub optional_a_zero_point_input: Option<usize>,
    pub optional_b_zero_point_input: Option<usize>,
}

impl Expansion for MatMulInteger {
    fn name(&self) -> Cow<str> {
        "MatMulInteger".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(
            inputs,
            2 + self.optional_a_zero_point_input.is_some() as usize
                + self.optional_b_zero_point_input.is_some() as usize,
        )?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, i32::datum_type())?;
        if let Some(a_zp) = self.optional_a_zero_point_input {
            s.equals(&inputs[a_zp].datum_type, &inputs[0].datum_type)?
        }
        if let Some(b_zp) = self.optional_b_zero_point_input {
            s.equals(&inputs[b_zp].datum_type, &inputs[1].datum_type)?
        }
        s.given_2(&inputs[0].shape, &inputs[1].shape, move |s, ashape, bshape| {
            let (_, _, cshape, _) =
                tract_hir::ops::matmul::compute_shapes(ashape, bshape, false, false, false)?;
            s.equals(&outputs[0].shape, cshape)
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let mut new_inputs =
            tract_hir::ops::binary::wire_rank_broadcast(prefix, target, &[inputs[0], inputs[1]])?;
        new_inputs.push(target.add_const(format!("{prefix}.bias"), tensor0(0i32))?);
        if let Some(o) = self.optional_a_zero_point_input {
            new_inputs.push(inputs[o]);
        } else {
            new_inputs.push(target.add_const(format!("{prefix}.a0"), tensor0(0i32))?);
        };
        new_inputs.push(target.add_const(format!("{prefix}.a_scale"), tensor0(1f32))?);
        if let Some(o) = self.optional_b_zero_point_input {
            new_inputs.push(inputs[o]);
        } else {
            new_inputs.push(target.add_const(format!("{prefix}.b0"), tensor0(0i32))?);
        };
        new_inputs.push(target.add_const(format!("{prefix}.b_scale"), tensor0(1f32))?);
        new_inputs.push(target.add_const(format!("{prefix}.c0"), tensor0(0i32))?);
        new_inputs.push(target.add_const(format!("{prefix}.c_scale"), tensor0(1f32))?);
        wire_as_einsum(prefix, target, &new_inputs, i32::datum_type())
    }
}

pub fn q_linear_mat_mul(
    _ctx: &ParsingContext,
    _node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    Ok((expand(QLinearMatMul), vec![]))
}

#[derive(Debug, Clone, new, Hash)]
struct QLinearMatMul;

impl Expansion for QLinearMatMul {
    fn name(&self) -> Cow<str> {
        "QLinearMatMul".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(inputs, 8)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &inputs[2].datum_type)?;
        s.equals(&inputs[3].datum_type, &inputs[5].datum_type)?;
        s.equals(&inputs[1].datum_type, f32::datum_type())?;
        s.equals(&inputs[4].datum_type, f32::datum_type())?;
        s.equals(&inputs[6].datum_type, f32::datum_type())?;
        s.equals(&outputs[0].datum_type, &inputs[7].datum_type)?;
        s.equals(&inputs[1].rank, &inputs[2].rank)?;
        s.equals(&inputs[4].rank, &inputs[5].rank)?;
        s.equals(&inputs[6].rank, &inputs[7].rank)?;
        s.given_2(&inputs[0].shape, &inputs[3].shape, move |s, ashape, bshape| {
            let (_, _, _, cshape) =
                tract_hir::ops::matmul::compute_shapes(ashape, bshape, false, false, false)?;
            s.equals(&outputs[0].shape, cshape)
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let mut new_inputs =
            tract_hir::ops::binary::wire_rank_broadcast(prefix, target, &[inputs[0], inputs[3]])?;
        new_inputs.push(target.add_const(format!("{prefix}.bias"), tensor0(0i32))?);
        for i in [2, 1, 5, 4, 7, 6] {
            new_inputs.push(inputs[i]);
        }
        wire_as_einsum(prefix, target, &new_inputs, target.outlet_fact(inputs[7])?.datum_type)
    }
}

fn wire_as_einsum(
    prefix: &str,
    target: &mut TypedModel,
    inputs: &[OutletId],
    output: DatumType,
) -> TractResult<TVec<OutletId>> {
    assert!(inputs.len() == 9);
    let rank = target.outlet_fact(inputs[0])?.rank();
    let ranks = inputs
        .iter()
        .map(|i| Ok(target.outlet_fact(*i)?.rank()))
        .collect::<TractResult<Vec<_>>>()?;
    let mut expr = AxesMapping::disconnected_for_ranks(&ranks, &ranks[0..1])?;
    expr = expr
        .renaming((InOut::In(0), rank - 2), 'm')?
        .linking('m', (InOut::Out(0), rank - 2))?
        .renaming((InOut::In(1), rank - 1), 'n')?
        .linking('n', (InOut::Out(0), rank - 1))?
        .renaming((InOut::In(0), rank - 1), 'k')?
        .linking('k', (InOut::In(1), rank - 2))?;
    for ax in 0..rank - 2 {
        expr = expr
            .linking((InOut::In(0), ax), (InOut::In(1), ax))?
            .linking((InOut::In(0), ax), (InOut::Out(0), ax))?;
    }
    if ranks[2] == 1 {
        expr = expr.linking('m', (InOut::In(2), 0))?;
    }
    if ranks[3] == 1 {
        expr = expr.linking('m', (InOut::In(3), 0))?;
    }
    if ranks[4] == 1 {
        expr = expr.linking('m', (InOut::In(4), 0))?;
    }
    if ranks[5] == 1 {
        expr = expr.linking('n', (InOut::In(5), 0))?;
    }
    if ranks[6] == 1 {
        expr = expr.linking('n', (InOut::In(6), 0))?;
    }
    if ranks[7] == 1 {
        expr = expr.linking('m', (InOut::In(7), 0))?;
    }
    if ranks[8] == 1 {
        expr = expr.linking('m', (InOut::In(8), 0))?;
    }
    let op = tract_core::ops::einsum::EinSum {
        axes: expr,
        operating_dt: i32::datum_type(),
        q_params: Some(output),
    };
    target.wire_node(prefix, op, inputs)
}
