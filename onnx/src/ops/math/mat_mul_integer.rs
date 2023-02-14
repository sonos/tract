use crate::model::ParsingContext;
use crate::pb::*;
use tract_core::ops::matmul::*;
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
        let a_and_b =
            tract_hir::ops::binary::wire_rank_broadcast(prefix, target, &[inputs[0], inputs[1]])?;
        let rank = target.outlet_fact(a_and_b[0])?.rank();
        let a0 = if let Some(o) = self.optional_a_zero_point_input {
            (o + 1).into()
        } else {
            let a_dt = target.outlet_fact(inputs[0])?.datum_type;
            Tensor::zero_scalar_dt(a_dt)?.into()
        };
        let b0 = if let Some(o) = self.optional_b_zero_point_input {
            (o + 1).into()
        } else {
            let b_dt = target.outlet_fact(inputs[1])?.datum_type;
            Tensor::zero_scalar_dt(b_dt)?.into()
        };
        let params = MatMulQParams {
            a0,
            b0,
            c0: tensor0(0i32).into(),
            a_scale: tensor0(1f32).into(),
            b_scale: tensor0(1f32).into(),
            c_scale: tensor0(1f32).into(),
        };
        let op = QMatMul::new(MatMulAxes::default_for_rank(rank), i32::datum_type(), params);
        let mut inputs: TVec<OutletId> = inputs.into();
        inputs[0] = a_and_b[0];
        inputs[1] = a_and_b[1];
        inputs.insert(2, target.add_const(format!("{prefix}.bias"), tensor0(0i32))?);
        target.wire_node(prefix, op, &inputs)
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
        let rank = target.outlet_fact(inputs[0])?.rank().max(target.outlet_fact(inputs[2])?.rank());
        let op = tract_core::ops::matmul::QMatMul::new(
            MatMulAxes::default_for_rank(rank),
            target.outlet_fact(inputs[7])?.datum_type,
            tract_core::ops::matmul::MatMulQParams::all_dynamic(3),
        );
        let a_and_b =
            tract_hir::ops::binary::wire_rank_broadcast(prefix, target, &[inputs[0], inputs[3]])?;
        let bias = target.add_const(format!("{prefix}.bias"), tensor0(0i32))?;
        target.wire_node(
            prefix,
            op,
            &[
                a_and_b[0], a_and_b[1], bias, inputs[2], inputs[1], inputs[5], inputs[4],
                inputs[7], inputs[6],
            ],
        )
    }
}
