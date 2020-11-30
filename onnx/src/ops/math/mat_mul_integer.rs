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

impl_dyn_hash!(MatMulInteger);

impl Expansion for MatMulInteger {
    fn name(&self) -> Cow<str> {
        "MatMulInteger".into()
    }

    op_onnx!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(
            &inputs,
            2 + self.optional_a_zero_point_input.is_some() as usize
                + self.optional_b_zero_point_input.is_some() as usize,
        )?;
        check_output_arity(&outputs, 1)?;
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
        let op = tract_core::ops::matmul::QMatMul::new(false, false, false, i32::datum_type());
        let mut a_and_b =
            tract_hir::ops::binary::wire_rank_broadcast(prefix, target, &[inputs[0], inputs[1]])?;
        let a0 = if let Some(o) = self.optional_a_zero_point_input {
            inputs[o]
        } else {
            let a_dt = target.outlet_fact(inputs[0])?.datum_type;
            target.add_const(format!("{}.a0", prefix), tensor0(0).cast_to_dt(a_dt)?.into_owned())?
        };
        let b0 = if let Some(o) = self.optional_b_zero_point_input {
            inputs[o]
        } else {
            let b_dt = target.outlet_fact(inputs[1])?.datum_type;
            target.add_const(format!("{}.b0", prefix), tensor0(0).cast_to_dt(b_dt)?.into_owned())?
        };
        let inputs = [
            a_and_b.remove(0),
            target.add_const(format!("{}.a_scale", prefix), tensor0(1f32))?,
            a0,
            a_and_b.remove(0),
            target.add_const(format!("{}.b_scale", prefix), tensor0(1f32))?,
            b0,
            target.add_const(format!("{}.c_scale", prefix), tensor0(1f32))?,
            target.add_const(format!("{}.c0", prefix), tensor0(0i32))?,
        ];
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

impl_dyn_hash!(QLinearMatMul);

impl Expansion for QLinearMatMul {
    fn name(&self) -> Cow<str> {
        "QLinearMatMul".into()
    }

    op_onnx!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(&inputs, 8)?;
        check_output_arity(&outputs, 1)?;
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
        let op = tract_core::ops::matmul::QMatMul::new(
            false,
            false,
            false,
            target.outlet_fact(inputs[7])?.datum_type,
        );
        let mut a_and_b =
            tract_hir::ops::binary::wire_rank_broadcast(prefix, target, &[inputs[0], inputs[3]])?;
        let mut inputs:TVec<OutletId> = inputs.into();
        inputs[0] = a_and_b.remove(0);
        inputs[3] = a_and_b.remove(0);
        target.wire_node(prefix, op, &inputs)
    }
}
