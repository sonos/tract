use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops::quant::QParams;

pub fn mat_mul_integer(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut options = crate::model::optional_inputs(node).skip(2);
    let op = MatMulInteger::new(options.next().unwrap(), options.next().unwrap());
    Ok((expand(op), vec![]))
}

fn cleanup_zero_point(mut t: Tensor) -> TractResult<Option<Tensor>> {
    if t.len() == 1 {
        t = t.into_shape(&[])?;
    }
    if t.rank() == 0 && t.cast_to_scalar::<f32>()? == 0.0 {
        Ok(None)
    } else {
        Ok(Some(t))
    }
}

#[derive(Debug, Clone, new, Hash)]
struct MatMulInteger {
    pub optional_a_zero_point_input: Option<usize>,
    pub optional_b_zero_point_input: Option<usize>,
}

tract_data::impl_dyn_hash!(MatMulInteger);

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
        let mut qp = QParams::new(i32::datum_type());
        if let Some(ix) = self.optional_a_zero_point_input {
            let zp = target
                .outlet_fact(inputs[ix])?
                .konst
                .as_ref()
                .context("zero_point_a must be a constant")?;
            if let Some(zp) = cleanup_zero_point(zp.clone().into_tensor())? {
                qp = qp.with_zero_point_a(&zp.into_arc_tensor());
            }
        };
        if let Some(ix) = self.optional_b_zero_point_input {
            let zp = target
                .outlet_fact(inputs[ix])?
                .konst
                .as_ref()
                .context("zero_point_b must be a constant")?;
            if let Some(zp) = cleanup_zero_point(zp.clone().into_tensor())? {
                qp = qp.with_zero_point_b(&zp.into_arc_tensor());
            }
        };
        let op = tract_hir::ops::matmul::MatMul::default().with_q_params(qp);
        let inputs = tract_hir::ops::binary::wire_rank_broadcast(prefix, target, &[inputs[0], inputs[1]])?;
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

tract_data::impl_dyn_hash!(QLinearMatMul);

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
        let mut qp = QParams::new(target.outlet_fact(inputs[7])?.datum_type);
        // a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
        let mut konsts: Vec<Tensor> = [2, 5, 7, 1, 4, 6]
            .iter()
            .rev()
            .map(|ix| {
                Ok(target
                    .outlet_fact(inputs[*ix])?
                    .konst
                    .clone()
                    .with_context(|| format!("Input {} must be a constant", ix))?
                    .into_tensor())
            })
            .collect::<TractResult<Vec<_>>>()?;
        if let Some(zp) = cleanup_zero_point(konsts.pop().unwrap())? {
            qp = qp.with_zero_point_a(&zp.into_arc_tensor());
        }
        if let Some(zp) = cleanup_zero_point(konsts.pop().unwrap())? {
            qp = qp.with_zero_point_b(&zp.into_arc_tensor());
        }
        if let Some(zp) = cleanup_zero_point(konsts.pop().unwrap())? {
            qp = qp.with_zero_point_c(&zp.into_arc_tensor());
        }
        let scale = konsts.pop().unwrap().to_scalar::<f32>()?
            * konsts.pop().unwrap().to_scalar::<f32>()?
            / konsts.pop().unwrap().to_scalar::<f32>()?;

        qp = qp.with_scale_factor(scale);
        let op = tract_hir::ops::matmul::MatMul::default().with_q_params(qp);
        let inputs = tract_hir::ops::binary::wire_rank_broadcast(prefix, target, &[inputs[0], inputs[3]])?;
        target.wire_node(prefix, op, &inputs)
    }
}
