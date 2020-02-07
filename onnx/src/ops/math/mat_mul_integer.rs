use crate::model::ParsingContext;
use crate::pb::*;
use tract_core::infer::*;
use tract_core::internal::*;
use tract_core::ops::quant::QParams;

pub fn mat_mul_integer(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut options = crate::model::optional_inputs(node).skip(2);
    let op = MatMulInteger::new(options.next().unwrap(), options.next().unwrap());
    Ok((Box::new(op), vec![]))
}

fn cleanup_zero_point(mut t: Tensor) -> TractResult<Option<Tensor>> {
    if t.len() == 1 {
        unsafe {
            t = t.into_shape(&[])?;
        }
    }
    if t.rank() == 0 && t.cast_to_scalar::<f32>()? == 0.0 {
        Ok(None)
    } else {
        Ok(Some(t))
    }
}

#[derive(Debug, Clone, new)]
struct MatMulInteger {
    pub optional_a_zero_point_input: Option<usize>,
    pub optional_b_zero_point_input: Option<usize>,
}

impl Op for MatMulInteger {
    fn name(&self) -> Cow<str> {
        "onnx.MatMulInteger".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for MatMulInteger {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut qp = QParams::new(i32::datum_type());
        if let Some(i) = self.optional_a_zero_point_input {
            if let Some(zp) = cleanup_zero_point(inputs[i].clone().into_tensor())? {
                qp = qp.with_zero_point_a(&zp.into_arc_tensor());
            }
        }
        if let Some(i) = self.optional_b_zero_point_input {
            if let Some(zp) = cleanup_zero_point(inputs[i].clone().into_tensor())? {
                qp = qp.with_zero_point_b(&zp.into_arc_tensor());
            }
        }
        let op = tract_core::ops::matmul::MatMul::default().with_q_params(qp);
        op.eval(inputs)
    }
}

impl InferenceRulesOp for MatMulInteger {
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
                tract_core::ops::matmul::infer_shapes(ashape, bshape, false, false, false)?;
            s.equals(&outputs[0].shape, cshape)
        })?;
        Ok(())
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let mut qp = QParams::new(i32::datum_type());
        if let Some(ix) = self.optional_a_zero_point_input {
            let zp = target
                .outlet_fact(mapping[&node.inputs[ix]])?
                .konst
                .as_ref()
                .ok_or("zero_point_a must be a constant")?;
            if let Some(zp) = cleanup_zero_point(zp.clone().into_tensor())? {
                qp = qp.with_zero_point_a(&zp.into_arc_tensor());
            }
        };
        if let Some(ix) = self.optional_b_zero_point_input {
            let zp = target
                .outlet_fact(mapping[&node.inputs[ix]])?
                .konst
                .as_ref()
                .ok_or("zero_point_b must be a constant")?;
            if let Some(zp) = cleanup_zero_point(zp.clone().into_tensor())? {
                qp = qp.with_zero_point_b(&zp.into_arc_tensor());
            }
        };
        let op = tract_core::ops::matmul::MatMul::default().with_q_params(qp);
        target.wire_node(&*node.name, op, &[mapping[&node.inputs[0]], mapping[&node.inputs[1]]])
    }

    inference_op_as_op!();
}

pub fn q_linear_mat_mul(
    _ctx: &ParsingContext,
    _node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    Ok((Box::new(QLinearMatMul), vec![]))
}

#[derive(Debug, Clone, new)]
struct QLinearMatMul;

impl Op for QLinearMatMul {
    fn name(&self) -> Cow<str> {
        "onnx.QLinearMatMul".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for QLinearMatMul {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp) = args_8!(inputs);
        let scale = a_scale.to_scalar::<f32>()? * b_scale.to_scalar::<f32>()?
            / y_scale.to_scalar::<f32>()?;
        let mut qp = QParams::new(y_zp.datum_type()).with_scale_factor(scale);
        if let Some(zp) = cleanup_zero_point(a_zp.into_tensor())? {
            qp = qp.with_zero_point_a(&zp.into_arc_tensor())
        }
        if let Some(zp) = cleanup_zero_point(b_zp.into_tensor())? {
            qp = qp.with_zero_point_b(&zp.into_arc_tensor())
        }
        if let Some(zp) = cleanup_zero_point(y_zp.into_tensor())? {
            qp = qp.with_zero_point_c(&zp.into_arc_tensor())
        }
        let op = tract_core::ops::matmul::MatMul::default().with_q_params(qp);
        op.eval(tvec!(a, b))
    }
}

impl InferenceRulesOp for QLinearMatMul {
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
                tract_core::ops::matmul::infer_shapes(ashape, bshape, false, false, false)?;
            s.equals(&outputs[0].shape, cshape)
        })?;
        Ok(())
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let mut qp = QParams::new(i32::datum_type());
        let a_zp = target
            .outlet_fact(mapping[&node.inputs[2]])?
            .konst
            .as_ref()
            .ok_or("zero_point_a must be a constant")?;
        if let Some(zp) = cleanup_zero_point(a_zp.clone().into_tensor())? {
            qp = qp.with_zero_point_a(&zp.into_arc_tensor());
        }
        let b_zp = target
            .outlet_fact(mapping[&node.inputs[5]])?
            .konst
            .as_ref()
            .ok_or("zero_point_b must be a constant")?;
        if let Some(zp) = cleanup_zero_point(b_zp.clone().into_tensor())? {
            qp = qp.with_zero_point_b(&zp.into_arc_tensor());
        }
        let op = tract_core::ops::matmul::MatMul::default().with_q_params(qp);
        target.wire_node(&*node.name, op, &[mapping[&node.inputs[0]], mapping[&node.inputs[3]]])
    }

    inference_op_as_op!();
}
