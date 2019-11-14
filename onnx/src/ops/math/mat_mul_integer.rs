use crate::model::ParsingContext;
use crate::pb::*;
use tract_core::internal::*;

pub fn mat_mul_integer(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut options = crate::model::optional_inputs(node).skip(2);
    let op = MatMulInteger::new(options.next().unwrap(), options.next().unwrap());
    Ok((Box::new(op), vec![]))
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
        let mut op = tract_core::ops::math::mat_mul::MatMul::default();
        if let Some(i) = self.optional_a_zero_point_input {
            op = op.with_zero_point_a(&inputs[i]);
        }
        if let Some(i) = self.optional_b_zero_point_input {
            op = op.with_zero_point_b(&inputs[i]);
        }
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
            let (_, _, cshape) =
                tract_core::ops::math::mat_mul::infer_shapes(ashape, bshape, false, false, false)?;
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
        let mut op = tract_core::ops::math::mat_mul::MatMul::default();
        if let Some(ix) = self.optional_a_zero_point_input {
            let zp = target
                .outlet_fact(mapping[&node.inputs[ix]])?
                .konst
                .as_ref()
                .ok_or("zero_point_a must be a constant")?;
            op = op.with_zero_point_a(&zp);
        };
        if let Some(ix) = self.optional_b_zero_point_input {
            let zp = target
                .outlet_fact(mapping[&node.inputs[ix]])?
                .konst
                .as_ref()
                .ok_or("zero_point_b must be a constant")?;
            op = op.with_zero_point_b(&zp);
        };
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
        let (a, _a_scale, a_zp, b, _b_scale, b_zp, _y_scale, _z_zp) = args_8!(inputs);
        let op = tract_core::ops::math::mat_mul::MatMul::default()
            .with_zero_point_a(&a_zp)
            .with_zero_point_b(&b_zp);
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
            let (_, _, cshape) =
                tract_core::ops::math::mat_mul::infer_shapes(ashape, bshape, false, false, false)?;
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
        let a_zp = target
            .outlet_fact(mapping[&node.inputs[2]])?
            .konst
            .as_ref()
            .ok_or("zero_point_a must be a constant")?;
        let b_zp = target
            .outlet_fact(mapping[&node.inputs[5]])?
            .konst
            .as_ref()
            .ok_or("zero_point_b must be a constant")?;
        let op = tract_core::ops::math::mat_mul::MatMul::default()
            .with_zero_point_a(&a_zp)
            .with_zero_point_b(&b_zp);
        target.wire_node(&*node.name, op, &[mapping[&node.inputs[0]], mapping[&node.inputs[3]]])
    }

    inference_op_as_op!();
}
