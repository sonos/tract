use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use tract_core::infer::*;
use tract_core::internal::*;
use tract_core::ops::quant::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("QuantizeLinear", quantize_linear);
    reg.insert("DequantizeLinear", dequantize_linear);
}

fn quantize_linear(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let op = QuantizeLinear::new(Some(2).filter(|_| node.input.len() == 3));
    Ok((Box::new(op), vec![]))
}

fn dequantize_linear(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let op = DequantizeLinear::new(Some(2).filter(|_| node.input.len() == 3));
    Ok((Box::new(op), vec![]))
}

#[derive(Debug, Clone, new, Default)]
pub struct QuantizeLinear {
    optional_zero_point_input: Option<usize>,
}

impl Op for QuantizeLinear {
    fn name(&self) -> Cow<str> {
        "onnx.QuantizeLinear".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for QuantizeLinear {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (x, y_scale, y_zero_point) = if self.optional_zero_point_input.is_some() {
            args_3!(inputs)
        } else {
            let (x, y_scale) = args_2!(inputs);
            (x, y_scale, rctensor0(0u8))
        };
        let y_scale = y_scale.as_slice::<f32>()?[0].recip();
        let x = x.cast_to::<f32>()?;
        let tensor = if y_zero_point.datum_type() == u8::datum_type() {
            let y_zero_point = y_zero_point.as_slice::<u8>()?[0];
            x.to_array_view::<f32>()?
                .map(|x| ((x * y_scale).round() as i32 + y_zero_point as i32).max(0).min(255) as u8)
                .into_arc_tensor()
        } else {
            let y_zero_point = y_zero_point.as_slice::<i8>()?[0];
            x.to_array_view::<f32>()?
                .map(|x| {
                    ((x * y_scale).round() as i32 + y_zero_point as i32).max(-128).min(127) as i8
                })
                .into_arc_tensor()
        };
        Ok(tvec!(tensor))
    }
}

impl InferenceRulesOp for QuantizeLinear {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(&inputs, 2 + self.optional_zero_point_input.is_some() as usize)?;
        check_output_arity(&outputs, 1)?;
        //         s.equals(&inputs[1].rank, 0)?; broken in Onnx test suite
        s.equals(&inputs[1].datum_type, f32::datum_type())?;
        if self.optional_zero_point_input.is_some() {
            s.equals(&outputs[0].datum_type, &inputs[2].datum_type)?;
        //            s.equals(&inputs[2].rank, 0)?; // broken in Onnx test suite
        } else {
            s.equals(&outputs[0].datum_type, u8::datum_type())?;
        }
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let scale = target
            .outlet_fact(mapping[&node.inputs[1]])?
            .konst
            .as_ref()
            .ok_or("y_scale must be a const")?
            .as_slice::<f32>()?[0]
            .recip();
        let zero_point = if self.optional_zero_point_input.is_some() {
            target
                .outlet_fact(mapping[&node.inputs[2]])?
                .konst
                .as_ref()
                .ok_or("y_zero_point must be a const")?
                .clone()
        } else {
            rctensor0(0u8)
        };
        let op: Box<dyn TypedOp> = if zero_point.datum_type() == u8::datum_type() {
            Box::new(quantize_linear_u8(scale, zero_point.as_slice::<u8>()?[0]))
        } else {
            Box::new(quantize_linear_i8(scale, zero_point.as_slice::<i8>()?[0]))
        };
        target.wire_node(&*node.name, op, &[mapping[&node.inputs[0]]])
    }

    inference_op_as_op!();
}


#[derive(Debug, Clone, new, Default)]
pub struct DequantizeLinear {
    optional_zero_point_input: Option<usize>,
}

impl Op for DequantizeLinear {
    fn name(&self) -> Cow<str> {
        "onnx.DequantizeLinear".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for DequantizeLinear {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (x, y_scale, x_zero_point) = if self.optional_zero_point_input.is_some() {
            args_3!(inputs)
        } else {
            let (x, y_scale) = args_2!(inputs);
            (x, y_scale, rctensor0(0u8))
        };
        let x_scale = y_scale.as_slice::<f32>()?[0];
        let x_zero_point = x_zero_point.cast_to::<i32>()?.as_slice::<i32>()?[0];
        let x = x.cast_to::<i32>()?;
        let tensor = x
            .to_array_view::<i32>()?
            .map(|&x| ((x - x_zero_point) as f32) * x_scale)
            .into_arc_tensor();
        Ok(tvec!(tensor))
    }
}

impl InferenceRulesOp for DequantizeLinear {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(&inputs, 2 + self.optional_zero_point_input.is_some() as usize)?;
        check_output_arity(&outputs, 1)?;
        //         s.equals(&inputs[1].rank, 0)?; broken in Onnx test suite
        s.equals(&inputs[1].datum_type, f32::datum_type())?;
        s.equals(&outputs[0].datum_type, f32::datum_type())?;
        if self.optional_zero_point_input.is_some() {
            s.equals(&inputs[0].datum_type, &inputs[2].datum_type)?;
            //            s.equals(&inputs[2].rank, 0)?; // broken in Onnx test suite
        }
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let scale = target
            .outlet_fact(mapping[&node.inputs[1]])?
            .konst
            .as_ref()
            .ok_or("y_scale must be a const")?
            .as_slice::<f32>()?[0];
        let zero_point = if self.optional_zero_point_input.is_some() {
            target
                .outlet_fact(mapping[&node.inputs[2]])?
                .konst
                .as_ref()
                .ok_or("y_zero_point must be a const")?
                .clone()
        } else {
            rctensor0(0u8)
        };
        let op: Box<dyn TypedOp> = if zero_point.datum_type() == u8::datum_type() {
            Box::new(DequantizeLinearF32::new(scale, zero_point.as_slice::<u8>()?[0] as i32))
        } else if zero_point.datum_type() == i8::datum_type() {
            Box::new(DequantizeLinearF32::new(scale, zero_point.as_slice::<i8>()?[0] as i32))
        } else {
            Box::new(DequantizeLinearF32::new(scale, zero_point.as_slice::<i32>()?[0] as i32))
        };
        target.wire_node(&*node.name, op, &[mapping[&node.inputs[0]]])
    }

    inference_op_as_op!();
}
