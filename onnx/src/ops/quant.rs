use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use tract_hir::internal::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("QuantizeLinear", quantize_linear);
    reg.insert("DequantizeLinear", dequantize_linear);
}

fn quantize_linear(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let op = QuantizeLinear::new(Some(2).filter(|_| node.input.len() == 3));
    Ok((expand(op), vec![]))
}

fn dequantize_linear(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let op = DequantizeLinear::new(Some(2).filter(|_| node.input.len() == 3));
    Ok((expand(op), vec![]))
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct QuantizeLinear {
    optional_zero_point_input: Option<usize>,
}

tract_linalg::impl_dyn_hash!(QuantizeLinear);

impl Expansion for QuantizeLinear {
    fn name(&self) -> Cow<str> {
        "QuantizeLinear".into()
    }

    op_onnx!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(&inputs, 2 + self.optional_zero_point_input.is_some() as usize)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[1].datum_type, f32::datum_type())?;
        if self.optional_zero_point_input.is_some() {
            s.equals(&outputs[0].datum_type, &inputs[2].datum_type)?;
        } else {
            s.equals(&outputs[0].datum_type, u8::datum_type())?;
        }
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let scale = target
            .outlet_fact(inputs[1])?
            .konst
            .as_ref()
            .ok_or("y_scale must be a const")?
            .as_slice::<f32>()?[0] // bug in onnx test suite
            .recip();
        let (dt, zero_point) = if let Some(input) = self.optional_zero_point_input {
            let tensor = target
                .outlet_fact(inputs[input])?
                .konst
                .as_ref()
                .ok_or("y_zero_point must be a const")?;
            (tensor.datum_type(), tensor.cast_to_scalar::<i32>()?)
        } else {
            (DatumType::I8, 0i32)
        };
        tract_hir::ops::quant::QuantizeLinear::new(scale, zero_point, dt).wire(
            prefix,
            target,
            &inputs[0..1],
        )
    }
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct DequantizeLinear {
    optional_zero_point_input: Option<usize>,
}

tract_linalg::impl_dyn_hash!(DequantizeLinear);

impl Expansion for DequantizeLinear {
    fn name(&self) -> Cow<str> {
        "DequantizeLinear".into()
    }

    op_onnx!();

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

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let scale = target
            .outlet_fact(inputs[1])?
            .konst
            .as_ref()
            .ok_or("y_scale must be a const")?
            .as_slice::<f32>()?[0];
        let zero_point = if let Some(input) = self.optional_zero_point_input {
            target
                .outlet_fact(inputs[input])?
                .konst
                .as_ref()
                .ok_or("y_zero_point must be a const")?
                .cast_to_scalar::<i32>()?
        } else {
            0
        };
        tract_hir::ops::quant::DequantizeLinear::new(scale, zero_point).wire(
            prefix,
            target,
            &inputs[0..1],
        )
    }
}
