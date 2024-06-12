use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use tract_hir::internal::*;
use tract_hir::ops::quant::*;
use tract_ndarray::ArrayViewD;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("QuantizeLinear", quantize_linear);
    reg.insert("DequantizeLinear", dequantize_linear);
    reg.insert("DynamicQuantizeLinear", dynamic_quantize_linear);
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

fn dynamic_quantize_linear(
    _ctx: &ParsingContext,
    _node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let op = DynamicQuantizeLinear::new();
    Ok((expand(op), vec![]))
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct QuantizeLinear {
    optional_zero_point_input: Option<usize>,
}



impl Expansion for QuantizeLinear {
    fn name(&self) -> Cow<str> {
        "QuantizeLinear".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(inputs, 2 + self.optional_zero_point_input.is_some() as usize)?;
        check_output_arity(outputs, 1)?;
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

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        use tract_hir::ops::quant::*;
        let scale = target
            .outlet_fact(inputs[1])?
            .konst
            .as_ref()
            .context("y_scale must be a const")?
            .as_slice::<f32>()?[0]
            .recip();
        let zero_point = if self.optional_zero_point_input.is_some() {
            target
                .outlet_fact(inputs[2])?
                .konst
                .as_ref()
                .context("y_zero_point must be a const")?
                .clone()
        } else {
            rctensor0(0u8)
        };
        let op: Box<dyn TypedOp> = if zero_point.datum_type() == u8::datum_type() {
            Box::new(quantize_linear_u8(scale, zero_point.as_slice::<u8>()?[0]))
        } else {
            Box::new(quantize_linear_i8(scale, zero_point.as_slice::<i8>()?[0]))
        };
        target.wire_node(prefix, op, &[inputs[0]])
    }
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct DequantizeLinear {
    optional_zero_point_input: Option<usize>,
}



impl Expansion for DequantizeLinear {
    fn name(&self) -> Cow<str> {
        "DequantizeLinear".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(inputs, 2 + self.optional_zero_point_input.is_some() as usize)?;
        check_output_arity(outputs, 1)?;
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
            .context("y_scale must be a const")?
            .as_slice::<f32>()?[0];
        let zero_point = if self.optional_zero_point_input.is_some() {
            target
                .outlet_fact(inputs[2])?
                .konst
                .as_ref()
                .context("y_zero_point must be a const")?
                .clone()
        } else {
            rctensor0(0u8)
        };
        let op: Box<dyn TypedOp> = if zero_point.datum_type() == u8::datum_type() {
            Box::new(DequantizeLinearF32::new(scale, zero_point.as_slice::<u8>()?[0] as i32))
        } else if zero_point.datum_type() == i8::datum_type() {
            Box::new(DequantizeLinearF32::new(scale, zero_point.as_slice::<i8>()?[0] as i32))
        } else {
            Box::new(DequantizeLinearF32::new(scale, zero_point.as_slice::<i32>()?[0]))
        };
        target.wire_node(prefix, op, &[inputs[0]])
    }
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct DynamicQuantizeLinear {}



impl Expansion for DynamicQuantizeLinear {
    fn name(&self) -> Cow<str> {
        "DynamicQuantizeLinear".into()
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(3)
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> TractResult<()> {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 3)?;
        s.equals(&inputs[0].datum_type, f32::datum_type())?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.equals(&outputs[0].datum_type, u8::datum_type())?;
        s.equals(&outputs[1].datum_type, f32::datum_type())?;
        s.equals(&outputs[1].rank, 0)?;
        s.equals(&outputs[2].datum_type, u8::datum_type())?;
        s.equals(&outputs[2].rank, 0)?;

        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let op: Box<dyn TypedOp> = Box::new(DynamicQuantizeLinearU8::new());
        target.wire_node(format!("{prefix}.dynamic_quantize"), op, &[inputs[0]])
    }
}

fn dynamic_quantize_linear_f32_u8(x: f32, scale: f32, zero_point: u8) -> u8 {
    (((x / scale).round() as i32) + zero_point as i32)
        .clamp(u8::MIN as i32, u8::MAX as i32) as u8
}

fn dynamic_quantize_linear_u8(scale: f32, zero_point: u8, xs: &[f32], ys: &mut [u8]) {
    xs.iter()
        .zip(ys.iter_mut())
        .for_each(|(x, y)| *y = dynamic_quantize_linear_f32_u8(*x, scale, zero_point));
}

fn scale_and_zero_point(v: ArrayViewD<f32>) -> (f32, u8) {
    // get the min and max of v and extend it to have zero included
    // in the interval [min, max]
    let (min, max) = v.fold((0., 0.), |(a_min, a_max), &v| {
        if v < a_min {
            (v, a_max)
        } else if v > a_max {
            (a_min, v)
        } else {
            (a_min, a_max)
        }
    });

    // quantize range
    let min_t = u8::MIN as f32;
    let max_t = u8::MAX as f32;

    let scale = (max - min) / max_t;

    let zero_point = -min / scale;
    let zero_point = zero_point.round();
    // clipping to [0, 255]
    let zero_point = zero_point.max(min_t);
    let zero_point = zero_point.min(max_t);

    let zero_point: u8 = zero_point as u8;

    (scale, zero_point)
}

#[derive(Clone, Debug, new, Hash)]
pub struct DynamicQuantizeLinearU8;

impl Op for DynamicQuantizeLinearU8 {
    fn name(&self) -> Cow<str> {
        "DynamicQuantizeLinearU8".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![])
    }

    fn validation(&self) -> Validation {
        Validation::Accurate
    }

    op_as_typed_op!();
}



impl EvalOp for DynamicQuantizeLinearU8 {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = &inputs[0];
        let input = input.cast_to::<f32>()?;
        let a_input = input.to_array_view::<f32>()?;
        let (scale, zero_point) = scale_and_zero_point(a_input);

        let mut dst = unsafe { Tensor::uninitialized_dt(u8::datum_type(), input.shape())? };
        // We cannot use quantize_linear_u8 here because it does `x * scale.recip()`
        // instead of `x / scale`. This change some number enough to be rounded to another integer.
        dynamic_quantize_linear_u8(
            scale,
            zero_point,
            input.as_slice::<f32>()?,
            dst.as_slice_mut::<u8>()?,
        );

        let quantized_tensor = dst.into_tvalue();
        let scale_tensor = tensor0(scale).into();
        let zero_point_tensor = tensor0(zero_point).into();

        Ok(tvec!(quantized_tensor, scale_tensor, zero_point_tensor))
    }
}

impl TypedOp for DynamicQuantizeLinearU8 {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut quantized_fact = inputs[0].clone();
        quantized_fact.datum_type = u8::datum_type();
        let scale_fact = f32::fact([0; 0]);
        let zero_fact = u8::fact([0; 0]);
        Ok(tvec!(quantized_fact, scale_fact, zero_fact))
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_ndarray::arr1;

    // Data for tests is from:
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#DynamicQuantizeLinear
    #[test]
    fn test_scale_and_zero_point() {
        let data: [(&[f32], f32, u8); 3] = [
            (&[0., 2., -3., -2.5, 1.34, 0.5], 0.019_607_844, 153),
            (&[-1., -2.1, -1.3, -2.5, -3.34, -4.], 0.015_686_275, 255),
            (&[1., 2.1, 1.3, 2.5, 3.34, 4., 1.5, 2.6, 3.9, 4., 3., 2.345], 0.015_686_275, 0),
        ];

        let epsilon = 0.00000001;
        for (v, scale_ok, zero_point_ok) in &data {
            let v = arr1(v).into_dyn();
            let v = v.view();
            let (scale, zero_point) = scale_and_zero_point(v);
            assert!((scale - scale_ok).abs() < epsilon);
            assert_eq!(zero_point, *zero_point_ok);
        }
    }

    #[test]
    fn test_dynamic_quantize_linear_u8() {
        let data: [(&[f32], &[u8]); 3] = [
            (&[0., 2., -3., -2.5, 1.34, 0.5], &[153, 255, 0, 26, 221, 179]),
            (&[-1., -2.1, -1.3, -2.5, -3.34, -4.], &[191, 121, 172, 96, 42, 0]),
            (
                &[1., 2.1, 1.3, 2.5, 3.34, 4., 1.5, 2.6, 3.9, 4., 3., 2.345],
                &[64, 134, 83, 159, 213, 255, 96, 166, 249, 255, 191, 149],
            ),
        ];

        for (v, quantized_ok) in &data {
            let v = arr1(v).into_dyn();
            let (scale, zero_point) = scale_and_zero_point(v.view());

            // same shape of v but with u8 type, values will be overwritten
            let mut quantized = v.mapv(|_| 0_u8);
            dynamic_quantize_linear_u8(
                scale,
                zero_point,
                v.as_slice().unwrap(),
                quantized.as_slice_mut().unwrap(),
            );
            assert_eq!(quantized.as_slice().unwrap(), *quantized_ok);
        }
    }
}
