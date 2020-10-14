use crate::internal::*;
use crate::ops::element_wise::ElementWiseOp;
use ndarray::ArrayViewD;
use num_traits::AsPrimitive;
use num_traits::Zero;
use tract_linalg::lut::Lut;

#[derive(Clone, Debug, Educe)]
#[educe(Hash)]
pub struct QParams {
    pub c_datum_type: DatumType,
    pub zero_point_a: Option<Arc<Tensor>>,
    pub zero_point_b: Option<Arc<Tensor>>,
    pub zero_point_c: Option<Arc<Tensor>>,
    #[educe(Hash(method = "hash_scale"))]
    pub scale_factor: Option<f32>,
}

fn hash_scale<H: std::hash::Hasher>(it: &Option<f32>, state: &mut H) {
    Hash::hash(&it.clone().unwrap_or(1.0).to_bits(), state)
}

fn cleanup_zeropoint(zp: &Arc<Tensor>) -> Option<Arc<Tensor>> {
    match zp.datum_type() {
        DatumType::U8 => cleanup_zeropoint_t::<u8>(zp),
        DatumType::I8 => cleanup_zeropoint_t::<i8>(zp),
        _ => Some(zp.clone()),
    }
}

fn cleanup_zeropoint_t<T: Datum + Zero + Copy>(zp: &Arc<Tensor>) -> Option<Arc<Tensor>> {
    let mut zp = zp.clone();
    if zp.rank() == 1 {
        let slice = zp.as_slice::<T>().unwrap();
        if slice[1..].iter().all(|&x| x == slice[0]) {
            zp = rctensor0(slice[0]);
        }
    }
    if zp.rank() == 0 && *zp.to_scalar::<T>().unwrap() == T::zero() {
        None
    } else {
        Some(zp.into_arc_tensor())
    }
}

impl QParams {
    pub fn new(dt: DatumType) -> QParams {
        QParams {
            c_datum_type: dt,
            zero_point_a: None,
            zero_point_b: None,
            zero_point_c: None,
            scale_factor: None,
        }
    }

    pub fn with_zero_point_a(self, zero_point: &Arc<Tensor>) -> QParams {
        QParams { zero_point_a: cleanup_zeropoint(zero_point), ..self }
    }

    pub fn with_zero_point_b(self, zero_point: &Arc<Tensor>) -> QParams {
        QParams { zero_point_b: cleanup_zeropoint(zero_point), ..self }
    }

    pub fn with_zero_point_c(self, zero_point: &Arc<Tensor>) -> QParams {
        QParams { zero_point_c: cleanup_zeropoint(zero_point), ..self }
    }

    pub fn with_scale_factor(self, scale_factor: f32) -> QParams {
        QParams { scale_factor: Some(scale_factor), ..self }
    }

    pub fn set_zero_point_a(&mut self, zero_point: &Arc<Tensor>) {
        self.zero_point_a = cleanup_zeropoint(zero_point);
    }

    pub fn set_zero_point_b(&mut self, zero_point: &Arc<Tensor>) {
        self.zero_point_b = cleanup_zeropoint(zero_point);
    }

    pub fn set_zero_point_c(&mut self, zero_point: &Arc<Tensor>) {
        self.zero_point_c = cleanup_zeropoint(zero_point);
    }

    pub fn set_scale_factor(&mut self, scale_factor: f32) {
        self.scale_factor = Some(scale_factor)
    }
}

pub fn quantize_linear_f32_u8(x: f32, scale: f32, zero_point: i32) -> u8 {
    (((x * scale).round() as i32) + zero_point as i32)
        .max(u8::min_value() as i32)
        .min(u8::max_value() as i32) as u8
}

pub fn quantize_linear_f32_i8(x: f32, scale: f32, zero_point: i32) -> i8 {
    (((x * scale).round() as i32) + zero_point as i32)
        .max(i8::min_value() as i32)
        .min(i8::max_value() as i32) as i8
}

element_wise_oop!(quantize_linear_u8,
 QuantizeLinearU8 {
     #[educe(Hash(method="hash_f32"))]
     scale: f32,
     zero_point: u8
 },
 [f32,i32] => u8 |op, xs, ys| {
     xs.iter().zip(ys.iter_mut()).for_each(|(x,y)|
                                           *y = quantize_linear_f32_u8(*x as f32, op.scale, op.zero_point as i32)
                                          );
     Ok(())
 };
 info: info_quantize_linear_u8
);

fn info_quantize_linear_u8(q: &QuantizeLinearU8) -> TractResult<Vec<String>> {
    Ok(vec![format!(
        "scale: {} zero_point: {} 1/scale: {}",
        q.scale,
        q.zero_point,
        q.scale.recip()
    )])
}

element_wise_oop!(quantize_linear_i8,
 QuantizeLinearI8 {
     #[educe(Hash(method="hash_f32"))]
     scale: f32,
     zero_point: i8
 },
 [f32,i32] => i8 |op, xs, ys| {
     xs.iter().zip(ys.iter_mut()).for_each(|(x,y)|
                                           *y = quantize_linear_f32_i8(*x as f32, op.scale, op.zero_point as i32)
                                          );
     Ok(())
 };
 info: info_quantize_linear_i8
);

fn info_quantize_linear_i8(q: &QuantizeLinearI8) -> TractResult<Vec<String>> {
    Ok(vec![format!(
        "scale: {} zero_point: {} 1/scale: {}",
        q.scale,
        q.zero_point,
        q.scale.recip()
    )])
}

fn dynamic_quantize_linear_f32_u8(x: f32, scale: f32, zero_point: u8) -> u8 {
    (((x / scale).round() as i32) + zero_point as i32)
        .max(u8::min_value() as i32)
        .min(u8::max_value() as i32) as u8
}

fn dynamic_quantize_linear_u8(scale: f32, zero_point: u8, xs: &[f32], ys: &mut [u8]) {
    xs.iter()
        .zip(ys.iter_mut())
        .for_each(|(x, y)| *y = dynamic_quantize_linear_f32_u8(*x, scale, zero_point));
}

fn scale_and_zero_point<'a>(v: ArrayViewD<'a, f32>) -> (f32, u8) {
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
    let min_t = u8::min_value() as f32;
    let max_t = u8::max_value() as f32;

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

    op_core_mir!();
    op_as_typed_op!();
}

tract_linalg::impl_dyn_hash!(DynamicQuantizeLinearU8);

impl EvalOp for DynamicQuantizeLinearU8 {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = &inputs[0];
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

        let quantized_tensor = dst.into_arc_tensor();
        let scale_tensor = rctensor0(scale);
        let zero_point_tensor = rctensor0(zero_point);

        Ok(tvec!(quantized_tensor, scale_tensor, zero_point_tensor))
    }
}

impl TypedOp for DynamicQuantizeLinearU8 {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut quantized_fact = inputs[0].clone();
        quantized_fact.datum_type = u8::datum_type();
        let shape = ShapeFact::from_dims(&[])?;
        let scale_fact = TypedFact::dt_shape(f32::datum_type(), shape.clone())?;
        let zero_fact = TypedFact::dt_shape(u8::datum_type(), shape)?;
        Ok(tvec!(quantized_fact, scale_fact, zero_fact))
    }

    as_op!();
}

#[derive(Clone, Debug, new, Educe)]
#[educe(Hash)]
pub struct DequantizeLinearF32 {
    #[educe(Hash(method = "hash_f32"))]
    scale: f32,
    zero_point: i32,
}

impl DequantizeLinearF32 {
    fn eval_t<T: Datum + AsPrimitive<i32>>(&self, input: &Tensor) -> TractResult<Tensor> {
        let mut output = unsafe { Tensor::uninitialized::<f32>(input.shape())? };
        input
            .as_slice::<T>()?
            .iter()
            .zip(output.as_slice_mut::<f32>()?.iter_mut())
            .for_each(|(x, y)| *y = (x.as_() - self.zero_point) as f32 * self.scale);
        Ok(output)
    }
}

impl Op for DequantizeLinearF32 {
    fn name(&self) -> Cow<str> {
        "DequantizeLinearF32".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("scale: {} zero_point: {}", self.scale, self.zero_point)])
    }

    fn validation(&self) -> Validation {
        Validation::Accurate
    }

    op_core_mir!();
    op_as_typed_op!();
}

tract_linalg::impl_dyn_hash!(DequantizeLinearF32);

impl EvalOp for DequantizeLinearF32 {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let output = match inputs[0].datum_type() {
            DatumType::I8 => self.eval_t::<i8>(&inputs[0])?,
            DatumType::I32 => self.eval_t::<i32>(&inputs[0])?,
            DatumType::U8 => self.eval_t::<u8>(&inputs[0])?,
            dt => bail!("Unsupported type {:?}", dt),
        };
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl TypedOp for DequantizeLinearF32 {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        fact.datum_type = f32::datum_type();
        Ok(tvec!(fact))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        Invariants::new_element_wise(model, node)
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

    fn declutter(
        &self,
        model: &TypedModel,
        dequant: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut current = dequant;
        let incoming_dt = model.node_input_facts(dequant.id)?[0].datum_type;
        while let Some(quant) = model.single_succ(current.id)? {
            let q_params = if let Some(op) = quant.op_as::<ElementWiseOp>() {
                if let Some(mop) = op.0.downcast_ref::<QuantizeLinearU8>() {
                    Some((mop.scale, mop.zero_point as i32, u8::datum_type()))
                } else if let Some(mop) = op.0.downcast_ref::<QuantizeLinearI8>() {
                    Some((mop.scale, mop.zero_point as i32, i8::datum_type()))
                } else {
                    None
                }
            } else {
                None
            };
            if let Some((scale, zero_point, dt)) = q_params {
                // first, try Op::quantize() on all ops in the chain
                let mut patch = TypedModelPatch::default();
                let mut wire: OutletId = patch.tap_model(model, dequant.inputs[0])?.into();
                let mut next = model.single_succ(dequant.id)?.unwrap();
                loop {
                    if let Some(op) = next
                        .op
                        .quantize(model, dequant, dt, scale, zero_point)
                        .with_context(|| format!("Quantizing {}", next))?
                    {
                        wire = patch.wire_node(&*next.name, op, [wire].as_ref())?[0];
                    } else {
                        break;
                    }
                    if next.id == current.id {
                        patch.shunt_outside(model, OutletId::new(quant.id, 0), wire)?;
                        return Ok(Some(patch));
                    } else {
                        next = model.single_succ(next.id)?.unwrap();
                    }
                }
                // or else make a lookup table
                if incoming_dt == DatumType::I8 || incoming_dt == DatumType::U8 {
                    let mut adhoc_model = TypedModel::default();
                    let mut wire = adhoc_model
                        .add_source("ad-hoc", TypedFact::dt_shape(dt, [256].as_ref())?)?;
                    let mut next = model.single_succ(dequant.id)?.unwrap();
                    let mut name = None;
                    // plug in dequant
                    wire = adhoc_model.wire_node(
                        &*dequant.name,
                        dequant.op.clone(),
                        [wire].as_ref(),
                    )?[0];
                    while next.id != quant.id {
                        name.get_or_insert_with(|| &*next.name);
                        wire =
                            adhoc_model.wire_node(&*next.name, next.op.clone(), [wire].as_ref())?
                                [0];
                        next = model.single_succ(next.id)?.unwrap();
                    }
                    // plug in quant
                    wire =
                        adhoc_model.wire_node(&*quant.name, quant.op.clone(), [wire].as_ref())?[0];
                    adhoc_model.set_output_outlets(&[wire])?;
                    let input = (0u8..=255).collect::<Vec<u8>>();
                    let input = match dt {
                        DatumType::I8 => unsafe {
                            tensor1(std::mem::transmute::<&[u8], &[i8]>(&*input))
                        },
                        DatumType::U8 => tensor1(&input),
                        _ => unreachable!(),
                    };
                    let output = SimplePlan::new(adhoc_model)?.run(tvec!(input))?.remove(0);
                    let table: &[u8] = match dt {
                        DatumType::I8 => unsafe { std::mem::transmute(output.as_slice::<i8>()?) },
                        DatumType::U8 => output.as_slice::<u8>()?,
                        _ => unreachable!(),
                    };
                    let op = lookup_table((tract_linalg::ops().lut_u8)(table));
                    let mut patch = TypedModelPatch::default();
                    let mut wire: OutletId = patch.tap_model(model, dequant.inputs[0])?.into();
                    wire = patch.wire_node(name.unwrap_or(&*dequant.name), op, [wire].as_ref())?[0];
                    patch.shunt_outside(model, OutletId::new(quant.id, 0), wire)?;
                    return Ok(Some(patch));
                }
            }
            let invariants = quant
                .op
                .invariants(model, quant)
                .with_context(|| format!("Querying invariants for {}", quant))?;
            if invariants.element_wise() {
                current = quant;
            } else {
                break;
            }
        }
        Ok(None)
    }

    as_op!();
}

element_wise_oop!(lookup_table,
 LookupTable {
     #[educe(Hash(method="hash_lookup_table"))]
     table: Box<dyn Lut>
 },
 [i8] => i8 |op, xs, ys| {
     ys.copy_from_slice(xs);
     unsafe {
         let casted = std::slice::from_raw_parts_mut(ys.as_mut_ptr() as *mut u8, ys.len());
         op.table.run(casted);
     }
     Ok(())
 },
 [u8] => u8 |op, xs, ys| {
     ys.copy_from_slice(xs);
     op.table.run(ys);
     Ok(())
 }
);

fn hash_lookup_table<H: std::hash::Hasher>(lut: &Box<dyn Lut>, h: &mut H) {
    Hash::hash_slice(lut.table(), h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    // Data for tests is from:
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#DynamicQuantizeLinear
    #[test]
    fn test_scale_and_zero_point() {
        let data: [(&[f32], f32, u8); 3] = [
            (&[0., 2., -3., -2.5, 1.34, 0.5], 0.0196078438, 153),
            (&[-1., -2.1, -1.3, -2.5, -3.34, -4.], 0.0156862754, 255),
            (&[1., 2.1, 1.3, 2.5, 3.34, 4., 1.5, 2.6, 3.9, 4., 3., 2.345], 0.0156862754, 0),
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
            let mut quantized = v.mapv(|_| 0 as u8);
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
