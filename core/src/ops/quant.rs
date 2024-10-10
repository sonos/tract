#![allow(clippy::unnecessary_cast)]

use crate::internal::*;
use crate::ops::element_wise::ElementWiseOp;
use crate::ops::math::QScale;
use num_traits::AsPrimitive;
use tract_linalg::lut::Lut;
use tract_linalg::mmm::RoundingPolicy;
use tract_linalg::Scaler;

use super::binary::TypedBinOp;
use super::math::round_ties_to_even;

pub fn quantize_linear_f32_u8(x: f32, scale: f32, zero_point: i32) -> u8 {
    (((x * scale).round() as i32) + zero_point)
        .clamp(u8::MIN as i32, u8::MAX as i32) as u8
}

pub fn quantize_linear_f32_i8(x: f32, scale: f32, zero_point: i32) -> i8 {
    (((x * scale).round() as i32) + zero_point)
        .clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

element_wise_oop!(quantize_linear_u8,
 QuantizeLinearU8 {
     scale: f32,
     zero_point: u8
 },
 [f16] => u8 |op, xs, ys| {
     xs.iter().zip(ys.iter_mut()).for_each(|(x,y)|
                                           *y = quantize_linear_f32_u8(x.to_f32(), op.scale, op.zero_point as i32)
                                          );
     Ok(())
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

#[derive(Clone, Debug, new)]
pub struct DequantizeLinearF32 {
    pub scale: f32,
    pub zero_point: i32,
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

    op_as_typed_op!();
}

impl EvalOp for DequantizeLinearF32 {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let output = match inputs[0].datum_type() {
            DatumType::I8 => self.eval_t::<i8>(&inputs[0])?,
            DatumType::I32 => self.eval_t::<i32>(&inputs[0])?,
            DatumType::U8 => self.eval_t::<u8>(&inputs[0])?,
            dt => bail!("Unsupported type {:?}", dt),
        };
        Ok(tvec!(output.into_tvalue()))
    }
}

impl TypedOp for DequantizeLinearF32 {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        fact.datum_type = f32::datum_type();
        Ok(tvec!(fact))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
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
                } else {
                    op.0.downcast_ref::<QuantizeLinearI8>()
                        .map(|mop| (mop.scale, mop.zero_point as i32, i8::datum_type()))
                }
            } else {
                None
            };
            if let Some((scale, zero_point, dt)) = q_params {
                // first, try Op::quantize() on all ops in the chain
                let mut patch = TypedModelPatch::default();
                let mut wire: OutletId = patch.tap_model(model, dequant.inputs[0])?;
                let mut next = model.single_succ(dequant.id)?.unwrap();
                loop {
                    if let Some(op) = next
                        .op
                        .quantize(model, dequant, dt, scale, zero_point)
                        .with_context(|| format!("Quantizing {next}"))?
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
                    let mut wire = adhoc_model.add_source("ad-hoc", dt.fact([256]))?;
                    let mut next = model.single_succ(dequant.id)?.unwrap();
                    let mut name = None;
                    // plug in dequant
                    wire = adhoc_model.wire_node(
                        &*dequant.name,
                        dequant.op.clone(),
                        [wire].as_ref(),
                    )?[0];
                    while next.id != quant.id {
                        name.get_or_insert(&*next.name);
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
                    let output =
                        SimplePlan::new(adhoc_model)?.run(tvec!(input.into_tvalue()))?.remove(0);
                    let table: &[u8] = match dt {
                        DatumType::I8 => unsafe { std::mem::transmute::<&[i8], &[u8]>(output.as_slice::<i8>()?) },
                        DatumType::U8 => output.as_slice::<u8>()?,
                        _ => unreachable!(),
                    };
                    let op = lookup_table((tract_linalg::ops().lut_u8)(table));
                    let mut patch = TypedModelPatch::default();
                    let mut wire: OutletId = patch.tap_model(model, dequant.inputs[0])?;

                    wire = patch.wire_node(name.unwrap_or(&*dequant.name), op, [wire].as_ref())?[0];
                    patch.shunt_outside(model, OutletId::new(quant.id, 0), wire)?;
                    return Ok(Some(patch));
                }
            }
            let (input_facts, output_facts) = model.node_facts(quant.id)?;
            let invariants = quant
                .op
                .axes_mapping(&input_facts, &output_facts)
                .with_context(|| format!("Querying invariants for {quant}"))?;
            if invariants.is_element_wise_unary() {
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

#[derive(Debug, Clone, Hash)]
pub struct Scale;

impl crate::ops::binary::BinMiniOp for Scale {
    fn name(&self) -> &'static str {
        "Scale"
    }
    fn result_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
        if !a.is_float() {
            bail!("Scale left operand must be float, got {:?}", a);
        }
        Ok(b)
    }

    fn operating_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
        if !a.is_float() {
            bail!("Scale left operand must be float, got {:?}", a);
        }
        Ok(b)
    }

    fn eval_out_of_place(&self, c: &mut Tensor, a: &Tensor, b: &Tensor) -> TractResult<()> {
        let a = a.cast_to::<f32>()?;
        let a = a.to_array_view::<f32>()?;
        unsafe fn eval_out_of_place_t<T: Datum + AsPrimitive<f32>>(
            c: &mut Tensor,
            a: &ndarray::ArrayViewD<f32>,
            b: &Tensor,
        ) where
            f32: AsPrimitive<T>,
        {
            let b = b.to_array_view_unchecked::<T>();
            let mut c = c.to_array_view_mut_unchecked::<T>();
            ndarray::Zip::from(&mut c)
                .and_broadcast(a)
                .and_broadcast(b)
                .for_each(|c, a, b| *c = scale_by(*b, *a))
        }
        unsafe { dispatch_numbers!(eval_out_of_place_t(b.datum_type())(c, &a, b)) }
        Ok(())
    }

    fn eval_in_a(&self, a: &mut Tensor, b: &Tensor) -> TractResult<()> {
        let a = a.to_array_view_mut::<f32>()?;
        let b = b.to_array_view::<f32>()?;
        ndarray::Zip::from(a).and_broadcast(b).for_each(|a, b| *a = scale_by(*b, *a));
        Ok(())
    }

    fn is_commutative(&self) -> bool {
        false
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let a = model.outlet_fact(node.inputs[0])?;
        if let Some(a) = &a.uniform {
            if a.cast_to_scalar::<f32>()? == 1. {
                return Ok(Some(TypedModelPatch::rewire(
                    model,
                    &node.inputs[1..2],
                    &[node.id.into()],
                    &|_p, x| Ok(x.into()),
                )?));
            } else if node.outputs[0].fact.datum_type == DatumType::I32 {
                let factor = a.cast_to_scalar::<f32>()?;
                let scaler = Scaler::new(factor, RoundingPolicy::Even);

                let op = ElementWiseOp(Box::new(QScale { scaler }), None);
                let patch =
                    TypedModelPatch::replace_single_op(model, node, &node.inputs[1..2], op)?;

                return Ok(Some(patch));
            }
        }
        Ok(None)
    }
}

#[inline]
pub(crate) fn scale_by<T: Datum + AsPrimitive<f32>>(b: T, a: f32) -> T
where
    f32: AsPrimitive<T>,
{
    let b = b.as_();
    (round_ties_to_even(b.abs() * a) * b.signum()).as_()
}

pub fn scale() -> TypedBinOp {
    TypedBinOp(Box::new(Scale), None)
}

/// Offsets i8 integers as u8 integers.
pub(crate) fn offset_i8_as_u8_elementwise(x: i8) -> u8 {
    (x as u8).wrapping_add(128)
}

#[derive(Debug, Clone)]
pub struct OffsetI8asU8 {}
impl ElementWiseMiniOp for OffsetI8asU8 {
    fn name(&self) -> String {
        format!("{}{}", self.prefix(), stringify!(OffsetI8asU8))
    }
    fn output_type(&self, input_type: DatumType) -> Option<DatumType> {
        Some(if let DatumType::QI8(qp) = input_type {
            let (zp, scale) = qp.zp_scale();
            DatumType::QU8(QParams::ZpScale { zero_point: zp + 128, scale })
        } else if input_type == DatumType::I8 {
            DatumType::U8
        } else {
            input_type
        })
    }
    fn eval_out_of_place(&self, t: &Tensor, out_dt: Option<DatumType>) -> TractResult<Tensor> {
        let output_type = out_dt.unwrap_or(self.output_type(t.datum_type()).unwrap());
        let mut dst = unsafe { Tensor::uninitialized_dt(output_type, t.shape())? };
        if t.datum_type().unquantized() == i8::datum_type() {
            t.as_slice::<i8>()?
                .iter()
                .zip(dst.as_slice_mut::<u8>()?.iter_mut())
                .for_each(|(x, y)| *y = offset_i8_as_u8_elementwise(*x));
            return Ok(dst);
        }

        bail!("{} does not support {:?}", self.name(), t.datum_type());
    }
}

pub fn offset_i8_as_u8() -> ElementWiseOp {
    ElementWiseOp(Box::new(OffsetI8asU8 {}), None)
}

/// Offsets u8 integers as i8 integers.
pub(crate) fn offset_u8_as_i8_elementwise(x: u8) -> i8 {
    x.wrapping_sub(128) as i8
}

#[derive(Debug, Clone)]
pub struct OffsetU8asI8 {}
impl ElementWiseMiniOp for OffsetU8asI8 {
    fn name(&self) -> String {
        format!("{}{}", self.prefix(), stringify!(OffsetU8asI8))
    }
    fn output_type(&self, input_type: DatumType) -> Option<DatumType> {
        Some(if let DatumType::QU8(qp) = input_type {
            let (zp, scale) = qp.zp_scale();
            DatumType::QI8(QParams::ZpScale { zero_point: zp - 128, scale })
        } else if input_type == DatumType::U8 {
            DatumType::I8
        } else {
            input_type
        })
    }
    fn eval_out_of_place(&self, t: &Tensor, out_dt: Option<DatumType>) -> TractResult<Tensor> {
        let output_type = out_dt.unwrap_or(self.output_type(t.datum_type()).unwrap());
        let mut dst = unsafe { Tensor::uninitialized_dt(output_type, t.shape())? };
        if t.datum_type().unquantized() == u8::datum_type() {
            t.as_slice::<u8>()?
                .iter()
                .zip(dst.as_slice_mut::<i8>()?.iter_mut())
                .for_each(|(x, y)| *y = offset_u8_as_i8_elementwise(*x));
            return Ok(dst);
        }

        bail!("{} does not support {:?}", self.name(), t.datum_type());
    }
}
pub fn offset_u8_as_i8() -> ElementWiseOp {
    ElementWiseOp(Box::new(OffsetU8asI8 {}), None)
}

#[cfg(test)]
pub mod scale {
    use crate::internal::*;
    use crate::ops::einsum::EinSum;
    use crate::ops::math::round_ties_to_even;
    use proptest::prelude::*;

    fn test_scale(a: i8, b: i8, scale: f32) {
        let expected = (((a as i32) * (b as i32)) as f32) / scale;
        let expected = round_ties_to_even(expected.abs()) * expected.signum();
        let expected = (expected as i32).clamp(-128, 127);
        let expected = tensor2(&[[expected as i8]]);

        let input = tvec!(tensor2(&[[b]]).into_tvalue());
        let mut model = TypedModel::default();
        let a = model.add_const("a", tensor2(&[[a]])).unwrap();
        let b = model.add_source("b", i8::fact([1, 1])).unwrap();
        let bias = model.add_const("bias", tensor0(0i32)).unwrap();
        let a0 = model.add_const("a0", tensor0(0i8)).unwrap();
        let a_scale = model.add_const("a_scale", tensor0(1f32)).unwrap();
        let b0 = model.add_const("b0", tensor0(0i8)).unwrap();
        let b_scale = model.add_const("b_scale", tensor0(1f32)).unwrap();
        let c0 = model.add_const("c0", tensor0(0i8)).unwrap();
        let c_scale = model.add_const("c_scale", tensor0(scale)).unwrap();
        let op = EinSum {
            axes: "mk,kn,,,,,,,->mn".parse().unwrap(),
            operating_dt: i32::datum_type(),
            q_params: Some(i8::datum_type()),
        };
        let output = model
            .wire_node("mmm", op, &[a, b, bias, a0, a_scale, b0, b_scale, c0, c_scale])
            .unwrap();
        model.set_output_outlets(&output).unwrap();

        let plain = model.clone().into_runnable().unwrap().run(input.clone()).unwrap();
        assert_eq!(*plain[0], expected);

        let optim = model.into_optimized().unwrap().into_runnable().unwrap().run(input).unwrap();
        assert_eq!(*optim[0], expected);
    }

    proptest! {
        #[test]
        fn prop(a in any::<i8>(), b in any::<i8>(), scale in 0.00001f32..1000.) {
            test_scale(a, b, scale);
        }
    }

    #[test]
    fn t1() {
        test_scale(-117, 15, 37.753822);
    }

    #[test]
    fn t2() {
        test_scale(-4, -60, 475.21674);
    }
}
