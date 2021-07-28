use crate::internal::*;
use crate::ops::element_wise::ElementWiseOp;
use crate::ops::math::QScale;
use num_traits::AsPrimitive;
use tract_linalg::lut::Lut;
use tract_linalg::mmm::RoundingPolicy;

use super::math::round_ties_to_even;

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

impl_dyn_hash!(DequantizeLinearF32);

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

    fn invariants(&self, inputs: &[&TypedFact], outputs: &[&TypedFact]) -> TractResult<Invariants> {
        Invariants::new_element_wise(inputs, outputs)
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
                    let mut wire =
                        adhoc_model.add_source("ad-hoc", TypedFact::dt_shape(dt, &[256]))?;
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
            let (input_facts, output_facts) = model.node_facts(quant.id)?;
            let invariants = quant
                .op
                .invariants(&input_facts, &output_facts)
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

#[derive(Debug, Clone, Hash)]
pub struct Scale;
impl_dyn_hash!(Scale);

impl crate::ops::binary::BinMiniOp for Scale {
    fn name(&self) -> &'static str {
        "Scale"
    }

    fn result_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
        if a != f32::datum_type() {
            bail!("Scale left operand must be f32, got {:?}", a);
        }
        Ok(b)
    }

    fn operating_datum_type(&self, a: DatumType, b: DatumType) -> TractResult<DatumType> {
        if a != f32::datum_type() {
            bail!("Scale left operand must be f32, got {:?}", a);
        }
        Ok(b)
    }

    fn eval_uniform_in_place(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()> {
        let a = a.to_scalar::<f32>()?;
        unsafe fn eval_in_place_t<T: Datum + AsPrimitive<f32>>(a: f32, b: &mut Tensor)
        where
            f32: AsPrimitive<T>,
        {
            b.as_slice_mut_unchecked::<T>().iter_mut().for_each(|x| *x = scale_by(*x, a));
        }
        unsafe { dispatch_numbers!(eval_in_place_t(b.datum_type())(*a, b)) }
        Ok(())
    }

    fn eval_unicast_in_place(&self, a: &Tensor, b: &mut Tensor) -> TractResult<()> {
        let a = a.to_array_view::<f32>()?;
        unsafe fn eval_in_place_t<T: Datum + AsPrimitive<f32>>(
            a: &ndarray::ArrayViewD<f32>,
            b: &mut Tensor,
        ) where
            f32: AsPrimitive<T>,
        {
            let mut b = b.to_array_view_mut_unchecked::<T>();
            ndarray::Zip::from(&mut b).and_broadcast(a).for_each(|b, a| *b = scale_by(*b, *a))
        }
        unsafe { dispatch_numbers!(eval_in_place_t(b.datum_type())(&a, b)) }
        Ok(())
    }

    fn eval_out_of_place(&self, c: &mut Tensor, a: &Tensor, b: &Tensor) -> TractResult<()> {
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

    fn declutter_unary(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        a: &Arc<Tensor>,
    ) -> TractResult<Option<TypedModelPatch>> {
        if a.is_uniform() && *a.to_scalar::<f32>()? == 1. {
            return Ok(Some(TypedModelPatch::shunt_one_op(model, node)?));
        } else if a.is_uniform() && node.outputs[0].fact.datum_type == DatumType::I32 {
            let factor = *a.to_scalar::<f32>()?;
            if factor <= 0.0 || factor >= 0.5 {
                return Ok(None);
            }
            let factor_bits = factor.to_bits();
            let current_exponent = factor_bits >> 23;
            let bumped_multi = f32::from_bits(factor_bits & 0x007fffff | 0x3f000000);
            let int_multi = (bumped_multi * (1i64 << 31) as f32).round() as i32;
            let shift = 126usize - current_exponent as usize;
            let op = ElementWiseOp(Box::new(QScale {
                mult: int_multi,
                shift,
                policy: RoundingPolicy::Even,
            }));
            let patch = TypedModelPatch::replace_single_op(model, node, &*node.inputs, op)?;

            return Ok(Some(patch));
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

pub mod scale {
    use crate::internal::*;
    use crate::ops::binary::*;

    pub fn bin_typed() -> TypedBinOp {
        TypedBinOp(Box::new(super::Scale))
    }
    pub fn unary(t: Arc<Tensor>) -> UnaryOp {
        UnaryOp::new(Box::new(super::Scale), t)
    }

    #[cfg(test)]
    mod test {
        use crate::internal::*;
        use crate::ops;
        use crate::ops::math::round_ties_to_even;
        use proptest::prelude::*;

        fn test_scale(a: i8, b: i8, scale: f32) {
            let expected = (((a as i32) * (b as i32)) as f32) / scale;
            let expected = round_ties_to_even(expected.abs()) * expected.signum();
            let expected = (expected as i32).max(-128).min(127);
            let expected = rctensor2(&[[expected as i8]]);

            let input = tvec!(tensor2(&[[b]]));
            let mut model = TypedModel::default();
            let a = model.add_const("a", tensor2(&[[a]])).unwrap();
            let b = model.add_source("b", TypedFact::dt_shape(i8::datum_type(), &[1, 1])).unwrap();
            let bias = model.add_const("bias", tensor0(0i32)).unwrap();
            let mut qp = ops::matmul::MatMulQParams::noop_static(i8::datum_type());
            qp.c_scale = tensor0(scale).into();
            let op = ops::matmul::QMatMul::new(false, false, false, i8::datum_type(), qp);
            let output = model.wire_node("mmm", op, &[a, b, bias]).unwrap();
            model.set_output_outlets(&*output).unwrap();

            let plain = model.clone().into_runnable().unwrap().run(input.clone()).unwrap();
            assert_eq!(&plain[0], &expected);

            let optim = model
                .into_optimized()
                .unwrap()
                .into_runnable()
                .unwrap()
                .run(input.clone())
                .unwrap();
            assert_eq!(&optim[0], &expected);
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
}

/// Offsets u8 integers as i8 integers.
pub(crate) fn offset_u8_as_i8_elementwise(x: u8) -> i8 {
    x.wrapping_sub(128) as i8
}

#[derive(Debug, Clone, Educe)]
#[educe(Hash)]
pub struct OffsetU8asI8 {}
impl_dyn_hash!(OffsetU8asI8);
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
    fn eval_out_of_place(&self, t: &Tensor) -> TractResult<Tensor> {
        let output_type = self.output_type(t.datum_type()).unwrap();
        let mut dst = unsafe { Tensor::uninitialized_dt(output_type, &t.shape())? };
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
    ElementWiseOp(Box::new(OffsetU8asI8 {}))
}
