#![allow(clippy::clone_on_copy)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::blocks_in_conditions)]

use super::array::MultiBroadcastTo;
use super::binary::TypedBinOp;
use crate::internal::*;
use crate::ops::quant::scale_by;
use num_traits::bounds::Bounded;
use num_traits::int::PrimInt;
use num_traits::{Float, Zero};
use tract_data::internal::ClampCast;
use tract_data::itertools::Itertools;
pub use tract_data::prelude::round_ties_to_even;
use tract_linalg::{ScaleShiftAndRound, Scaler};
use tract_num_traits::AsPrimitive;

#[cfg(feature = "complex")]
mod complex;
#[cfg(feature = "complex")]
pub use complex::{ComplexToInnerDim, InnerDimToComplex};

bin_to_super_type!(add, Add,
                   linalg: Add,
                   neutral_element: 0,
                   validation: Validation::Rounding,
                   q: [i8, u8, i32, i32] => add_quant;
                   q_op_on_f32: |a: f32, b: f32| -> f32 {a+b},
                   [f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64, TDim] => |c, a, b| *c = a.clone() + b);

fn add_quant<T>(c: &mut T, a: &T, b: &T, zp: i32, _: f32)
where
    T: PrimInt + Bounded + AsPrimitive<i64> + Datum,
    i64: AsPrimitive<T>,
{
    *c = (a.as_() + b.as_() - zp as i64).clamp_cast()
}

bin_to_super_type!(sub, Sub,
                   linalg:Sub,
                   is_commutative: false,
                   neutral_element: 0,
                   q: [i8, u8, i32, i32] => sub_quant;
                   q_op_on_f32: |a: f32, b: f32| -> f32 {a-b},
                   [f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64, TDim] => |c, a, b| *c = a.clone() - b);

bin_to_super_type!(subf, SubF,
                   linalg:SubF,
                   is_commutative: false,
                   neutral_element: 0,
                   q: [i8, u8, i32, i32] => subf_quant;
                   q_op_on_f32: |a: f32, b: f32| -> f32 {b - a},
                   [f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64, TDim] => |c, a, b| *c = b.clone() - a);

fn sub_quant<T>(c: &mut T, a: &T, b: &T, zp: i32, _: f32)
where
    T: PrimInt + Bounded + AsPrimitive<i16> + Datum,
    i16: AsPrimitive<T>,
{
    *c = (a.as_() - b.as_() + zp as i16).clamp_cast()
}

fn subf_quant<T>(c: &mut T, a: &T, b: &T, zp: i32, _: f32)
where
    T: PrimInt + Bounded + AsPrimitive<i16> + Datum,
    i16: AsPrimitive<T>,
{
    *c = (b.as_() - a.as_() + zp as i16).clamp_cast()
}

bin_to_super_type!(mul, Mul,
                   cost: |dt| tvec!((Cost::FMA(dt), 1)),
                   declutter: declutter_mul,
                   eval_override: |a:TValue, b: TValue, c_dt: DatumType| -> TractResult<Tensor> {
                    // we apply only if type is QU8 zp_scale datum type
                    if let (DatumType::QU8(QParams::ZpScale {zero_point: a_zp, scale: a_scale}),
                            DatumType::QU8(QParams::ZpScale {zero_point: b_zp, scale: b_scale}),
                            DatumType::QU8(QParams::ZpScale {zero_point: c_zp, scale: c_scale})) =
                        (a.datum_type(), b.datum_type(), c_dt)
                    {
                           let multiplier = a_scale  * b_scale * (1.0/ c_scale);
                           let a = a.to_array_view::<u8>()?;
                           let b = b.to_array_view::<u8>()?;
                           let c_shape = crate::broadcast::multi_broadcast(&[a.shape(), b.shape()]).context("no broadcast solution")?;
                           let mut c = Tensor::zero_dt(c_dt, &c_shape)?;
                           let view = c.to_array_view_mut::<u8>()?;
                           crate::ndarray::Zip::from(view)
                               .and_broadcast(a)
                               .and_broadcast(b)
                               .for_each(|c,a,b| *c = (scale_by((*a as i32 - a_zp as i32) * (*b as i32 - b_zp as i32), multiplier) + c_zp as i32).clamp_cast());
                           Ok(c)
                        } else {
                            Mul.generic_eval(a, b, c_dt)
                        }
                    },
                   linalg: Mul,
                   neutral_element: 1,
                   out_of_place: |c:&mut Tensor, a:&Tensor, b: &Tensor| -> TractResult<bool> {
                       if c.datum_type() == TDim::datum_type() &&
                           a.datum_type() == TDim::datum_type() && b.datum_type() == TDim::datum_type() {
                               let a = a.to_array_view::<TDim>()?;
                               let b = b.cast_to::<i32>()?;
                               let b = b.to_array_view::<i32>()?;
                               let c = c.to_array_view_mut::<TDim>()?;
                               crate::ndarray::Zip::from(c).and_broadcast(a).and_broadcast(b).for_each(|c,a,b| *c = a.clone() * *b);
                               Ok(true)
                           }
                       else {
                           match c.datum_type() {
                               DatumType::QI8(params) => {
                                   let (zp, scale) = params.zp_scale();
                                   let a = a.to_array_view::<i8>()?;
                                   let b = b.to_array_view::<i8>()?;
                                   let c = c.to_array_view_mut::<i8>()?;
                                   crate::ndarray::Zip::from(c)
                                       .and_broadcast(a)
                                       .and_broadcast(b)
                                       .for_each(|c,a,b| *c = (scale_by((*a as i16 - zp as i16) * (*b as i16 - zp as i16), scale) + zp as i16).clamp_cast());
                                   Ok(true)
                               }
                               DatumType::QU8(params) => {
                                   let (zp, scale) = params.zp_scale();
                                   let a = a.to_array_view::<u8>()?;
                                   let b = b.to_array_view::<u8>()?;
                                   let c = c.to_array_view_mut::<u8>()?;
                                   crate::ndarray::Zip::from(c)
                                       .and_broadcast(a)
                                       .and_broadcast(b)
                                       .for_each(|c,a,b| *c = (scale_by((*a as i32 - zp as i32) * (*b as i32 - zp as i32), scale) + zp as i32).clamp_cast());
                                   Ok(true)
                               }
                               _ => Ok(false)
                           }
                       }
                   },
                   q: [i8, u8, i32] => |c, a, b, zp, scale| {
                    *c = (scale_by((a.clone() as i32 - zp as i32) * (*b as i32 - zp as i32) , scale) + zp as i32).clamp_cast()
                   };
                   q_op_on_f32: |a: f32, b: f32| a * b,
[f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64, TDim] => |c, a, b| *c = a.clone() * b
);

bin_to_super_type!(div, Div,
cost: |dt| tvec!((Cost::Div(dt), 1)),
declutter: declutter_div,
eval_override: |a:TValue, b: TValue, c_dt: DatumType| -> TractResult<Tensor> {
    if
        a.datum_type() == TDim::datum_type() && b.datum_type() == TDim::datum_type() {
            let a = a.to_array_view::<TDim>()?;
            let b = b.to_array_view::<TDim>()?;
            let c_shape = crate::broadcast::multi_broadcast(&[a.shape(), b.shape()]).context("no broadcast solution")?;
            unsafe {
                let a = a.broadcast(&*c_shape).unwrap();
                let b = b.broadcast(&*c_shape).unwrap();
                let mut c = Tensor::uninitialized_dt(DatumType::TDim, &c_shape)?;
                let mut view = c.to_array_view_mut::<TDim>()?;
                for coords in crate::ndarray::indices(&*c_shape) {
                    let (p, q) = a[&coords].maybe_div(&b[&coords])?;
                    view[&coords] = p/q;
                }
                Ok(c)
            }
        } else if let (DatumType::QU8(QParams::ZpScale {zero_point: a_zp, scale: a_scale}),
                       DatumType::QU8(QParams::ZpScale {zero_point: b_zp, scale: b_scale}),
                       DatumType::QU8(QParams::ZpScale {zero_point: c_zp, scale: c_scale})) =
                (a.datum_type(), b.datum_type(), c_dt) {

               let multiplier = a_scale / (b_scale * c_scale);
                let a = a.to_array_view::<u8>()?;
                let b = b.to_array_view::<u8>()?;
                let c_shape = crate::broadcast::multi_broadcast(&[a.shape(), b.shape()]).context("no broadcast solution")?;
                let mut c = Tensor::zero_dt(c_dt, &c_shape)?;
                let view = c.to_array_view_mut::<u8>()?;
                crate::ndarray::Zip::from(view)
                    .and_broadcast(a)
                    .and_broadcast(b)
                    // maintain division in f32 before rescale to maintain high accuracy
                    .for_each(|c,a,b| *c = (
                            scale_by(
                                (*a as i32 - a_zp as i32) as f32 / (*b as i32 - b_zp as i32) as f32, multiplier
                            ) as i32 + c_zp as i32
                        ).clamp_cast());
                Ok(c)
        } else {
            Div.generic_eval(a, b, c_dt)
        }
},
is_commutative: false,
neutral_element: 1,
out_of_place: |c:&mut Tensor, a:&Tensor, b: &Tensor| -> TractResult<bool> {
    if c.datum_type() == TDim::datum_type() &&
        a.datum_type() == TDim::datum_type() && b.datum_type() == TDim::datum_type() {
            let a = a.to_array_view::<TDim>()?;
            let b = b.cast_to::<i32>()?;
            let b = b.to_array_view::<i32>()?;
            let c = c.to_array_view_mut::<TDim>()?;
            crate::ndarray::Zip::from(c).and_broadcast(a).and_broadcast(b).for_each(|c,a,b| *c = a.clone() / *b);
            Ok(true)
        } else if c.datum_type().is_quantized() || b.datum_type().is_quantized() || a.datum_type().is_quantized() {
            let a_f32 = a.cast_to::<f32>()?;
            let a_f32 = a_f32.to_array_view::<f32>()?;
            let b_f32 = b.cast_to::<f32>()?;
            let b_f32 = b_f32.to_array_view::<f32>()?;
            let c_f32 = &a_f32 / &b_f32;
            *c = c_f32.into_tensor().cast_to_dt(c.datum_type())?.into_owned();
            Ok(true)
        } else {
            Ok(false)
        }
},
q_op_on_f32: |a: f32, b: f32| a / b,
[f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64] => |c, a, b| *c = a.clone() / b
);

bin_to_super_type!(rem, Rem,
                                      eval_override: |a:TValue, b: TValue, c_dt: DatumType| -> TractResult<Tensor> {
                                          if
                                              a.datum_type() == TDim::datum_type() && b.datum_type() == TDim::datum_type() {
                                                  let a = a.to_array_view::<TDim>()?;
                                                  let b = b.cast_to::<i32>()?;
                                                  let b = b.to_array_view::<i32>()?;
                                                  let c_shape = crate::broadcast::multi_broadcast(&[a.shape(), b.shape()]).context("no broadcast solution")?;
                                                  unsafe {
                                                      let mut c = Tensor::uninitialized_dt(DatumType::TDim, &c_shape)?;
                                                      let view = c.to_array_view_mut::<TDim>()?;
                                                      crate::ndarray::Zip::from(view).and_broadcast(a).and_broadcast(b).for_each(|c,a,b| *c = a.clone() % *b);
                                                      Ok(c)
                                                  }
                                              } else {
                                                  Rem.generic_eval(a,b, c_dt)
                                              }
                                      },
                                      out_of_place: |c:&mut Tensor, a:&Tensor, b: &Tensor| -> TractResult<bool> {
                                          if c.datum_type() == TDim::datum_type() &&
                                              a.datum_type() == TDim::datum_type() && b.datum_type() == TDim::datum_type() {
                                                  let a = a.to_array_view::<TDim>()?;
                                                  let b = b.cast_to::<i32>()?;
                                                  let b = b.to_array_view::<i32>()?;
                                                  let c = c.to_array_view_mut::<TDim>()?;
                                                  crate::ndarray::Zip::from(c).and_broadcast(a).and_broadcast(b).for_each(|c,a,b| *c = a.clone() % *b);
                                                  Ok(true)
                                              } else {
                                                  Ok(false)
                                              }
                                      },
                                      [f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64] => |c, a, b| *c = a.clone() % b);

bin_to_super_type!(min, Min, linalg:Min,
                   q: [i8, u8, i32] => |c, a, b, _, _| *c = if a < b { *a } else { *b };
                   q_op_on_f32: |a: f32, b: f32| a.min(b),
                   [f16, f32, f64] => |c,a,b| *c = a.min(*b),
                   [TDim] => |c,a,b| *c = a.clone().mini(b.clone()),
                   [i8, i16, i32, i64, u8, u16, u32, u64] => |c, a, b| *c = *a.min(b));

bin_to_super_type!(max, Max,
                   eval_override: |a:TValue, b: TValue, c_dt: DatumType| -> TractResult<Tensor> {
                   // Attempt to optimize relu case
                    if let (DatumType::QU8(QParams::ZpScale {zero_point: a_zp, scale: a_scale}),
                            DatumType::QU8(QParams::ZpScale {zero_point: b_zp, scale: b_scale}),
                            DatumType::QU8(QParams::ZpScale {zero_point: c_zp, scale: c_scale})) =
                        (a.datum_type(), b.datum_type(), c_dt)
                    {
                        if a.is_uniform() || b.is_uniform() {
                            // select e between a and b as uniform if exist
                            // and d remaining a or b
                            let (d, d_zp, d_scale, e, e_zp, e_scale) = if a.is_uniform() && !b.is_uniform() {
                                (&b, &b_zp, &b_scale, &a, &a_zp, &a_scale)
                            } else {
                                (&a, &a_zp, &a_scale, &b, &b_zp, &b_scale)
                            };
                            if e.is_uniform() { // may be relu or any scalar
                                let e = e.cast_to::<u8>()?.as_slice::<u8>()?[0];
                                let e_val_as_d_aligned: i32 = scale_by(e as i32 - e_zp, e_scale / d_scale);
                                let multiplier = d_scale  * (1.0/ c_scale);
                                let d = d.to_array_view::<u8>()?;
                                let mut c = Tensor::zero_dt(c_dt, d.shape())?;
                                let view = c.to_array_view_mut::<u8>()?;
                                crate::ndarray::Zip::from(view)
                                    .and_broadcast(d)
                                    .for_each(|c,d| {
                                        let d_min_zp = *d as i32 - *d_zp as i32;
                                        let c_val: i32 = if d_min_zp < e_val_as_d_aligned {
                                            e_val_as_d_aligned
                                        } else {
                                            d_min_zp
                                        };
                                        *c = (scale_by(c_val, multiplier) + c_zp as i32).clamp_cast();
                                    });
                                return Ok(c)
                            }
                        }
                    }
                    Max.generic_eval(a, b, c_dt)
                   },
                   linalg:Max,
                   q: [i8, u8, i32] => |c, a, b, _, _| *c = if a < b { *b } else { *a };
                   q_op_on_f32: |a: f32, b: f32| -> f32 {a.max(b)},
                   [f16, f32, f64] => |c,a,b| *c = a.max(*b),
                   [TDim] => |c,a,b| *c = a.clone().maxi(b.clone()),
                   [i8, i16, i32, i64, u8, u16, u32, u64] => |c, a, b| *c = *a.max(b));

bin_to_super_type!(pow, Pow,
                   declutter: declutter_pow,
                   is_commutative: false,
                   neutral_element: 1,
                   q_op_on_f32: |a: f32, b: f32| -> f32 {a.powf(b)},
                   [f16, f32, f64] => |c,a,b| *c = a.powf(*b),
                   [i32, i64] => |c,a,b| *c = a.pow(*b as u32));

bin_to_super_type!(shift_left, ShiftLeft,
                   is_commutative: false,
                   [i8, i16, i32, i64, u8, u16, u32, u64] => |c, a, b| *c = *a << *b);
bin_to_super_type!(shift_right, ShiftRight,
                   is_commutative: false,
                   [i8, i16, i32, i64, u8, u16, u32, u64] => |c, a, b| *c = *a >> *b);

fn declutter_mul(
    _op: &Mul,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    if node.inputs[0] == node.inputs[1] && !node.outputs[0].fact.datum_type.is_quantized() {
        return Ok(Some(TypedModelPatch::replace_single_op(
            model,
            node,
            &node.inputs[0..1],
            square(),
        )?));
    }

    if let Some(uniform) = crate::ops::binary::one_input_is_uniform(model, node)? {
        let var_fact = model.outlet_fact(uniform.var)?;
        if uniform.uni.cast_to_scalar::<f64>()? == 0.0 {
            let shapes =
                model.node_input_facts(node.id)?.iter().map(|f| &f.shape).collect::<TVec<_>>();
            let shape: ShapeFact =
                crate::broadcast::multi_broadcast(&shapes).context("Failed to broadcast")?.into();
            return Ok(Some(TypedModelPatch::rewire(
                model,
                &[],
                &[node.id.into()],
                &|patch, _| {
                    let scalar = patch.add_const(
                        format!("{}.zero", node.name),
                        if uniform.uni.datum_type().is_quantized() {
                            let output_dt = node.outputs[0].fact.datum_type;
                            Arc::new(uniform.uni.clone().cast_to_dt(output_dt)?.into_owned())
                        } else {
                            uniform.uni.clone()
                        },
                    )?;
                    let op = MultiBroadcastTo::new(shape.clone());
                    patch.wire_node(&node.name, op, &[scalar])
                },
            )?));
        }
        let dt = uniform.uni.datum_type();
        if !dt.is_quantized() {
            // avoid cast potential with Q tensor
            let integer = uniform.uni.cast_to_scalar::<i64>()?;
            if tensor0(integer)
                .cast_to_dt(uniform.uni.datum_type())?
                .close_enough(&uniform.uni, false)
                .is_ok()
                && uniform.uni.cast_to_scalar::<i64>()?.count_ones() == 1
                && dt.is_integer()
            {
                let shift = integer.trailing_zeros();
                return Ok(Some(TypedModelPatch::rewire(
                    model,
                    &[uniform.var],
                    &[node.id.into()],
                    &|patch, taps| {
                        let shift = patch.add_const(
                            format!("{}.shift", node.name),
                            tensor0(shift)
                                .cast_to_dt(dt)?
                                .into_owned()
                                .broadcast_into_rank(var_fact.rank())?,
                        )?;
                        patch.wire_node(&node.name, shift_left(), &[taps[0], shift])
                    },
                )?));
            }
        }
    }
    if let Some(patch) = declutter_mul_const_mul_const(model, node)? {
        return Ok(Some(patch));
    }
    Ok(None)
}

fn declutter_mul_const_mul_const(
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let input_facts = model.node_input_facts(node.id)?;
    let Some(const_slot) = input_facts.iter().position(|f| f.konst.is_some()) else {
        return Ok(None);
    };
    let prec = model.node(node.inputs[1 - const_slot].node);
    let Some(prec_mul) = prec.op_as::<TypedBinOp>() else {
        return Ok(None);
    };
    if prec.outputs[0].successors.len() > 1 {
        return Ok(None);
    };
    if !prec_mul.0.is::<Mul>() {
        return Ok(None);
    }
    let prec_input_facts = model.node_input_facts(prec.id)?;
    let Some(prec_const_slot) = prec_input_facts.iter().position(|f| f.konst.is_some()) else {
        return Ok(None);
    };

    let const_fact = model.outlet_fact(node.inputs[const_slot])?;
    let prec_const_fact = model.outlet_fact(prec.inputs[prec_const_slot])?;
    // todo: extend to anything broadcast compatible
    if !const_fact.shape.volume().is_one() && !prec_const_fact.shape.volume().is_one() {
        return Ok(None);
    }
    if !const_fact.datum_type.is_float() {
        return Ok(None);
    }
    let result = mul()
        .eval(tvec!(
            const_fact.konst.clone().unwrap().into_tvalue(),
            prec_const_fact.konst.clone().unwrap().into_tvalue()
        ))?
        .remove(0)
        .into_arc_tensor();
    let mut patch = TypedModelPatch::default();
    let konst = patch.add_const(&prec.name, result)?;
    let input_tap = patch.tap_model(model, prec.inputs[1 - prec_const_slot])?;
    let wire = patch.wire_node(&node.name, mul(), &[konst, input_tap])?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(Some(patch))
}

fn declutter_div(
    _op: &Div,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    if let &[p, q] = &*model.node_input_facts(node.id)? {
        let dt = q.datum_type;
        if let Some(q) = &q.uniform {
            if let Ok(integer) = q.cast_to_scalar::<i64>() {
                if tensor0(integer).cast_to_dt(dt)?.close_enough(q, false).is_ok()
                    && dt.is_integer()
                    && q.cast_to_scalar::<i64>()?.count_ones() == 1
                {
                    let shift = integer.trailing_zeros();
                    return Ok(Some(TypedModelPatch::rewire(
                        model,
                        &[node.inputs[0]],
                        &[node.id.into()],
                        &|patch, taps| {
                            let shift = patch.add_const(
                                format!("{}.shift", node.name),
                                tensor0(shift)
                                    .cast_to_dt(dt)?
                                    .into_owned()
                                    .broadcast_into_rank(p.rank())?,
                            )?;
                            patch.wire_node(&node.name, shift_right(), &[taps[0], shift])
                        },
                    )?));
                }
            }
        }
        if dt.is_float() {
            return Ok(Some(TypedModelPatch::rewire(
                model,
                &node.inputs,
                &[node.id.into()],
                &|patch, taps| {
                    let q =
                        patch.wire_node(format!("{}-recip", node.name), recip(), &[taps[1]])?[0];
                    patch.wire_node(&node.name, mul(), &[taps[0], q])
                },
            )?));
        }
    }
    Ok(None)
}

fn declutter_pow(
    _op: &Pow,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let b = model.outlet_fact(node.inputs[1])?;
    if let Some(b) = &b.uniform {
        let b = b.cast_to_scalar::<f32>()?;
        if b == 2.0 {
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &[node.inputs[0]],
                square(),
            )?));
        } else if b == 0.5 {
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &[node.inputs[0]],
                sqrt(),
            )?));
        }
    }
    Ok(None)
}

element_wise!(abs, Abs, [i8, i16, i32, i64, f16, f32, i32] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.abs());
    Ok(())
};
q: [i8, u8, i32, i32] => f32::abs;
operating_datum_type: |dt| if dt == TDim::datum_type() { i64::datum_type() } else { dt }
);

element_wise!(exp, Exp, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.exp());
    Ok(())
};
q: [i8, u8, i32, i32] => f32::exp;
validation: Validation::Rounding
);

element_wise!(ln, Ln, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.ln());
    Ok(())
};
q: [i8, u8, i32, i32] => f32::ln;
validation: Validation::Rounding
);

element_wise!(square, Square, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.powi(2));
    Ok(())
};
q: [i8, u8, i32, i32] => |f : f32| f.powi(2);
validation: Validation::Rounding
);

element_wise!(sqrt, Sqrt, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.sqrt());
    Ok(())
};
q: [i8, u8, i32, i32] => f32::sqrt;
validation: Validation::Rounding
);

element_wise!(recip, Recip, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.recip());
    Ok(())
};
q: [i8, u8, i32, i32] => f32::recip;
cost: |dt| {tvec!((Cost::Div(dt), 1))};
declutter: declutter_recip;
validation: Validation::Rounding
);

fn declutter_recip(model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
    use super::element_wise::*;
    if let Some(prec) = model.single_prec(node.id)? {
        if let Some(ew) = prec.op_as::<ElementWiseOp>() {
            let repl = if ew.0.is::<Sqrt>() {
                Some(rsqrt())
            } else if ew.0.is::<Rsqrt>() {
                Some(sqrt())
            } else {
                None
            };
            if let Some(repl) = repl {
                let mut patch = TypedModelPatch::default();
                let mut wire = patch.tap_model(model, prec.inputs[0])?;
                wire = patch.wire_node(&node.name, repl, &[wire])?[0];
                patch.shunt_outside(model, node.id.into(), wire)?;
                return Ok(Some(patch));
            }
        }
    }
    Ok(None)
}

element_wise!(rsqrt, Rsqrt, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.sqrt().recip());
    Ok(())
};
q: [i8, u8, i32] => |x : f32| x.sqrt().recip();
validation: Validation::Rounding
);

element_wise!(ceil, Ceil, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.ceil());
    Ok(())
}, [i8, i16,i32, i64, u8, u16, u32, u64, TDim] => |_, _| Ok(());
q: [i8, u8, i32] => f32::recip);

element_wise!(floor, Floor, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.floor());
    Ok(())
}, [i8, i16,i32, i64, u8, u16, u32, u64, TDim] => |_, _| Ok(());
q: [i8, u8, i32] => f32::floor);

element_wise!(round, Round, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.round());
    Ok(())
}, [i8, i16,i32, i64, u8, u16, u32, u64, TDim] => |_, _| Ok(());
q: [i8, u8, i32] => f32::round);

element_wise!(q_scale, QScale{scaler: Scaler},[i32] => |op, xs| {
    xs.iter_mut().for_each(|x| *x = x.q_scale(op.scaler));
    Ok(())
});

element_wise!(round_half_to_even, RoundHalfToEven,
[f32] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = round_ties_to_even(*x));
    Ok(())
},
[f16] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = f16::from_f32(round_ties_to_even(x.to_f32())));
    Ok(())
};
q: [i8, u8, i32] => round_ties_to_even);

element_wise!(cos, Cos, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.cos());
    Ok(())
};
q: [i8, u8, i32] => f32::cos);

element_wise!(sin, Sin, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.sin());
    Ok(())
};
q: [i8, u8, i32] => f32::sin);

element_wise!(tan, Tan, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.tan());
    Ok(())
};
q: [i8, u8, i32] => f32::tan);

element_wise!(acos, Acos, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.acos());
    Ok(())
};
q: [i8, u8, i32] => f32::acos);

element_wise!(asin, Asin, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.asin());
    Ok(())
};
q: [i8, u8, i32] => f32::asin);

element_wise!(atan, Atan, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.atan());
    Ok(())
};
q: [i8, u8, i32] => f32::atan);

element_wise!(cosh, Cosh, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.cosh());
    Ok(())
};
q: [i8, u8, i32] => f32::cosh);

element_wise!(sinh, Sinh, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.sinh());
    Ok(())
};
q: [i8, u8, i32] => f32::sinh);

element_wise!(tanh, Tanh,
 [f16] => |_, xs| { (tract_linalg::ops().tanh_f16)().run(xs) },
 [f32] => |_, xs| { (tract_linalg::ops().tanh_f32)().run(xs) },
 [f64] => |_, xs| { xs.iter_mut().for_each(|x| *x = x.tanh()); Ok(()) };
 q: [i8, u8, i32] => f32::tanh;
 cost: |dt| {tvec!((Cost::FMA(dt), 11), (Cost::Div(dt), 1))}
);

element_wise!(erf, Erf,
 [f32] => |_, xs| { (tract_linalg::ops().erf_f32)().run(xs) },
 [f16] => |_, xs| {
     let mut f32s = xs.iter().map(|x| x.to_f32()).collect_vec();
     (tract_linalg::ops().erf_f32)().run(&mut f32s)?;
     xs.iter_mut().zip(f32s.into_iter()).for_each(|(x, f)| *x = f16::from_f32(f));
     Ok(())
};
 cost: |dt| {tvec!((Cost::FMA(dt), 11), (Cost::Div(dt), 1))}
);

element_wise!(acosh, Acosh, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.acosh());
    Ok(())
};
q: [i8, u8, i32] => f32::acosh);
element_wise!(asinh, Asinh, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.asinh());
    Ok(())
};
q: [i8, u8, i32] => f32::asinh);
element_wise!(atanh, Atanh, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.atanh());
    Ok(())
};
q: [i8, u8, i32] => f32::atanh);

element_wise!(neg, Neg, [i8, i16, i32, i64, f16, f32, f64, TDim] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = -x.clone());
    Ok(())
};
q: [i8, u8, i32] => |x: f32| -x);

element_wise!(sign, Sign, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = if x.is_zero() { *x } else { x.signum() });
    Ok(())
};
q: [i8, u8, i32] => f32::signum);

#[cfg(test)]
mod tests {
    use crate::ops::binary::TypedBinOp;

    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_mul() {
        let a = arr2(&[[1., 2.], [3., 4.]]);
        let b = arr2(&[[1., 0.], [0., 0.]]);
        assert_eq!(a * b, arr2(&[[1., 0.], [0., 0.]]));
    }

    #[test]
    fn dot() {
        let a = arr2(&[[1., 2.], [3., 4.]]);
        let b = arr2(&[[1., 0.], [0., 0.]]);
        assert_eq!(a.dot(&b), arr2(&[[1., 0.], [3., 0.]]));
    }

    #[test]
    fn mul_as_shift_left() -> TractResult<()> {
        let mut model = TypedModel::default();
        let x = model.add_source("x", i32::fact([2usize, 2]))?;
        let a = model.add_const("a", tensor0(4i32).broadcast_into_rank(2)?.into_arc_tensor())?;
        let y = model.wire_node("y", mul(), &[x, a])?[0];
        model.set_output_outlets(&[y])?;
        let result = SimplePlan::new(&model)?.run(tvec!(tensor2(&[[1, 2], [3, 4]]).into()))?;
        assert_eq!(*result[0], tensor2(&[[4, 8], [12, 16]]));
        let decluttered = model.into_decluttered()?;
        let result =
            SimplePlan::new(&decluttered)?.run(tvec!(tensor2(&[[1, 2], [3, 4]]).into()))?;
        assert_eq!(*result[0], tensor2(&[[4, 8], [12, 16]]));
        let op = decluttered
            .node(decluttered.output_outlets()?[0].node)
            .op()
            .downcast_ref::<TypedBinOp>()
            .unwrap();
        assert!(op.0.downcast_ref::<ShiftLeft>().is_some());
        Ok(())
    }

    #[test]
    fn div_as_shift() -> TractResult<()> {
        let mut model = TypedModel::default();
        let x = model.add_source("a", i32::fact([2usize, 2]))?;
        let s = model.add_const("shift", tensor2(&[[4]]))?;
        let y = model.wire_node("c", div(), [x, s].as_ref())?[0];
        model.set_output_outlets(&[y])?;
        let result = SimplePlan::new(&model)?.run(tvec!(tensor2(&[[16, 32], [64, 68]]).into()))?;
        assert_eq!(*result[0], tensor2(&[[4, 8], [16, 17]]));
        let decluttered = model.into_decluttered()?;
        let result =
            SimplePlan::new(&decluttered)?.run(tvec!(tensor2(&[[16, 32], [64, 68]]).into()))?;
        assert_eq!(*result[0], tensor2(&[[4, 8], [16, 17]]));
        let op = decluttered
            .node(decluttered.output_outlets()?[0].node)
            .op()
            .downcast_ref::<TypedBinOp>()
            .unwrap();
        assert!(op.0.downcast_ref::<ShiftRight>().is_some());
        Ok(())
    }
}
