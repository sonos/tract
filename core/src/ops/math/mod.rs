use super::binary::*;
use crate::internal::*;
use crate::ops::quant::scale_by;
use num_traits::bounds::Bounded;
use num_traits::int::PrimInt;
use num_traits::{Float, Zero};
use tract_data::internal::ClampCast;
pub use tract_data::prelude::round_ties_to_even;
use tract_linalg::mmm::RoundingPolicy;
use tract_linalg::ScaleShiftAndRound;
use tract_num_traits::AsPrimitive;

bin_to_super_type!(add, Add,
    declutter_unary: declutter_unary_add,
    flip:commute,
    linalg: Add,
    validation: Validation::Rounding,
    q: [i8, u8] => add_quant;
    [f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64, TDim] => |c, a, b| *c = a.clone() + b);

fn add_quant<T>(c: &mut T, a: &T, b: &T, zp: i32, _: f32)
where
    T: PrimInt + Bounded + AsPrimitive<i16>,
    i16: AsPrimitive<T>,
{
    *c = (a.as_() + b.as_() - zp as i16).clamp_cast()
}

fn declutter_unary_add(
    _op: &Add,
    model: &TypedModel,
    node: &TypedNode,
    a: &Arc<Tensor>,
) -> TractResult<Option<TypedModelPatch>> {
    if a.as_uniform().and_then(|a| a.cast_to_scalar::<f64>().ok()).map(|n| n == 0.).unwrap_or(false)
    {
        Ok(Some(TypedModelPatch::shunt_one_op(model, node)?))
    } else {
        Ok(None)
    }
}

bin_to_super_type!(sub, Sub, 
    declutter_unary: declutter_unary_sub, flip:flip_sub, linalg:Sub,
    q: [i8, u8] => sub_quant;
    [f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64, TDim] => |c, a, b| *c = a.clone() - b);

fn sub_quant<T>(c: &mut T, a: &T, b: &T, zp: i32, _: f32)
where
    T: PrimInt + Bounded + AsPrimitive<i16>,
    i16: AsPrimitive<T>,
{
    *c = (a.as_() - b.as_() + zp as i16).clamp_cast()
}
fn declutter_unary_sub(
    _op: &Sub,
    model: &TypedModel,
    node: &TypedNode,
    a: &Arc<Tensor>,
) -> TractResult<Option<TypedModelPatch>> {
    if a.as_uniform().and_then(|a| a.cast_to_scalar::<f64>().ok()).map(|n| n == 0.).unwrap_or(false)
    {
        Ok(Some(TypedModelPatch::replace_single_op(
            model,
            node,
            &node.inputs,
            crate::ops::math::neg(),
        )?))
    } else {
        Ok(None)
    }
}

bin_to_super_type!(mul, Mul,
 cost: |dt| tvec!((Cost::FMA(dt), 1)),
 declutter_unary: declutter_unary_mul,
 flip: commute,
 linalg: Mul,
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
                    .for_each(|c,a,b| *c = scale_by((*a as i16 - zp as i16) * (*b as i16 - zp as i16) + zp as i16, scale).clamp_cast());
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
                    .for_each(|c,a,b| *c = scale_by((*a as i16 - zp as i16) * (*b as i16 - zp as i16) + zp as i16, scale).clamp_cast());
                    Ok(true)
                 }
                 _ => Ok(false)
             }
         }
 },
 [f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64] => |c, a, b| *c = a.clone() * b
);

bin_to_super_type!(div, Div,
 cost: |dt| tvec!((Cost::Div(dt), 1)),
 declutter_bin: declutter_bin_div,
 flip: flip_div,
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
 [f32, i8, i16, i32, i64, u8, u16, u32, u64, f16, f64] => |c, a, b| *c = a.clone() / b
);

bin_to_super_type!(rem, Rem,
                   eval_override: |a:Arc<Tensor>, b: Arc<Tensor>| -> TractResult<Tensor> {
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
                               Rem.generic_eval(a,b)
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

bin_to_super_type!(min, Min, flip:commute, linalg:Min,
                   [f32, f64] => |c,a,b| *c = a.min(*b),
                   [i8, i16, i32, i64, u8, u16, u32, u64] => |c, a, b| *c = *a.min(b));
bin_to_super_type!(max, Max, flip:commute, linalg:Max,
                   [f32, f64] => |c,a,b| *c = a.max(*b),
                   [i8, i16, i32, i64, u8, u16, u32, u64] => |c, a, b| *c = *a.max(b));

bin_to_super_type!(pow, Pow,
                   flip: flip_pow,
                   [f32, f64] => |c,a,b| *c = a.powf(*b),
                   [i32, i64] => |c,a,b| *c = a.pow(*b as u32));
bin_to_super_type!(flipped_pow, FlippedPow,
                   declutter_unary: declutter_unary_flipped_pow,
                   [f32, f64] => |c,a,b| *c = b.powf(*a),
                   [i32, i64] => |c,a,b| *c = b.pow(*a as u32));

bin_to_super_type!(shift_left, ShiftLeft,
                   [i8, i16, i32, i64, u8, u16, u32, u64] => |c, a, b| *c = *a << *b);
bin_to_super_type!(shift_right, ShiftRight,
                   [i8, i16, i32, i64, u8, u16, u32, u64] => |c, a, b| *c = *a >> *b);
bin_to_super_type!(flipped_shift_left, FlippedShiftLeft,
                   [i8, i16, i32, i64, u8, u16, u32, u64] => |c, a, b| *c = *b << *a);
bin_to_super_type!(flipped_shift_right, FlippedShiftRight,
                   [i8, i16, i32, i64, u8, u16, u32, u64] => |c, a, b| *c = *b >> *a);

fn flip_sub(_op: &dyn BinMiniOp, t: &Arc<Tensor>) -> Option<UnaryOp> {
    let mut t = t.clone().into_tensor();
    fn negate<T: Datum + std::ops::Neg<Output = T>>(t: &mut Tensor) {
        t.as_slice_mut::<T>().unwrap().iter_mut().for_each(|p| *p = -p.clone());
    }
    (|t: &mut Tensor| -> TractResult<()> {
        dispatch_signed!(negate(t.datum_type())(t));
        Ok(())
    })(&mut t)
    .unwrap();
    Some(UnaryOp::new(Box::new(Add), Arc::new(t)))
}

fn flip_div(_op: &dyn BinMiniOp, t: &Arc<Tensor>) -> Option<UnaryOp> {
    let mut t = t.clone().into_tensor();
    fn inverse<T: Datum + num_traits::Float>(t: &mut Tensor) {
        t.as_slice_mut::<T>().unwrap().iter_mut().for_each(|p| *p = p.recip());
    }
    (|t: &mut Tensor| -> TractResult<()> {
        dispatch_floatlike!(inverse(t.datum_type())(t));
        Ok(())
    })(&mut t)
    .unwrap();
    Some(UnaryOp::new(Box::new(Mul), Arc::new(t)))
}

fn flip_pow(_op: &dyn BinMiniOp, t: &Arc<Tensor>) -> Option<UnaryOp> {
    Some(UnaryOp::new(Box::new(FlippedPow), t.clone()))
}

fn declutter_unary_mul(
    _op: &Mul,
    model: &TypedModel,
    node: &TypedNode,
    a: &Arc<Tensor>,
) -> TractResult<Option<TypedModelPatch>> {
    if let Some(patch) = declutter_as_shift(model, node, a, Box::new(FlippedShiftLeft))? {
        Ok(Some(patch))
    } else if let Some(patch) = declutter_unary_mul_magic_values(model, node, a)? {
        Ok(Some(patch))
    } else {
        Ok(None)
    }
}

fn declutter_unary_mul_magic_values(
    model: &TypedModel,
    node: &TypedNode,
    a: &Arc<Tensor>,
) -> TractResult<Option<TypedModelPatch>> {
    if a.is_uniform()
        && a.cast_to_scalar::<f64>()? == 1.0
        && model.outlet_fact(node.inputs[0])? == &node.outputs[0].fact
    {
        return Ok(Some(TypedModelPatch::shunt_one_op(model, node)?));
    } else if a.is_uniform() && a.cast_to_scalar::<f64>()?.is_zero() {
        let mut patch = TypedModelPatch::default();
        let fact = model.outlet_fact(node.inputs[0])?;
        let zero = Tensor::zero_dt(fact.datum_type, &[])?;
        let zero = patch.add_const(format!("{}.zero", node.name), zero)?;
        let shape = crate::broadcast::multi_broadcast(&[
            fact.shape.to_vec(),
            a.shape().iter().map(|d| d.to_dim()).collect(),
        ]).with_context(|| format!("Can not broadcast {:?} and {:?}", fact.shape, a))?;
        let broadcast = crate::ops::array::MultiBroadcastTo::new(shape.into());
        let broadcast = patch.wire_node(&node.name, broadcast, &[zero])?;
        patch.shunt_outside(model, node.id.into(), broadcast[0])?;
        Ok(Some(patch))
    } else {
        Ok(None)
    }
}

fn declutter_bin_div(
    _op: &Div,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    if let Some(p) = declutter_div_as_shift(model, node)? {
        return Ok(Some(p));
    }
    let fact = model.outlet_fact(node.inputs[0])?;
    if fact.datum_type == f32::datum_type()
        || fact.datum_type == f64::datum_type()
        || fact.datum_type == f16::datum_type()
    {
        let mut patch = TypedModelPatch::default();
        let num = patch.tap_model(model, node.inputs[0])?;
        let denum = patch.tap_model(model, node.inputs[1])?;
        let denum = patch.wire_node(format!("{}-recip", node.name), recip(), &[denum])?[0];
        let out = patch.wire_node(&node.name, mul::bin_typed(), &[num, denum])?[0];
        patch.shunt_outside(model, node.id.into(), out)?;
        return Ok(Some(patch));
    }
    Ok(None)
}

fn declutter_div_as_shift(
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let a = model.node_input_facts(node.id)?[1];
    if let Some(a) = &a.konst {
        declutter_as_shift(model, node, a, Box::new(FlippedShiftRight))
    } else {
        return Ok(None);
    }
}

fn declutter_as_shift(
    model: &TypedModel,
    node: &TypedNode,
    t: &Arc<Tensor>,
    mini_op: Box<dyn BinMiniOp>,
) -> TractResult<Option<TypedModelPatch>> {
    let input = model.node_input_facts(node.id)?[0];
    if t.len() > 0 && t.datum_type().is_integer() && input.datum_type.is_integer() {
        let arg = t.cast_to::<i64>()?;
        if arg.as_slice::<i64>()?.iter().all(|i| *i > 0 && i.count_ones() == 1) {
            let mut shift = arg.into_owned();
            shift
                .as_slice_mut::<i64>()?
                .iter_mut()
                .for_each(|i| *i = (63 - i.abs().leading_zeros()) as _);
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs[0..=0],
                UnaryOp {
                    a: shift.cast_to_dt(input.datum_type)?.into_owned().into_arc_tensor(),
                    mini_op,
                },
            )?));
        }
    }
    Ok(None)
}

fn declutter_unary_flipped_pow(
    _op: &FlippedPow,
    model: &TypedModel,
    node: &TypedNode,
    a: &Arc<Tensor>,
) -> TractResult<Option<TypedModelPatch>> {
    if let Some(a) = a.as_uniform() {
        let a = a.cast_to_scalar::<f32>()?;
        if a == 1.0 {
            return Ok(Some(TypedModelPatch::shunt_one_op(model, node)?));
        } else if a == 2.0 {
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs,
                square(),
            )?));
        } else if a == 3.0 {
            return Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, cube())?));
        } else if a == 0.5 {
            return Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, sqrt())?));
        }
    }
    Ok(None)
}

element_wise!(abs, Abs, [i8, i16, i32, i64, f16, f32, i32] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.abs());
    Ok(())
};
q: [i8, u8] => f32::abs);

element_wise!(exp, Exp, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.exp());
    Ok(())
};
q: [i8, u8] => f32::exp;
validation: Validation::Rounding
);

element_wise!(ln, Ln, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.ln());
    Ok(())
};
q: [i8, u8] => f32::ln;
validation: Validation::Rounding
);

element_wise!(square, Square, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.powi(2));
    Ok(())
};
q: [i8, u8] => |f : f32| f.powi(2);
validation: Validation::Rounding
);

element_wise!(cube, Cube, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.powi(3));
    Ok(())
};
q: [i8, u8] => |f : f32| f.powi(3);
validation: Validation::Rounding
);

element_wise!(sqrt, Sqrt, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.sqrt());
    Ok(())
};
q: [i8, u8] => f32::sqrt;
validation: Validation::Rounding
);

element_wise!(recip, Recip, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.recip());
    Ok(())
};
q: [i8, u8] => f32::recip;
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
q: [i8, u8] => |x : f32| x.sqrt().recip();
validation: Validation::Rounding
);

element_wise!(ceil, Ceil, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.ceil());
    Ok(())
};
q: [i8, u8] => f32::recip);

element_wise!(floor, Floor, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.floor());
    Ok(())
};
q: [i8, u8] => f32::floor);

element_wise!(round, Round, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.round());
    Ok(())
};
q: [i8, u8] => f32::round);

element_wise!(q_scale, QScale {mult: i32, policy: RoundingPolicy, shift: usize},[i32] => |op, xs| {
    xs.iter_mut().for_each(|x| *x = x.q_scale(op.mult, op.shift, op.policy));
    Ok(())
});

element_wise!(round_half_to_even, RoundHalfToEven,[ f32] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = round_ties_to_even(*x));
    Ok(())
};
q: [i8, u8] => round_ties_to_even);

element_wise!(cos, Cos, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.cos());
    Ok(())
};
q: [i8, u8] => f32::cos);

element_wise!(sin, Sin, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.sin());
    Ok(())
};
q: [i8, u8] => f32::sin);

element_wise!(tan, Tan, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.tan());
    Ok(())
};
q: [i8, u8] => f32::tan);

element_wise!(acos, Acos, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.acos());
    Ok(())
};
q: [i8, u8] => f32::acos);

element_wise!(asin, Asin, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.asin());
    Ok(())
};
q: [i8, u8] => f32::asin);

element_wise!(atan, Atan, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.atan());
    Ok(())
};
q: [i8, u8] => f32::atan);

element_wise!(cosh, Cosh, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.cosh());
    Ok(())
};
q: [i8, u8] => f32::cosh);

element_wise!(sinh, Sinh, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.sinh());
    Ok(())
};
q: [i8, u8] => f32::sinh);

element_wise!(tanh, Tanh,
 [f32] => |_, xs| { (tract_linalg::ops().tanh_f32)().run(xs) },
 [f16, f64] => |_, xs| { xs.iter_mut().for_each(|x| *x = x.tanh()); Ok(()) };
 q: [i8, u8] => f32::tanh;
 cost: |dt| {tvec!((Cost::FMA(dt), 11), (Cost::Div(dt), 1))}
);

element_wise!(acosh, Acosh, [f16, f32, f64] => |_, xs| { 
    xs.iter_mut().for_each(|x| *x = x.acosh()); 
    Ok(()) 
};
q: [i8, u8] => f32::acosh);
element_wise!(asinh, Asinh, [f16, f32, f64] => |_, xs| { 
    xs.iter_mut().for_each(|x| *x = x.asinh()); 
    Ok(()) 
};
q: [i8, u8] => f32::asinh);
element_wise!(atanh, Atanh, [f16, f32, f64] => |_, xs| { 
    xs.iter_mut().for_each(|x| *x = x.atanh()); 
    Ok(()) 
};
q: [i8, u8] => f32::atanh);

element_wise!(neg, Neg, [i8, i16, i32, i64, f16, f32, f64, TDim] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = -x.clone());
    Ok(())
};
q: [i8, u8] => |x: f32| -x);

element_wise!(sign, Sign, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = if x.is_zero() { *x } else { x.signum() });
    Ok(())
};
q: [i8, u8] => f32::signum);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn mul() {
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
    fn mul_as_shift() -> TractResult<()> {
        let mut model = TypedModel::default();
        let x = model.add_source("a", TypedFact::dt_shape(i32::datum_type(), &[2usize, 2]))?;
        let y = model.wire_node("c", mul::unary(rctensor2(&[[4]])), [x].as_ref())?[0];
        model.set_output_outlets(&[y])?;
        let result = SimplePlan::new(&model)?.run(tvec!(tensor2(&[[1, 2], [3, 4]])))?;
        assert_eq!(result[0], rctensor2(&[[4, 8], [12, 16]]));
        let decluttered = model.declutter()?;
        let result = SimplePlan::new(&decluttered)?.run(tvec!(tensor2(&[[1, 2], [3, 4]])))?;
        assert_eq!(result[0], rctensor2(&[[4, 8], [12, 16]]));
        let op = decluttered.node(1).op().downcast_ref::<UnaryOp>().unwrap();
        assert!(op.mini_op.downcast_ref::<FlippedShiftLeft>().is_some());
        Ok(())
    }

    #[test]
    fn div_as_shift() -> TractResult<()> {
        let mut model = TypedModel::default();
        let x = model.add_source("a", TypedFact::dt_shape(i32::datum_type(), &[2usize, 2]))?;
        let s = model.add_const("shift", tensor2(&[[4]]))?;
        let y = model.wire_node("c", div::bin_typed(), [x, s].as_ref())?[0];
        model.set_output_outlets(&[y])?;
        let result = SimplePlan::new(&model)?.run(tvec!(tensor2(&[[16, 32], [64, 68]])))?;
        assert_eq!(result[0], rctensor2(&[[4, 8], [16, 17]]));
        let decluttered = model.declutter()?;
        let result = SimplePlan::new(&decluttered)?.run(tvec!(tensor2(&[[16, 32], [64, 68]])))?;
        assert_eq!(result[0], rctensor2(&[[4, 8], [16, 17]]));
        let op = decluttered.node(1).op().downcast_ref::<UnaryOp>().unwrap();
        assert!(op.mini_op.downcast_ref::<FlippedShiftRight>().is_some());
        Ok(())
    }
}
