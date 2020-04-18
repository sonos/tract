use crate::internal::*;
use num_traits::{Float, Zero};

use super::binary::*;

bin_to_super_type!(add, Add,
                   flip:commute,
                   validation: Validation::Rounding,
                   [f32, i8, i16, i32, i64, u8, u16, f16, f64, TDim] => |c, a, b| *c = a.clone() + b);
bin_to_super_type!(sub, Sub, flip:flip_sub,
                   [f32, i8, i16, i32, i64, u8, u16, f16, f64, TDim] => |c, a, b| *c = a.clone() - b);

bin_to_super_type!(mul, Mul,
                   cost: |dt| tvec!((Cost::FMA(dt), 1)),
                   declutter_unary: declutter_unary_mul,
                   flip: commute,
                   out_of_place: |c:&mut Tensor, a:&Tensor, b: &Tensor| -> TractResult<bool> {
                       if c.datum_type() == TDim::datum_type() &&
                           a.datum_type() == TDim::datum_type() && b.datum_type() == TDim::datum_type() {
                               let a = a.to_array_view::<TDim>()?;
                               let b = b.cast_to::<i32>()?;
                               let b = b.to_array_view::<i32>()?;
                               let c = c.to_array_view_mut::<TDim>()?;
                               crate::ndarray::Zip::from(c).and_broadcast(a).and_broadcast(b).apply(|c,a,b| *c = a.clone() * *b);
                               Ok(true)
                           } else {
                               Ok(false)
                           }
                   },
                   [f32, i8, i16, i32, i64, u8, u16, f16, f64] => |c, a, b| *c = a.clone() * b
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
                               crate::ndarray::Zip::from(c).and_broadcast(a).and_broadcast(b).apply(|c,a,b| *c = a.clone() / *b);
                               Ok(true)
                           } else {
                               Ok(false)
                           }
                   },
                   [f32, i8, i16, i32, i64, u8, u16, f16, f64] => |c, a, b| *c = a.clone() / b
                  );

bin_to_super_type!(rem, Rem,
                   out_of_place: |c:&mut Tensor, a:&Tensor, b: &Tensor| -> TractResult<bool> {
                       if c.datum_type() == TDim::datum_type() &&
                           a.datum_type() == TDim::datum_type() && b.datum_type() == TDim::datum_type() {
                               let a = a.to_array_view::<TDim>()?;
                               let b = b.cast_to::<i32>()?;
                               let b = b.to_array_view::<i32>()?;
                               let c = c.to_array_view_mut::<TDim>()?;
                               crate::ndarray::Zip::from(c).and_broadcast(a).and_broadcast(b).apply(|c,a,b| *c = a.clone() % *b);
                               Ok(true)
                           } else {
                               Ok(false)
                           }
                   },
                   [f32, i8, i16, i32, i64, u8, u16, f16, f64] => |c, a, b| *c = a.clone() % b);
bin_to_super_type!(min, Min, flip:commute,
                   [f32, f64] => |c,a,b| *c = a.min(*b),
                   [i8, i16, i32, i64, u8, u16] => |c, a, b| *c = *a.min(b));
bin_to_super_type!(max, Max, flip:commute,
                   [f32, f64] => |c,a,b| *c = a.max(*b),
                   [i8, i16, i32, i64, u8, u16] => |c, a, b| *c = *a.max(b));
bin_to_super_type!(pow, Pow,
                   [f32, f64] => |c,a,b| *c = a.powf(*b));

bin_to_super_type!(shift_left, ShiftLeft,
                   [i8, i16, i32, i64, u8, u16] => |c, a, b| *c = *a << *b);
bin_to_super_type!(shift_right, ShiftRight,
                   [i8, i16, i32, i64, u8, u16] => |c, a, b| *c = *a >> *b);
bin_to_super_type!(flipped_shift_left, FlippedShiftLeft,
                   [i8, i16, i32, i64, u8, u16] => |c, a, b| *c = *b << *a);
bin_to_super_type!(flipped_shift_right, FlippedShiftRight,
                   [i8, i16, i32, i64, u8, u16] => |c, a, b| *c = *b >> *a);

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
    if a.cast_to::<f64>()?.as_slice::<f64>()?.iter().all(|v| *v == 1.0) {
        return Ok(Some(TypedModelPatch::shunt_one_op(model, node)?))
    } else if a.cast_to::<f64>()?.as_slice::<f64>()?.iter().all(|v| *v == 0.0) {
        let fact = model.outlet_fact(node.inputs[0])?;
        if let Some(shape) = fact.shape.as_finite() {
            let zeros = Tensor::zero_dt(fact.datum_type, shape)?;
            return Ok(Some(TypedModelPatch::replace_single_op(model, node, &[], crate::ops::konst::Const(zeros.into()))?))
        }
    }
    Ok(None)
}

fn declutter_bin_div(
    _op: &Div,
    model: &TypedModel,
    node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
    if let Some(p) = declutter_div_as_shift(model, node)? {
        return Ok(Some(p))
    }
    let fact = model.outlet_fact(node.inputs[0])?;
    if fact.datum_type == f32::datum_type() || fact.datum_type == f64::datum_type() || fact.datum_type == f16::datum_type() {
        let mut patch = TypedModelPatch::default();
        let num = patch.tap_model(model, node.inputs[0])?;
        let denum = patch.tap_model(model, node.inputs[1])?;
        let denum = patch.wire_node(format!("{}-recip", node.name), recip(), &[denum])?[0];
        let out = patch.wire_node(&node.name, mul::bin_typed(), &[num, denum])?[0];
        patch.shunt_outside(node.id.into(), out)?;
        return Ok(Some(patch))
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

element_wise!(abs, Abs, [f16, f32, i32] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.abs());
    Ok(())
});

element_wise!(exp, Exp, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.exp());
    Ok(())
};
validation: Validation::Rounding
);

element_wise!(ln, Ln, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.ln());
    Ok(())
};
validation: Validation::Rounding
);

element_wise!(square, Square, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.powi(2));
    Ok(())
};
validation: Validation::Rounding
);

element_wise!(sqrt, Sqrt, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.sqrt());
    Ok(())
};
validation: Validation::Rounding
);

element_wise!(recip, Recip, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.recip());
    Ok(())
};
cost: |dt| {tvec!((Cost::Div(dt), 1))};
validation: Validation::Rounding
);

element_wise!(rsqrt, Rsqrt, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.sqrt().recip());
    Ok(())
};
validation: Validation::Rounding
);

element_wise!(ceil, Ceil, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.ceil());
    Ok(())
});

element_wise!(floor, Floor, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.floor());
    Ok(())
});

element_wise!(round, Round, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.round());
    Ok(())
});

element_wise!(scalar_min_max, ScalarMinMax { min: Tensor, max: Tensor },
              [f32, f64] => |m, xs| {
                  let max = m.max.cast_to_scalar()?;
                  let min = m.min.cast_to_scalar()?;
                  xs.iter_mut().for_each(|x| { *x = x.max(max).min(min) });
                  Ok(())
              },
              [i8, u8] => |m, xs| {
                  let max = m.max.cast_to_scalar()?;
                  let min = m.min.cast_to_scalar()?;
                  xs.iter_mut().for_each(|x| *x = std::cmp::max(std::cmp::min(*x, min), max));
                  Ok(())
              };
              quantize: quantize_scalar_min_max
             );

fn quantize_scalar_min_max(
    op: &ScalarMinMax,
    dt: DatumType,
    scale: f32,
    zero_point: i32,
    ) -> TractResult<Option<Box<dyn ElementWiseMiniOp>>> {
    use crate::ops::quant::*;
    let min = op.min.cast_to_scalar::<f32>()?;
    let max = op.max.cast_to_scalar::<f32>()?;
    let (min, max) = match dt {
        DatumType::U8 => (
            tensor0(quantize_linear_f32_u8(min, scale, zero_point)),
            tensor0(quantize_linear_f32_u8(max, scale, zero_point)),
            ),
        DatumType::I8 => (
            tensor0(quantize_linear_f32_i8(min, scale, zero_point)),
            tensor0(quantize_linear_f32_i8(max, scale, zero_point)),
            ),
        dt => bail!("Unsupported Q type: {:?}", dt),
    };
    Ok(Some(Box::new(ScalarMinMax { min, max })))
}

element_wise!(scalar_min, ScalarMin { min: Tensor },
              [f32, f64] => |m, xs| {
                  let min = m.min.cast_to_scalar()?;
                  xs.iter_mut().for_each(|x| *x = x.min(min));
                  Ok(())
              },
              [i8, u8] => |m, xs| {
                  let min = m.min.cast_to_scalar()?;
                  xs.iter_mut().for_each(|x| *x = std::cmp::min(*x, min));
                  Ok(())
              };
              quantize: quantize_scalar_min
             );

fn quantize_scalar_min(
    op: &ScalarMin,
    dt: DatumType,
    scale: f32,
    zero_point: i32,
    ) -> TractResult<Option<Box<dyn ElementWiseMiniOp>>> {
    use crate::ops::quant::*;
    let min = op.min.cast_to_scalar::<f32>()?;
    let min = match dt {
        DatumType::U8 => tensor0(quantize_linear_f32_u8(min, scale, zero_point)),
        DatumType::I8 => tensor0(quantize_linear_f32_i8(min, scale, zero_point)),
        dt => bail!("Unsupported Q type: {:?}", dt),
    };
    Ok(Some(Box::new(ScalarMin { min })))
}

element_wise!(scalar_max, ScalarMax { max: Tensor },
              [f32, f64] => |m, xs| {
                  let max = m.max.cast_to_scalar()?;
                  xs.iter_mut().for_each(|x| *x = x.max(max));
                  Ok(())
              },
              [i8, u8] => |m, xs| {
                  let max = m.max.cast_to_scalar()?;
                  xs.iter_mut().for_each(|x| *x = std::cmp::max(*x, max));
                  Ok(())
              };
              quantize: quantize_scalar_max
             );

fn quantize_scalar_max(
    op: &ScalarMax,
    dt: DatumType,
    scale: f32,
    zero_point: i32,
    ) -> TractResult<Option<Box<dyn ElementWiseMiniOp>>> {
    use crate::ops::quant::*;
    let max = op.max.cast_to_scalar::<f32>()?;
    let max = match dt {
        DatumType::U8 => tensor0(quantize_linear_f32_u8(max, scale, zero_point)),
        DatumType::I8 => tensor0(quantize_linear_f32_i8(max, scale, zero_point)),
        dt => bail!("Unsupported Q type: {:?}", dt),
    };
    Ok(Some(Box::new(ScalarMax { max })))
}

element_wise!(cos, Cos, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.cos());
    Ok(())
});

element_wise!(sin, Sin, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.sin());
    Ok(())
});

element_wise!(tan, Tan, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.tan());
    Ok(())
});

element_wise!(acos, Acos, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.acos());
    Ok(())
});

element_wise!(asin, Asin, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.asin());
    Ok(())
});

element_wise!(atan, Atan, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.atan());
    Ok(())
});

element_wise!(cosh, Cosh, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.cosh());
    Ok(())
});

element_wise!(sinh, Sinh, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = x.sinh());
    Ok(())
});

element_wise!(tanh, Tanh,
              [f32] => |_, xs| { (tract_linalg::ops().tanh_f32)().run(xs); Ok(()) },
              [f16, f64] => |_, xs| { xs.iter_mut().for_each(|x| *x = x.tanh()); Ok(()) };
              cost: |dt| {tvec!((Cost::FMA(dt), 11), (Cost::Div(dt), 1))}
             );

element_wise!(acosh, Acosh, [f16, f32, f64] => |_, xs| { xs.iter_mut().for_each(|x| *x = x.acosh()); Ok(()) });
element_wise!(asinh, Asinh, [f16, f32, f64] => |_, xs| { xs.iter_mut().for_each(|x| *x = x.asinh()); Ok(()) });
element_wise!(atanh, Atanh, [f16, f32, f64] => |_, xs| { xs.iter_mut().for_each(|x| *x = x.atanh()); Ok(()) });

element_wise!(neg, Neg, [i8, i16, i32, i64, f16, f32, f64, TDim] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = -x.clone());
    Ok(())
});

element_wise!(sign, Sign, [f16, f32, f64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = if x.is_zero() { *x } else { x.signum() });
    Ok(())
});

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
        let x =
            model.add_source("a", TypedFact::dt_shape(i32::datum_type(), [2usize, 2].as_ref())?)?;
        let y = model.wire_node("c", mul::unary(rctensor2(&[[4]])), [x].as_ref())?[0];
        model.set_output_outlets(&[y])?;
        let result = SimplePlan::new(&model)?.run(tvec!(tensor2(&[[1, 2], [3, 4]])))?;
        assert_eq!(result[0], rctensor2(&[[4, 8], [12, 16]]));
        let decluttered = model.declutter()?;
        let result = SimplePlan::new(&decluttered)?.run(tvec!(tensor2(&[[1, 2], [3, 4]])))?;
        assert_eq!(result[0], rctensor2(&[[4, 8], [12, 16]]));
        let op = decluttered.node_op(1).downcast_ref::<UnaryOp>().unwrap();
        assert!(op.mini_op.downcast_ref::<FlippedShiftLeft>().is_some());
        Ok(())
    }

    #[test]
    fn div_as_shift() -> TractResult<()> {
        let mut model = TypedModel::default();
        let x =
            model.add_source("a", TypedFact::dt_shape(i32::datum_type(), [2usize, 2].as_ref())?)?;
        let s = model.add_const("shift", tensor2(&[[4]]))?;
        let y = model.wire_node("c", div::bin_typed(), [x, s].as_ref())?[0];
        model.set_output_outlets(&[y])?;
        let result = SimplePlan::new(&model)?.run(tvec!(tensor2(&[[16, 32], [64, 68]])))?;
        assert_eq!(result[0], rctensor2(&[[4, 8], [16, 17]]));
        let decluttered = model.declutter()?;
        let result = SimplePlan::new(&decluttered)?.run(tvec!(tensor2(&[[16, 32], [64, 68]])))?;
        assert_eq!(result[0], rctensor2(&[[4, 8], [16, 17]]));
        let op = decluttered.node_op(1).downcast_ref::<UnaryOp>().unwrap();
        assert!(op.mini_op.downcast_ref::<FlippedShiftRight>().is_some());
        Ok(())
    }
}
