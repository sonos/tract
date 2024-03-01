#![allow(clippy::bool_comparison)]
#![allow(clippy::unnecessary_cast)]

mod ite;
pub use ite::IfThenElse;

use ndarray::*;

use crate::broadcast::multi_broadcast;
use crate::internal::*;

use super::binary::BinMiniOp;
use super::element_wise::ElementWiseOp;

bin_to_super_type!(and, And,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = (a as i64 != 0 && b as i64 != 0) as _);
bin_to_super_type!(or, Or,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = (a as i64 != 0 || b as i64 != 0) as _);
bin_to_super_type!(xor, Xor, /*flip: commute, */ [bool] => |c, &a, &b| *c = a ^ b);
bin_to_bool!(equals, Equals,
 [bool, u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64, TDim] => |c, a, b | *c = a == b
);
bin_to_bool!(not_equals, NotEquals, /* flip: commute, */
 [bool, u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64, TDim] => |c, a, b | *c = a != b
);

bin_to_bool!(less, Less,
             codegen: codegen_compare_to_zero,
             operating_datum_type: operating_datum_type_for_cmp,
             [bool, u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64] => |c, &a, &b | *c = a < b);
bin_to_bool!(less_equal, LessEqual,
             codegen: codegen_compare_to_zero,
             operating_datum_type: operating_datum_type_for_cmp,
             [bool, u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64] => |c, &a, &b | *c = a <= b);
bin_to_bool!(greater, Greater,
             codegen: codegen_compare_to_zero,
             operating_datum_type: operating_datum_type_for_cmp,
             [bool, u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64] => |c, &a, &b | *c = a > b);
bin_to_bool!(greater_equal, GreaterEqual,
             codegen: codegen_compare_to_zero,
             operating_datum_type: operating_datum_type_for_cmp,
             [bool, u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64] => |c, &a, &b | *c = a >= b);

pub fn operating_datum_type_for_cmp(a: DatumType, b: DatumType) -> TractResult<DatumType> {
    let dt = a
        .common_super_type(b)
        .with_context(|| format_err!("No super type for {:?} and {:?}", a, b))?;
    if dt == DatumType::TDim {
        Ok(DatumType::I64)
    } else {
        Ok(dt)
    }
}

fn codegen_compare_to_zero(
    op: &dyn BinMiniOp,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let facts = model.node_input_facts(node.id)?;
    if let Some(uniform) = crate::ops::binary::one_input_is_uniform(model, node)? {
        let dt = facts[0].datum_type;
        if (dt.is_signed() || dt.is_float()) && *uniform.uni == Tensor::zero_scalar_dt(dt)? {
            let reversed = uniform.left_is_uniform;
            let mapped = || -> Box<dyn ElementWiseMiniOp> {
                macro_rules! m {
                    ($bin: ty, $same: expr, $other: expr) => {
                        if op.is::<$bin>() {
                            return if reversed { Box::new($other) } else { Box::new($same) };
                        };
                    };
                }
                m!(Less, LessThanZero {}, GreaterEqualThanZero {});
                m!(LessEqual, LessEqualThanZero {}, GreaterThanZero {});
                m!(Greater, GreaterThanZero {}, LessEqualThanZero {});
                m!(GreaterEqual, GreaterEqualThanZero {}, LessThanZero {});
                unreachable!();
            };
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &[uniform.var],
                ElementWiseOp(mapped(), None),
            )?));
        }
    }
    Ok(None)
}

element_wise_oop!(less_than_zero, LessThanZero, [f16, f32, f64, i8, i16, i32, i64] => bool |_op, xs, ys| {
    xs.iter().zip(ys.iter_mut()).for_each(|(x,y)| *y = *x < num_traits::Zero::zero());
    Ok(())
});

element_wise_oop!(less_equal_than_zero, LessEqualThanZero, [f16, f32, f64, i8, i16, i32, i64] => bool |_op, xs, ys| {
    xs.iter().zip(ys.iter_mut()).for_each(|(x,y)| *y = *x <= num_traits::Zero::zero());
    Ok(())
});

element_wise_oop!(greater_than_zero, GreaterThanZero, [f16, f32, f64, i8, i16, i32, i64] => bool |_op, xs, ys| {
    xs.iter().zip(ys.iter_mut()).for_each(|(x,y)| *y = *x > num_traits::Zero::zero());
    Ok(())
});

element_wise_oop!(greater_equal_than_zero, GreaterEqualThanZero, [f16, f32, f64, i8, i16, i32, i64] => bool |_op, xs, ys| {
    xs.iter().zip(ys.iter_mut()).for_each(|(x,y)| *y = *x >= num_traits::Zero::zero());
    Ok(())
});

element_wise!(not, Not, [bool] => |_, vs| {
    vs.iter_mut().for_each(|a| *a = !*a);
    Ok(())
});

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Iff;

impl Iff {
    pub unsafe fn eval_t<T: Datum>(
        cond: &ArrayViewD<bool>,
        out: &mut Tensor,
        t: &Tensor,
        f: &Tensor,
    ) {
        Zip::from(out.to_array_view_mut_unchecked::<T>())
            .and_broadcast(cond)
            .and_broadcast(t.to_array_view_unchecked::<T>())
            .and_broadcast(f.to_array_view_unchecked::<T>())
            .for_each(|r, c, t, f| *r = if *c { t.clone() } else { f.clone() })
    }
}

impl Op for Iff {
    fn name(&self) -> Cow<str> {
        "Iff".into()
    }
    op_as_typed_op!();
}

impl EvalOp for Iff {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (cond, t, f) = args_3!(inputs);
        anyhow::ensure!(t.datum_type() == f.datum_type());
        let shape: TVec<usize> = multi_broadcast(&[cond.shape(), t.shape(), f.shape()])
            .ok_or_else(|| {
                format_err!(
                    "Incompatible shapes {:?}, {:?} and {:?}",
                    cond.shape(),
                    t.shape(),
                    f.shape()
                )
            })?;
        unsafe {
            let mut result = Tensor::uninitialized_dt(t.datum_type(), &shape)?;
            let cond = cond.to_array_view::<bool>()?;
            dispatch_datum_by_size!(Self::eval_t(t.datum_type())(&cond, &mut result, &t, &f));
            Ok(tvec!(result.into_tvalue()))
        }
    }
}

impl TypedOp for Iff {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        anyhow::ensure!(inputs.len() == 3, "Iff expects 3 intputs.");
        if inputs[1].datum_type != inputs[2].datum_type {
            bail!("Then and else tensors type mismatch ({:?} and {:?}).", inputs[1], inputs[2]);
        }
        if inputs[0].rank() != inputs[1].rank() || inputs[0].rank() != inputs[2].rank() {
            bail!("Inconsistent ranks, {:?}", inputs);
        }
        let shape = multi_broadcast(&[
            inputs[0].shape.to_tvec(),
            inputs[1].shape.to_tvec(),
            inputs[2].shape.to_tvec(),
        ])
        .unwrap();
        Ok(tvec!(inputs[1].datum_type.fact(shape)))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }
}

bin_to_super_type!(bitand, BitAnd,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = a & b);
bin_to_super_type!(bitor, BitOr,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = a | b);
bin_to_super_type!(bitxor, BitXor,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = a ^ b);

element_wise!(bitnot, BitNot, [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |_, xs| {
    xs.iter_mut().for_each(|x| *x = !*x);
    Ok(())
});
