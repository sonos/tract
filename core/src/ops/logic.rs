use ndarray::*;

use crate::broadcast::multi_broadcast;
use crate::internal::*;

use super::binary::{commute, BinMiniOp};
use super::element_wise::ElementWiseOp;

bin_to_super_type!(and, And, flip: commute,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = (a as i64 != 0 && b as i64 != 0) as _);
bin_to_super_type!(or, Or, flip: commute,
                   [bool, u8, u16, u32, u64, i8, i16, i32, i64] => |c, &a, &b| *c = (a as i64 != 0 || b as i64 != 0) as _);
bin_to_super_type!(xor, Xor, flip: commute, [bool] => |c, &a, &b| *c = a ^ b);
bin_to_bool!(equals, Equals, flip: commute,
 [bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, TDim] => |c, a, b | *c = a == b
);
bin_to_bool!(not_equals, NotEquals, flip: commute,
 [bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, TDim] => |c, a, b | *c = a != b
);

bin_to_bool!(less, Less, 
    codegen_unary: codegen_compare_to_zero,
    [bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64] => |c, &a, &b | *c = a < b);
bin_to_bool!(less_equal, LessEqual,
    codegen_unary: codegen_compare_to_zero,
    [bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64] => |c, &a, &b | *c = a <= b);
bin_to_bool!(greater, Greater,
    codegen_unary: codegen_compare_to_zero,
    [bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64] => |c, &a, &b | *c = a > b);
bin_to_bool!(greater_equal, GreaterEqual,
    codegen_unary: codegen_compare_to_zero,
    [bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64] => |c, &a, &b | *c = a >= b);

fn codegen_compare_to_zero(
    op: &dyn BinMiniOp,
    model: &TypedModel,
    node: &TypedNode,
    a: &Arc<Tensor>,
) -> TractResult<Option<TypedModelPatch>> {
    if let Some(a) = a.as_uniform() {
        if (a.datum_type().is_signed() || a.datum_type().is_float())
            && a == Tensor::zero_scalar_dt(a.datum_type())?
        {
            let op: Box<dyn ElementWiseMiniOp> = if op.is::<Less>() {
                Box::new(GreaterThanZero {})
            } else if op.is::<LessEqual>() {
                Box::new(GreaterEqualThanZero {})
            } else if op.is::<Greater>() {
                Box::new(LessThanZero {})
            } else if op.is::<GreaterEqual>() {
                Box::new(LessEqualThanZero {})
            } else {
                unreachable!();
            };
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs,
                ElementWiseOp(op),
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

impl_dyn_hash!(Iff);

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
    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for Iff {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
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
            let mut result = Tensor::uninitialized_dt(t.datum_type(), &*shape)?;
            let cond = cond.to_array_view::<bool>()?;
            dispatch_datum_by_size!(Self::eval_t(t.datum_type())(&cond, &mut result, &t, &f));
            Ok(tvec!(result.into_arc_tensor()))
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

    fn invariants(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<Invariants> {
        let a = &inputs[0];
        let b = &inputs[1];
        let c = &inputs[2];
        assert!(a.rank() == b.rank() && b.rank() == c.rank());
        let rank = a.rank();
        Ok((0..rank)
            .into_iter()
            .map(|axis| AxisInfo {
                inputs: tvec!(Some(axis), Some(axis), Some(axis)),
                outputs: tvec!(Some(axis)),
                period: 1,
                disposable: true,
            })
            .collect())
    }
}
