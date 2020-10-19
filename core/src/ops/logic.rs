use ndarray::*;

use crate::broadcast::multi_broadcast;
use crate::internal::*;

use super::binary::commute;

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

bin_to_bool!(lesser, Lesser, [bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64] => |c, &a, &b | *c = a < b);
bin_to_bool!(lesser_equal, LesserEqual, [bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64] => |c, &a, &b | *c = a <= b);
bin_to_bool!(greater, Greater, [bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64] => |c, &a, &b | *c = a > b);
bin_to_bool!(greater_equal, GreaterEqual, [bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64] => |c, &a, &b | *c = a >= b);

element_wise!(not, Not, [bool] => |_, vs| {
    vs.iter_mut().for_each(|a| *a = !*a);
    Ok(())
});

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Iff;

tract_data::impl_dyn_hash!(Iff);

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
            .apply(|r, c, t, f| *r = if *c { t.clone() } else { f.clone() })
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
        let shape = multi_broadcast(&[
            inputs[0].shape.to_tvec(),
            inputs[1].shape.to_tvec(),
            inputs[2].shape.to_tvec(),
        ])
        .unwrap();
        Ok(tvec!(TypedFact::dt_shape(inputs[1].datum_type, &*shape)?))
    }
}
