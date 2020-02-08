use ndarray::*;

use crate::broadcast::multi_broadcast;
use crate::internal::*;

use super::binary::commute;

bin_to_super_type!(and, And, flip: commute,
     [bool, u8, i8, i16, i32, i64] => |c, &a, &b| *c = (a as i64 != 0 && b as i64 != 0) as _);
bin_to_super_type!(or, Or, flip: commute,
     [bool, u8, i8, i16, i32, i64] => |c, &a, &b| *c = (a as i64 != 0 || b as i64 != 0) as _);
bin_to_super_type!(xor, Xor, flip: commute, [bool] => |c, &a, &b| *c = a ^ b);
bin_to_bool!(equals, Equals, flip: commute,
     [bool, u8, i8, i16, i32, i64, f32, f64, TDim] => |c, a, b | *c = a == b
);

bin_to_bool!(lesser, Lesser, [bool, u8, i8, i16, i32, i64, f32, f64] => |c, &a, &b | *c = a < b);
bin_to_bool!(lesser_equal, LesserEqual, [bool, u8, i8, i16, i32, i64, f32, f64] => |c, &a, &b | *c = a <= b);
bin_to_bool!(greater, Greatser, [bool, u8, i8, i16, i32, i64, f32, f64] => |c, &a, &b | *c = a > b);
bin_to_bool!(greater_equal, GreaterEqual, [bool, u8, i8, i16, i32, i64, f32, f64] => |c, &a, &b | *c = a >= b);

element_wise!(not, Not, [bool] => |_, vs| {
    vs.iter_mut().for_each(|a| *a = !*a);
    Ok(())
});

#[derive(Debug, Clone, new, Default)]
pub struct Iff;

impl Iff {
    pub fn eval_t<T: Datum>(
        shape: &[usize],
        cond: &ArrayViewD<bool>,
        t: Arc<Tensor>,
        f: Arc<Tensor>,
    ) -> TractResult<Tensor> {
        let mut result = unsafe { Tensor::uninitialized::<T>(shape)? };
        Zip::from(result.to_array_view_mut::<T>()?)
            .and_broadcast(cond)
            .and_broadcast(t.to_array_view::<T>()?)
            .and_broadcast(f.to_array_view::<T>()?)
            .apply(|r, c, t, f| *r = if *c { t.clone() } else { f.clone() });
        Ok(result)
    }
}

impl Op for Iff {
    fn name(&self) -> Cow<str> {
        "Iff".into()
    }
    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Iff {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (cond, t, f) = args_3!(inputs);
        let shape: TVec<usize> = multi_broadcast(&[cond.shape(), t.shape(), f.shape()])
            .ok_or_else(|| {
                format!(
                    "Incompatible shapes {:?}, {:?} and {:?}",
                    cond.shape(),
                    t.shape(),
                    f.shape()
                )
            })?;
        let cond = cond.to_array_view::<bool>()?;
        let c = dispatch_datum!(Self::eval_t(t.datum_type())(&*shape, &cond, t, f))?;
        Ok(tvec!(c.into_arc_tensor()))
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
