use crate::broadcast::multi_broadcast;
use crate::internal::*;
use crate::ndarray::Zip;

#[derive(Clone, Copy, Debug, Hash)]
pub enum Comp {
    Eq,
    NE,
    LT,
    GT,
    GTE,
    LTE,
}

use tract_data::UndeterminedSymbol;
use Comp::*;

impl Op for Comp {
    fn name(&self) -> Cow<str> {
        match *self {
            Eq => "==",
            NE => "!=",
            LT => "<",
            GT => ">",
            LTE => "<=",
            GTE => ">=",
        }
        .into()
    }

    op_as_typed_op!();
}

impl Comp {
    fn eval<T: Datum + PartialOrd>(&self, a: &Tensor, b: &Tensor) -> TractResult<Tensor> {
        let a = a.to_array_view::<T>()?;
        let b = b.to_array_view::<T>()?;
        let shape = multi_broadcast(&[a.shape(), b.shape()])?;
        let mut c = unsafe { Tensor::uninitialized::<bool>(&shape)? };
        let mut view = c.to_array_view_mut::<bool>()?;
        let zipped = Zip::from(&mut view).and_broadcast(&a).and_broadcast(&b);
        match *self {
            Eq => zipped.for_each(|c, a, b| *c = a == b),
            NE => zipped.for_each(|c, a, b| *c = a != b),
            LT => zipped.for_each(|c, a, b| *c = a < b),
            GT => zipped.for_each(|c, a, b| *c = a > b),
            LTE => zipped.for_each(|c, a, b| *c = a <= b),
            GTE => zipped.for_each(|c, a, b| *c = a >= b),
        }
        Ok(c)
    }
}

impl EvalOp for Comp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        if inputs[0].datum_type() == TDim::datum_type() {
            let mut a = inputs[0].clone().into_tensor();
            let mut b = inputs[1].clone().into_tensor();
            for a in a.as_slice_mut::<TDim>()? {
                *a = a.eval(&session.resolved_symbols);
            }
            for b in b.as_slice_mut::<TDim>()? {
                *b = b.eval(&session.resolved_symbols);
            }
            if let (Ok(a), Ok(b)) = (a.cast_to::<i64>(), b.cast_to::<i64>()) {
                return Ok(tvec!(self.eval::<i64>(&a, &b)?.into_tvalue()));
            }
            let scope = a
                .as_slice::<TDim>()?
                .iter()
                .chain(b.as_slice::<TDim>().unwrap().iter())
                .find_map(|d| d.find_scope())
                .unwrap();
            let a = inputs[0].to_array_view::<TDim>()?;
            let b = inputs[0].to_array_view::<TDim>()?;
            let shape = multi_broadcast(&[a.shape(), b.shape()])?;
            let mut c = unsafe { Tensor::uninitialized::<bool>(&shape)? };
            let mut view = c.to_array_view_mut::<bool>()?;
            let a = a.broadcast(&*shape).unwrap();
            let b = b.broadcast(&*shape).unwrap();
            for ixs in tract_ndarray::indices(&*shape) {
                let (a, b) = (&a[&ixs], &b[&ixs]);
                view[&ixs] = match *self {
                    Eq => a == b,
                    NE => a != b,
                    GTE => {
                        if scope.prove_positive_or_zero(&(a.clone() - b)) {
                            true
                        } else if scope.prove_positive_or_zero(&(b.clone() - a - 1)) {
                            false
                        } else {
                            bail!(UndeterminedSymbol(a.clone() - b));
                        }
                    }
                    GT => {
                        if scope.prove_positive_or_zero(&(a.clone() - b - 1)) {
                            true
                        } else if scope.prove_positive_or_zero(&(b.clone() - a)) {
                            false
                        } else {
                            bail!(UndeterminedSymbol(a.clone() - b));
                        }
                    }
                    LTE => {
                        if scope.prove_positive_or_zero(&(b.clone() - a)) {
                            true
                        } else if scope.prove_positive_or_zero(&(a.clone() - b - 1)) {
                            false
                        } else {
                            bail!(UndeterminedSymbol(a.clone() - b));
                        }
                    }
                    LT => {
                        if scope.prove_positive_or_zero(&(b.clone() - a - 1)) {
                            true
                        } else if scope.prove_positive_or_zero(&(a.clone() - b)) {
                            false
                        } else {
                            bail!(UndeterminedSymbol(a.clone() - b));
                        }
                    }
                };
            }
            Ok(tvec!(c.into_tvalue()))
        } else {
            let t = dispatch_numbers!(Self::eval(inputs[0].datum_type())(
                self, &inputs[0], &inputs[1]
            ))?;
            Ok(tvec!(t.into_tvalue()))
        }
    }
}

impl TypedOp for Comp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let shape = multi_broadcast(&[&inputs[0].shape, &inputs[1].shape])?;
        Ok(tvec!(bool::datum_type().fact(shape)))
    }

    as_op!();
}

/*
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
             [bool, u8, u16, u32, u64, i8, i16, iua32, i64, f16, f32, f64] => |c, &a, &b | *c = a <= b);
bin_to_bool!(greater, Greater,
             codegen: codegen_compare_to_zero,
             operating_datum_type: operating_datum_type_for_cmp,
             [bool, u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64] => |c, &a, &b | *c = a > b);
bin_to_bool!(greater_equal, GreaterEqual,
             codegen: codegen_compare_to_zero,
             operating_datum_type: operating_datum_type_for_cmp,
             [bool, u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64] => |c, &a, &b | *c = a >= b);

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
*/
