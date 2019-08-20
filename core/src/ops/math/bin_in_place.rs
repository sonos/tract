use crate::internal::*;
use std::fmt;

clone_trait_object!(BinMiniOp);
trait BinMiniOp: fmt::Debug + objekt::Clone + Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn result_datum_type(&self, a: DatumType, _b: DatumType) -> TractResult<DatumType> {
        Ok(a)
    }
    fn eval_out_of_place(&self, c: &mut Tensor, a: &Tensor, b: &Tensor) -> TractResult<()>;
}

#[derive(Debug, Clone)]
struct InferenceBinOp(Box<dyn BinMiniOp>);

impl Op for InferenceBinOp {
    fn name(&self) -> Cow<str> {
        self.0.name().into()
    }
}

impl StatelessOp for InferenceBinOp {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (a, b) = args_2!(inputs);
        let b = b.cast_to_dt(a.datum_type())?;
        let c_shape = crate::broadcast::multi_broadcast(&[a.shape(), b.shape()])
            .ok_or("Can not compute resulting shape")?;
        let mut c = unsafe { Tensor::uninitialized_dt(a.datum_type(), &*c_shape)? };
        self.0.eval_out_of_place(&mut c, a.as_ref(), b.as_ref())?;
        Ok(tvec!(c.into_arc_tensor()))
    }
}

impl InferenceRulesOp for InferenceBinOp {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;

        s.with(&inputs[0].shape, move |s, a_shape| {
            s.with(&inputs[1].shape, move |s, b_shape| {
                if let Ok(Some(c_shape)) =
                    crate::analyser::helpers::infer_shape_broadcasting(&[&a_shape, &b_shape])
                {
                    s.equals(&outputs[0].shape, c_shape)?;
                }
                Ok(())
            })
        })?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        Ok(())
    }
    inference_op_as_op!();
}

macro_rules! bin {
    ($func:ident, $Op:ident, $( [$($typ:ident),*] => $closure:expr ),*) => {
        #[derive(Debug, Clone)]
        struct $Op;
        impl BinMiniOp for $Op {
            fn name(&self) -> &'static str {
                stringify!($Op)
            }

            fn eval_out_of_place(&self, c: &mut Tensor, a: &Tensor, b: &Tensor) -> TractResult<()> {
                $(
                    $(if c.datum_type() == $typ::datum_type() {
                        let a = a.to_array_view::<$typ>()?;
                        let b = b.to_array_view::<$typ>()?;
                        let mut c = c.to_array_view_mut::<$typ>()?;
                        ndarray::Zip::from(&mut c).and_broadcast(a).and_broadcast(b).apply($closure);
                        return Ok(())
                    }
                    )*
                )*
                bail!("{} does not support {:?}", self.name(), c.datum_type());
            }
        }

        pub fn $func() -> impl InferenceOp {
            InferenceBinOp(Box::new($Op))
        }
    };
}

bin!(add, Add, [f32, i8, i16, i32, i64, u8, u16, f16, f64, TDim] => |c, a, b| *c = a.clone() + b);
bin!(sub, Sub, [f32, i8, i16, i32, i64, u8, u16, f16, f64, TDim] => |c, a, b| *c = a.clone() - b);
bin!(mul, Mul, [f32, i8, i16, i32, i64, u8, u16, f16, f64, TDim] => |c, a, b| *c = a.clone() * b);
bin!(div, Div, [f32, i8, i16, i32, i64, u8, u16, f16, f64, TDim] => |c, a, b| *c = a.clone() / b);
bin!(rem, Rem, [f32, i8, i16, i32, i64, u8, u16, f16, f64, TDim] => |c, a, b| *c = a.clone() % b);
bin!(min, Min,
     [f32, f64] => |c,a,b| *c = a.min(*b),
     [i8, i16, i32, i64, u8, u16] => |c, a, b| *c = *a.min(b));
bin!(max, Max,
     [f32, f64] => |c,a,b| *c = a.max(*b),
     [i8, i16, i32, i64, u8, u16] => |c, a, b| *c = *a.max(b));
bin!(pow, Pow,
     [f32, f64] => |c,a,b| *c = a.powf(*b)
     );
