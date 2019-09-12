use crate::internal::*;
use downcast_rs::Downcast;
use std::fmt;

pub trait UnaMiniOp: fmt::Debug + objekt::Clone + Send + Sync + 'static + Downcast {
    fn name(&self) -> &'static str;

    fn eval_in_place(&self, t: &mut Tensor) -> TractResult<()>;
}
clone_trait_object!(UnaMiniOp);
downcast_rs::impl_downcast!(UnaMiniOp);

#[derive(Debug, Clone)]
pub struct UnaOp(pub Box<dyn UnaMiniOp>);

impl Op for UnaOp {
    fn name(&self) -> Cow<str> {
        format!("{}", self.0.name()).into()
    }

    op_as_typed_op!();
}

impl StatelessOp for UnaOp {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut t = args_1!(inputs).into_tensor();
        self.0.eval_in_place(&mut t)?;
        Ok(tvec!(t.into_arc_tensor()))
    }
}

impl InferenceRulesOp for UnaOp {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }
    to_typed!();
    inference_op_as_op!();
}

impl TypedOp for UnaOp {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(TypedTensorInfo::dt_shape(inputs[0].datum_type, inputs[0].shape.clone())?))
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let fact = target.outlet_fact(input)?.clone();
        let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
        Ok(tvec!(OutletId::new(id, 0)))
    }
}

#[macro_export]
macro_rules! unary {
    ($func:ident, $Op:ident, $( [$($typ:ident),*] => $f:expr),*) => {
        #[derive(Debug, Clone)]
        pub struct $Op;
        impl $crate::ops::unary::UnaMiniOp for $Op {
            fn name(&self) -> &'static str {
                stringify!($Op)
            }
            fn eval_in_place(&self, t: &mut Tensor) -> TractResult<()> {
                $(
                    $(if t.datum_type() == $typ::datum_type() {
                        let t: &mut[$typ] = t.as_slice_mut::<$typ>()?;
                        let f: fn(&mut[$typ]) = $f;
                        f(t);
                        return Ok(())
                    }
                    )*
                )*
                bail!("{} does not support {:?}", self.name(), t.datum_type());
            }
        }
        pub fn $func() -> $crate::ops::unary::UnaOp {
            $crate::ops::unary::UnaOp(Box::new($Op))
        }
    }
}
