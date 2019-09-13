use crate::internal::*;
use downcast_rs::Downcast;
use std::fmt;

pub trait UnaryMiniOp: fmt::Debug + objekt::Clone + Send + Sync + 'static + Downcast {
    fn name(&self) -> &'static str;
    fn eval_in_place(&self, t: &mut Tensor) -> TractResult<()>;
}
clone_trait_object!(UnaryMiniOp);
downcast_rs::impl_downcast!(UnaryMiniOp);

#[derive(Debug, Clone)]
pub struct UnaryOp(pub Box<dyn UnaryMiniOp>);

impl Op for UnaryOp {
    fn name(&self) -> Cow<str> {
        format!("{}", self.0.name()).into()
    }

    fn axes_info(&self, model: &TypedModel, node: &TypedNode) -> TractResult<AxesInfo> {
        let a = model.outlet_fact(node.inputs[0])?;
        Ok((0..a.shape.rank()).into_iter().map(|axis| AxisInfo::simple(axis)).collect())
    }

    canonic!();
    op_as_typed_op!();
}

impl StatelessOp for UnaryOp {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut t = args_1!(inputs).into_tensor();
        self.0.eval_in_place(&mut t)?;
        Ok(tvec!(t.into_arc_tensor()))
    }
}

impl InferenceRulesOp for UnaryOp {
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

impl TypedOp for UnaryOp {
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

    typed_op_as_op!();
}

#[macro_export]
macro_rules! unary {
    ($func:ident, $Op:ident $({$($var: ident : $var_typ: path),*})?, $( [$($typ:ident),*] => $f:expr),*) => {
        #[derive(Debug, Clone)]
        pub struct $Op { $( $(pub $var: $var_typ),* )? }
        impl $crate::ops::unary::UnaryMiniOp for $Op {
            fn name(&self) -> &'static str {
                stringify!($Op)
            }
            fn eval_in_place(&self, t: &mut Tensor) -> TractResult<()> {
                $(
                    $(if t.datum_type() == $typ::datum_type() {
                        let t: &mut[$typ] = t.as_slice_mut::<$typ>()?;
                        let f: fn(&Self, &mut[$typ]) = $f;
                        f(self, t);
                        return Ok(())
                    }
                    )*
                )*
                bail!("{} does not support {:?}", self.name(), t.datum_type());
            }
        }
        pub fn $func($( $($var: $var_typ),* )?) -> $crate::ops::unary::UnaryOp {
            $crate::ops::unary::UnaryOp(Box::new($Op { $( $($var),* )? } ))
        }
    }
}
