use tract_core::internal::*;

use crate::model::ParsingContext;

pub fn fixed_affine_component(ctx: &ParsingContext, name: &str) -> TractResult<Box<InferenceOp>> {
    let component = &ctx.proto_model.components[name];
    Ok(Box::new(Affine {
        linear_params: Arc::clone(component.attributes.get("LinearParams").ok_or("missing attribute LinearParams")?),
        bias_params: Arc::clone(component.attributes.get("BiasParams").ok_or("missing attribute ViasParams")?),
    }))
}

#[derive(Clone, Debug, new)]
struct Affine {
    linear_params: Arc<Tensor>,
    bias_params: Arc<Tensor>,
}

impl Op for Affine {
    fn name(&self) -> std::borrow::Cow<str> {
        "kaldi.Affine".into()
    }
}

impl StatelessOp for Affine {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unimplemented!();
    }
}

impl InferenceRulesOp for Affine {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        Ok(())
    }

    inference_op_as_op!();
}
