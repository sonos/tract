use tract_core::internal::*;
use tract_core::ndarray;

use crate::model::ParsingContext;

pub fn renorm(ctx: &ParsingContext, name: &str) -> TractResult<Box<InferenceOp>> {
    let component = &ctx.proto_model.components[name];
    let rms = *component
        .attributes
        .get("TargetRms")
        .ok_or("missing attributes TargetRms")?
        .to_scalar::<f32>()?;
    Ok(Box::new(Renorm::new(rms)))
}

#[derive(Clone, Debug, new)]
struct Renorm {
    target_rms: f32,
}

impl Op for Renorm {
    fn name(&self) -> std::borrow::Cow<str> {
        "kaldi.Renorm".into()
    }

    fn translation_invariants(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Vec<TranslationInvariant>> {
        Ok(vec![TranslationInvariant { axis: 0, period: 1 }])
    }
}

impl StatelessOp for Renorm {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let mut input: ndarray::Array2<f32> =
            input.into_tensor().into_array()?.into_dimensionality::<ndarray::Ix2>()?;
        let rms_sqrt_d = self.target_rms * (input.shape()[1] as f32).sqrt();
        input.genrows_mut().into_iter().for_each(|mut row| {
            let factor = rms_sqrt_d
                * row.iter().map(|x| x.powi(2)).sum::<f32>().sqrt().max(std::f32::EPSILON).recip();
            row.mapv_inplace(|row| row * factor)
        });
        Ok(tvec!(input.into_arc_tensor()))
    }
}

impl InferenceRulesOp for Renorm {
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

    inference_op_as_op!();
}
