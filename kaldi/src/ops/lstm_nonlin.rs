use crate::model::NodeLine;
use crate::model::ParsingContext;
use tract_core::internal::*;

pub fn lstm_nonlin(ctx: &ParsingContext, name: &str) -> TractResult<Box<dyn InferenceOp>> {
    let node = &ctx.proto_model.config_lines.nodes.iter().find(|l| l.0 == name);
    let line = if let Some((_, NodeLine::Component(line))) = node {
        line
    } else {
        bail!("Could not find component {}", name);
    };
    let component = &ctx.proto_model.components[&line.component];
    let params: &Tensor = component.attributes.get("Params").ok_or("missing attribute Params")?;
    Ok(Box::new(LstmNonlin { peepholes_params: params.to_owned() }))
}

#[derive(Clone, Debug, new)]
pub struct LstmNonlin {
    peepholes_params: Tensor,
}

impl Op for LstmNonlin {
    fn name(&self) -> std::borrow::Cow<str> {
        "kaldi.LstmNonlin".into()
    }
    fn translation_invariants(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
    ) -> TractResult<Vec<TranslationInvariant>> {
        Ok(vec![TranslationInvariant { axis: 0, period: 1 }])
    }
    to_typed!();
}

impl StatelessOp for LstmNonlin {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        use tract_core::ndarray::*;
        use tract_core::ops::nn::sigmoid::sigmoid_f32;
        use tract_core::ops::nn::tanh::tanh_f32;

        let input = args_1!(inputs);
        let input = input.to_array_view::<f32>()?.into_dimensionality()?;
        let params = self.peepholes_params.to_array_view::<f32>()?.into_dimensionality()?;
        let t_len = input.shape()[0];
        let cell_dim = input.shape()[1] / 5;
        let mut output = Array2::<f32>::zeros((t_len, 2 * cell_dim));
        for t in 0..t_len {
            for x in 0..cell_dim {
                let i_part = input[(t, 0 * cell_dim + x)];
                let f_part = input[(t, 1 * cell_dim + x)];
                let c_part = input[(t, 2 * cell_dim + x)];
                let o_part = input[(t, 3 * cell_dim + x)];
                let c_prev = input[(t, 4 * cell_dim + x)];

                let w_ic = params[(0, x)];
                let w_fc = params[(1, x)];
                let w_oc = params[(2, x)];

                let i_t = sigmoid_f32(i_part + w_ic * c_prev);
                let f_t = sigmoid_f32(f_part + w_fc * c_prev);
                let c_t = f_t * c_prev + i_t * tanh_f32(c_part);
                let o_t = sigmoid_f32(o_part + w_oc * c_t);
                let m_t = o_t * tanh_f32(c_t);

                output[(t, x)] = c_t;
                output[(t, cell_dim + x)] = m_t;
            }
        }
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl InferenceRulesOp for LstmNonlin {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, f32::datum_type())?;
        s.equals(&outputs[0].datum_type, f32::datum_type())?;
        s.equals(&inputs[0].rank, 2)?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?;
        s.equals(5 * outputs[0].shape[1].bex(), 2 * inputs[0].shape[1].bex())?;
        Ok(())
    }

    inference_op_as_op!();
}

impl TypedOp for LstmNonlin {
    typed_op_as_op!();

    fn output_facts(
        &self,
        inputs: &[&NormalizedTensorInfo],
    ) -> TractResult<TVec<NormalizedTensorInfo>> {
        Ok(tvec!(NormalizedTensorInfo::dt_shape(
            inputs[0].datum_type,
            [inputs[0].shape.dim(0), inputs[1].shape.dim(1) * 5 / 2].as_ref()
        )?))
    }
}
