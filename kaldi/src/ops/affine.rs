use tract_core::internal::*;
use tract_core::infer::*;
use tract_core::ndarray;

use crate::model::NodeLine;
use crate::model::ParsingContext;

pub fn affine_component(ctx: &ParsingContext, name: &str) -> TractResult<Box<dyn InferenceOp>> {
    let node = &ctx.proto_model.config_lines.nodes.iter().find(|l| l.0 == name);
    let line = if let Some((_, NodeLine::Component(line))) = node {
        line
    } else {
        bail!("Could not find component {}", name);
    };
    let component = &ctx.proto_model.components[&line.component];
    let (kernel_len, dilation) = line.input.as_conv_shape_dilation().unwrap_or((1, 1));
    let kernel: &Tensor =
        component.attributes.get("LinearParams").ok_or("missing attribute LinearParams")?;
    let bias = component.attributes.get("BiasParams").ok_or("missing attribute BiasParams")?;
    // O•TI -> t -> TI•O -> T•I•O = HWIO
    let o_ti = kernel.to_array_view::<f32>()?;
    let t_i_o_shape = (kernel_len, kernel.len() / kernel_len / bias.len(), bias.len());
    let t_i_o = ndarray::Array::from_shape_vec(t_i_o_shape, o_ti.t().iter().cloned().collect())?;
    Ok(Box::new(Affine {
        kernel_len,
        dilation,
        linear_params: t_i_o.into_arc_tensor(),
        bias_params: Arc::clone(bias),
    }))
}

#[derive(Clone, Debug, new)]
struct Affine {
    kernel_len: usize,
    dilation: usize,
    linear_params: Arc<Tensor>, // TIO
    bias_params: Arc<Tensor>,
}

impl Affine {
    fn as_conv(&self) -> tract_core::ops::cnn::Conv {
        use tract_core::ops::cnn::*;
        use tract_core::ops::nn::*;
        Conv::default()
            .hwc()
            .hwio()
            .bias_input(2)
            .dilations(tvec!(self.dilation))
            .kernel_shape(tvec!(self.kernel_len))
    }
}

impl Op for Affine {
    fn name(&self) -> std::borrow::Cow<str> {
        "kaldi.Affine".into()
    }

    not_a_typed_op!();
}

impl StatelessOp for Affine {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        inputs.push(Arc::clone(&self.linear_params));
        inputs.push(Arc::clone(&self.bias_params));
        self.as_conv().eval(inputs)
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
        s.equals(&inputs[0].datum_type, self.linear_params.datum_type())?;
        s.equals(&outputs[0].datum_type, self.linear_params.datum_type())?;
        s.equals(&inputs[0].rank, 2)?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&outputs[0].shape[1], &self.linear_params.shape()[2].to_dim())?;
        s.equals(&inputs[0].shape[1], &self.linear_params.shape()[1].to_dim())?;
        s.given(&inputs[0].shape, move |s, ishape| {
            let oshape = self.as_conv().output_shape(&*ishape, self.linear_params.shape());
            s.equals(&outputs[0].shape[0], &oshape[0])
        })?;
        Ok(())
    }

    fn incorporate(
        &self,
        model: &InferenceModel,
        node: &InferenceNode,
    ) -> TractResult<Option<InferenceModelPatch>> {
        let mut patch = InferenceModelPatch::default();

        let input = patch.tap_model(model, node.inputs[0])?;
        let lin = patch.add_const(format!("{}-Linear", node.name), self.linear_params.clone())?;
        let bias = patch.add_const(format!("{}-Bias", node.name), self.bias_params.clone())?;

        let wire = patch.wire_node(
            format!("{}-Conv", node.name),
            self.as_conv(),
            [input, lin.into(), bias.into()].as_ref(),
        )?[0];
        patch.shunt_outside(node.id.into(), wire)?;
        Ok(Some(patch))
    }

    inference_op_as_op!();
}
