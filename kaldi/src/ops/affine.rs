use tract_hir::internal::*;

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
        component.attributes.get("LinearParams").context("missing attribute LinearParams")?;
    let bias = component.attributes.get("BiasParams").context("missing attribute BiasParams")?;
    // O•TI -> t -> TI•O -> T•I•O = HWIO
    let o_ti = kernel.to_array_view::<f32>()?;
    let t_i_o_shape = (kernel_len, kernel.len() / kernel_len / bias.len(), bias.len());
    let t_i_o =
        tract_ndarray::Array::from_shape_vec(t_i_o_shape, o_ti.t().iter().cloned().collect())?;
    Ok(expand(Affine {
        kernel_len,
        dilation,
        linear_params: t_i_o.into_arc_tensor(),
        bias_params: Arc::clone(bias),
    }))
}

#[derive(Clone, Debug, new, Hash)]
struct Affine {
    kernel_len: usize,
    dilation: usize,
    linear_params: Arc<Tensor>, // TIO
    bias_params: Arc<Tensor>,
}

tract_data::impl_dyn_hash!(Affine);

impl Affine {
    fn as_conv(&self) -> tract_hir::ops::cnn::Conv {
        use tract_hir::ops::cnn::*;
        Conv::default()
            .hwc()
            .hwio()
            .bias_input(2)
            .dilations(tvec!(self.dilation))
            .kernel_shape(tvec!(self.kernel_len))
    }
}

impl Expansion for Affine {
    fn name(&self) -> std::borrow::Cow<str> {
        "Affine".into()
    }

    op_kaldi!();

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
            let oshape = self.as_conv().output_shape(&*ishape, self.linear_params.shape())?;
            s.equals(&outputs[0].shape[0], &oshape[0])
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        use tract_hir::ops::cnn::*;
        use tract_hir::ops::nn::*;
        use tract_hir::tract_core::ops::cnn::KernelFormat;
        model.wire_node(
            prefix,
            ConvUnary {
                pool_spec: PoolSpec::new(
                    DataFormat::HWC,
                    tvec!(self.kernel_len),
                    PaddingSpec::Valid,
                    Some(tvec!(self.dilation)),
                    None,
                    Some(self.bias_params.len()),
                ),
                kernel_fmt: KernelFormat::HWIO,
                kernel: self.linear_params.clone(),
                group: 1,
                bias: Some(self.bias_params.clone().into_arc_tensor()),
                q_params: None,
            },
            inputs,
        )
    }
}
