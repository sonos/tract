use std::str;

use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::cnn::deconv::adjustments;
use tract_core::ops::cnn::KernelFormat;
use tract_hir::ops::cnn::PaddingSpec;
use tract_hir::ops::nn::DataFormat;
use tract_hir::{internal::*, ops::cnn::PoolSpec};

pub fn conv_transpose(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let padding_spec = super::pad(node, false)?;
    let strides = super::strides(node)?;
    let dilations = super::dilations(node)?;
    let adjustments = node.get_attr_opt_tvec::<usize>("output_padding")?;
    let output_shape = node.get_attr_opt_tvec::<usize>("output_shape")?;
    let group = node.get_attr_opt::<usize>("group")?.unwrap_or(1);
    Ok((
        expand(ConvTranspose::new(
            padding_spec,
            strides,
            dilations,
            adjustments,
            output_shape,
            group,
            node.input.len() == 3,
        )),
        vec![],
    ))
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct ConvTranspose {
    padding_spec: PaddingSpec,
    strides: Option<TVec<usize>>,
    dilations: Option<TVec<usize>>,
    adjustments: Option<TVec<usize>>,
    output_shape: Option<TVec<usize>>,
    group: usize,
    have_bias: bool,
}

impl Expansion for ConvTranspose {
    fn name(&self) -> Cow<str> {
        "ConvTranspose".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2 + self.have_bias as usize)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[0].rank, &inputs[1].rank)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?; // N
        s.equals(&inputs[0].shape[1], &inputs[1].shape[0])?; // O
        s.equals(&outputs[0].shape[1], (self.group as i64) * inputs[1].shape[1].bex())?; // I

        s.given_2(&inputs[0].shape, &inputs[1].shape, move |s, x_shape, w_shape| {
            if let (Ok(x_shape), Ok(w_shape)) = (
                x_shape.iter().map(|d| d.to_usize()).collect::<TractResult<TVec<usize>>>(),
                w_shape.iter().map(|d| d.to_usize()).collect::<TractResult<TVec<usize>>>(),
            ) {
                let y_shape = if let Some(output_shape) = &self.output_shape {
                    let mut y_shape = x_shape;
                    y_shape[1] = w_shape[1] * self.group;
                    for (ix, d) in output_shape.iter().enumerate() {
                        y_shape[ix + 2] = *d;
                    }
                    y_shape
                } else {
                    // ONNX deconv kernels are stored as gi_o_h_w (convolution are go_i_hw)
                    // so tract KernelFormat (in|out)put_channel functions do not work.
                    let ci = w_shape[0];
                    let co = w_shape[1] * self.group;
                    let pool_spec = PoolSpec::new(
                        DataFormat::NCHW,
                        w_shape[2..].into(),
                        self.padding_spec.clone(),
                        self.dilations.clone(),
                        self.strides.clone(),
                        ci,
                        co,
                    );
                    tract_core::ops::cnn::deconv::output_shape(
                        &pool_spec,
                        &x_shape,
                        &self.adjustments.clone().unwrap_or_else(|| tvec!(0; x_shape.len() - 2 )),
                    )?
                };
                let y_shape = y_shape.iter().map(|x| x.to_dim()).collect::<TVec<TDim>>();
                s.equals(&outputs[0].shape, y_shape)?;
            }
            Ok(())
        })?;
        if self.have_bias {
            s.equals(&inputs[2].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[2].rank, 1)?;
            s.equals(&inputs[2].shape[0], &outputs[0].shape[1])?;
        }
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(k) = target.outlet_fact(inputs[1])?.konst.clone() {
            let zeros = tvec!(0; k.rank() - 2);
            // ONNX deconv kernels are stored as gi_o_h_w (convolution are go_i_hw)
            let kernel = k
                .into_tensor()
                .split_axis(0, self.group)?
                .move_axis(1, 2)?
                .collapse_axis_with_next(0);

            let bias = if self.have_bias {
                Some(
                    target
                        .outlet_fact(inputs[2])?
                        .konst
                        .clone()
                        .context("bias must be a constant")?,
                )
            } else {
                None
            };

            let ci = KernelFormat::OIHW.input_channels(kernel.shape(), self.group).into_owned();
            let co = KernelFormat::OIHW.output_channels(kernel.shape(), self.group).into_owned();
            let op = if let Some(output_shape) = &self.output_shape {
                let x_shape = &target.outlet_fact(inputs[0])?.shape;
                let pool_spec = PoolSpec::new(
                    DataFormat::NCHW,
                    kernel.shape()[2..].into(),
                    self.padding_spec.clone(),
                    self.dilations.clone(),
                    self.strides.clone(),
                    ci,
                    co,
                );
                let adjustments = adjustments(
                    &pool_spec,
                    &x_shape.as_concrete().context("expects concrete dim for deconv")?[2..],
                    output_shape,
                )?;
                tract_core::ops::cnn::DeconvUnary::new(
                    pool_spec,
                    KernelFormat::OIHW,
                    kernel.into_arc_tensor(),
                    bias,
                    adjustments,
                    self.group,
                )
            } else {
                let pool_spec = PoolSpec::new(
                    DataFormat::NCHW,
                    kernel.shape()[2..].into(),
                    self.padding_spec.clone(),
                    self.dilations.clone(),
                    self.strides.clone(),
                    ci,
                    co,
                );
                tract_core::ops::cnn::DeconvUnary::new(
                    pool_spec,
                    KernelFormat::OIHW,
                    kernel.into_arc_tensor(),
                    bias,
                    self.adjustments.clone().unwrap_or(zeros),
                    self.group,
                )
            };
            target.wire_node(prefix, op, &[inputs[0]])
        } else {
            bail!("Kernel values are expected to be constant.")
        }
    }
}
