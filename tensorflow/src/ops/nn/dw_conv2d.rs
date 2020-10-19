use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;
use tract_hir::internal::*;
use tract_hir::ops::cnn::*;
use tract_hir::ops::nn::*;

pub fn depthwise_conv2d(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let data_format = super::data_format(pb)?;
    let padding = super::padding(pb)?;
    let strides = super::strides(pb)?.into();
    let dilations: TVec<usize> = pb.get_attr_list_int("dilations")?.into();
    if dilations.len() != 4 || dilations[0] != 1 && dilations[3] != 1 {
        bail!("dilations must be of the form [1, h, v, 1], found {:?}", dilations)
    };
    Ok(expand(DepthwiseConv2d::new(data_format, padding, strides, dilations)))
}

#[derive(Debug, Clone, new, Hash)]
pub struct DepthwiseConv2d {
    data_format: DataFormat,
    padding: PaddingSpec,
    strides: TVec<usize>,
    dilations: TVec<usize>,
}

tract_data::impl_dyn_hash!(DepthwiseConv2d);

impl Expansion for DepthwiseConv2d {
    fn name(&self) -> Cow<str> {
        "DepthwiseConv2dNative".into()
    }

    op_tf!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
        ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].rank, 4)?;
        s.equals(&inputs[1].rank, 4)?;
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&outputs[0].rank, 4)?;
        s.given_2(&inputs[0].shape, &inputs[1].shape, move |s, img, ker| {
            let img = self.data_format.shape(img)?;
            s.equals(&inputs[1].shape[2], &inputs[0].shape[img.c_axis()])?;
            s.equals(&outputs[0].shape[img.n_axis().unwrap()], img.n_dim().unwrap())?;
            if let Ok(ker) = ker.iter().map(|d| d.to_usize()).collect::<TractResult<TVec<_>>>() {
                let output_shape = self.padding.compute(
                    img.hw_dims(),
                    &ker[0..2],
                    &self.dilations[img.hw_axes()],
                    &self.strides[img.hw_axes()],
                    );
                let in_channels = ker[2].to_usize()?;
                let multiplier = ker[3].to_usize()?;
                s.equals(&outputs[0].shape[img.h_axis()], &output_shape[0].output)?;
                s.equals(&outputs[0].shape[img.h_axis() + 1], &output_shape[1].output)?;
                s.equals(&outputs[0].shape[img.c_axis()], (in_channels * multiplier).to_dim())?;
            }
            Ok(())
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
        ) -> TractResult<TVec<OutletId>> {
        let input = model.outlet_fact(inputs[0])?;
        let kernel = model.outlet_fact(inputs[1])?;
        let input_shape = input.shape.to_tvec();
        let kernel_shape = if let Some(s) = kernel.shape.as_finite() {
            s
        } else {
            bail!("Do not expect streaming on kernel dims");
        };
        let shape = self.data_format.shape(&input_shape)?;
        let mut conv = Conv::default()
            .hwio()
            .group(kernel_shape[2])
            .dilations(self.dilations[shape.hw_axes()].into())
            .strides(self.strides[shape.hw_axes()].into())
            .padding(self.padding.clone());
        if self.data_format == DataFormat::NHWC {
            conv = conv.nhwc()
        }
        let conv = conv.to_unary(&[input, kernel])?.context("Failed to translate")?;
        model.wire_node(prefix, conv, &inputs[0..1])
    }
}
