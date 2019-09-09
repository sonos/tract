use crate::model::ParsingContext;
use crate::tfpb::node_def::NodeDef;
use tract_core::internal::*;
use tract_core::ops::cnn::*;
use tract_core::ops::nn::*;

pub fn depthwise_conv2d(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let data_format = super::data_format(pb)?;
    let padding = super::padding(pb)?;
    let strides = super::strides(pb)?.into();
    let dilations: TVec<usize> = pb.get_attr_list_int("dilations")?.into();
    if dilations.len() != 4 || dilations[0] != 1 && dilations[3] != 1 {
        Err(format!("dilations must be of the form [1, h, v, 1], found {:?}", dilations))?
    };
    Ok(Box::new(DepthwiseConv2d::new(data_format, padding, strides, dilations)))
}

#[derive(Debug, Clone, new)]
pub struct DepthwiseConv2d {
    data_format: DataFormat,
    padding: PaddingSpec,
    strides: TVec<usize>,
    dilations: TVec<usize>,
}

impl DepthwiseConv2d {
    fn to_core(&self, input_shape: &[TDim], kernel_shape: &[usize]) -> TractResult<Conv> {
        let shape = self.data_format.shape(&input_shape);
        let group = kernel_shape[2];
        let conv = Conv::new(
            self.data_format.clone(),
            KernelFormat::HWIO,
            Some(self.dilations[shape.hw_axes()].into()),
            None,
            self.padding.clone(),
            Some(self.strides[shape.hw_axes()].into()),
            group,
        );
        Ok(conv)
    }
}

impl Op for DepthwiseConv2d {
    fn name(&self) -> Cow<str> {
        "tf.DepthwiseConv2dNative".into()
    }

    fn cost(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let img = inputs[0];
        let ker = inputs[1].shape.as_finite().ok_or("Can not stream kernel")?;
        let shape = self.data_format.shape(img.shape.to_tvec());
        let output_dims = self.padding.compute(
            shape.hw_dims(),
            &ker[0..2],
            &self.dilations[1..3],
            &self.strides[1..3],
        );
        let n_output_points: TDim = output_dims.iter().map(|d| d.output.clone()).product::<TDim>();
        let kernel_surface = ker[0] * ker[1];
        let out_channels = ker[2] * ker[3];
        Ok(tvec!((
            Cost::FMA(f32::datum_type()),
            shape.n().clone() * out_channels * n_output_points * kernel_surface
        )))
    }

    not_a_typed_op!(); // FIXME translate to core as to_fixed instead of declutter, get rid of typed op impl
}

impl StatelessOp for DepthwiseConv2d {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let ishape: TVec<TDim> = inputs[0].shape().iter().map(|i| i.to_dim()).collect();
        let kshape = inputs[1].shape();
        self.to_core(&*ishape, kshape)?.eval(inputs)
    }
}

impl InferenceRulesOp for DepthwiseConv2d {
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
            let img = self.data_format.shape(img);
            s.equals(&inputs[1].shape[2], &inputs[0].shape[img.c_axis()])?;
            s.equals(&outputs[0].shape[img.n_axis()], img.n_dim())?;
            if ker.iter().all(|d| d.to_integer().is_ok()) {
                let ker: TVec<usize> =
                    ker.iter().map(|d| d.to_integer().unwrap() as usize).collect();
                let output_shape = self.padding.compute(
                    img.hw_dims(),
                    &ker[0..2],
                    &self.dilations[img.hw_axes()],
                    &self.strides[img.hw_axes()],
                );
                let in_channels = ker[2].to_integer()?;
                let multiplier = ker[3].to_integer()?;
                s.equals(&outputs[0].shape[img.h_axis()], &output_shape[0].output)?;
                s.equals(&outputs[0].shape[img.h_axis() + 1], &output_shape[1].output)?;
                s.equals(&outputs[0].shape[img.c_axis()], (in_channels * multiplier).to_dim())?;
            }
            Ok(())
        })?;
        Ok(())
    }

    inference_op_as_op!();

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let input = target.outlet_fact(mapping[&node.inputs[0]])?;
        let kernel = target.outlet_fact(mapping[&node.inputs[1]])?;
        let input_shape = input.shape.to_tvec();
        let kernel_shape = if let Some(s) = kernel.shape.as_finite() {
            s
        } else {
            bail!("Do not expect streaming on kernel dims");
        };
        let conv = self.to_core(&*input_shape, kernel_shape)?;
        target.wire_node(&*node.name, conv, [mapping[&node.inputs[0]], mapping[&node.inputs[1]]].as_ref())
    }
}
