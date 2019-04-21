use tract_core::internal::*;
use tract_core::ndarray::*;
use tract_core::ops::nn::*;

pub fn depthwise_conv2d(pb: &crate::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
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

impl Op for DepthwiseConv2d {
    fn name(&self) -> Cow<str> {
        "tf.DepthwiseConv2dNative".into()
    }

    fn cost(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let img = inputs[0];
        let ker = inputs[1].shape.as_finite().ok_or("Can not stream kernel")?;
        let shape = self.data_format.shape(img.shape.to_tvec());
        let output_dims = self.padding.compute(shape.hw_dims(), &ker[0..2], &self.dilations[1..3], &self.strides[1..3]);
        let n_output_points: TDim = output_dims.iter().map(|d| d.output).product::<TDim>();
        let kernel_surface = ker[0] * ker[1];
        let out_channels = ker[2] * ker[3];
        Ok(tvec!((
            Cost::FMA(f32::datum_type()),
            shape.n()
                * out_channels
                * n_output_points
                * kernel_surface
        )))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        let input_shape = inputs[0].shape.to_tvec();
        let shape = self.data_format.shape(&input_shape);
        let conv = tract_core::ops::nn::Conv::new(
            self.data_format.clone(),
            KernelFormat::HWIO,
            Some(self.dilations[shape.hw_axes()].into()),
            None,
            self.padding.clone(),
            Some(self.strides[shape.hw_axes()].into()),
            shape.c_dim().to_integer()? as usize,
        );
        Ok(Some(TypedModelPatch::replace_single_op(model, node, &*node.inputs, conv)?))
    }
}

impl StatelessOp for DepthwiseConv2d {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (img, ker) = args_2!(inputs);
        let img = img.to_array_view::<f32>()?;
        let ker = ker.to_array_view::<f32>()?;
        let input_shape = self.data_format.shape(img.shape());
        let patch = PatchSpec {
            data_format: self.data_format.clone(),
            dilations: self.dilations[input_shape.hw_axes()].into(),
            kernel_shape: ker.shape()[0..2].into(),
            padding: self.padding.clone(),
            strides: self.strides[input_shape.hw_axes()].into(),
            input_full_shape: img.shape().into()
        }.into_patch();
        let out_channels = ker.shape()[2] * ker.shape()[3];
        let visitor = patch.wrap(&img);
        let output_shape = patch.output_full_shape(out_channels);
        let output = ArrayD::<f32>::from_shape_fn(&*output_shape, |mut coords| {
            let k = coords[input_shape.c_axis()] / ker.shape()[3];
            let q = coords[input_shape.c_axis()] % ker.shape()[3];
            coords[input_shape.c_axis()] = k;
            let mut it = visitor.at(coords.slice());
            let mut sum = 0.0f32;
            for di in 0..ker.shape()[0] {
                for dj in 0..ker.shape()[1] {
                    let vi = it.next().unwrap().unwrap_or(0.0);
                    let vk = ker[[di, dj, k, q]];
                    sum += vi * vk;
                }
            }
            sum
        });
        Ok(tvec!(output.into()))
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
                let ker:TVec<usize> = ker.iter().map(|d| d.to_integer().unwrap() as usize).collect();
                let output_shape = self.padding.compute(
                    img.hw_dims(),
                    &ker[0..2],
                    &self.dilations[img.hw_axes()],
                    &self.strides[img.hw_axes()],
                );
                let in_channels = ker[2].to_integer()?;
                let multiplier = ker[3].to_integer()?;
                s.equals(&outputs[0].shape[img.h_axis()], output_shape[0].output)?;
                s.equals(&outputs[0].shape[img.h_axis() + 1], output_shape[1].output)?;
                s.equals(&outputs[0].shape[img.c_axis()], (in_channels * multiplier).to_dim())?;
            }
            Ok(())
        })?;
        Ok(())
    }
}
