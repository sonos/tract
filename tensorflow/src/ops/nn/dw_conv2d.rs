use tract_core::internal::*;
use tract_core::ndarray::*;
use tract_core::ops::nn::*;

pub fn depthwise_conv2d(pb: &crate::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
    let fmt = super::data_format(pb)?;
    let padding = super::padding(pb)?;
    let strides = super::strides(pb)?.into();
    let dilations: TVec<usize> = pb.get_attr_list_int("dilations")?.into();
    if dilations.len() != 4 || dilations[0] != 1 && dilations[3] != 1 {
        Err(format!("dilations must be of the form [1, h, v, 1], found {:?}", dilations))?
    };
    Ok(Box::new(DepthwiseConv2d::new(fmt, padding, strides, dilations)))
}

#[derive(Debug, Clone, new)]
pub struct DepthwiseConv2d {
    fmt: DataFormat,
    padding: PaddingSpec,
    strides: TVec<usize>,
    dilations: TVec<usize>,
}

impl Op for DepthwiseConv2d {
    fn name(&self) -> Cow<str> {
        "tf.DepthwiseConv2dNative".into()
    }
}

impl StatelessOp for DepthwiseConv2d {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (img, ker) = args_2!(inputs);
        let img = img.to_array_view::<f32>()?;
        let ker = ker.to_array_view::<f32>()?;
        let input_shape = self.fmt.shape(img.shape());
        let patch = Patch::new(
            self.fmt.clone(),
            self.dilations[input_shape.hw_axes()].into(),
            ker.shape()[0..2].into(),
            &self.padding,
            self.strides[input_shape.hw_axes()].into(),
            img.shape().into(),
        );
        println!("{:#?}", patch);
        println!("{:?}", img);
        println!("{:?}", ker);
        let out_channels = ker.shape()[2] * ker.shape()[3];
        let visitor = patch.wrap(&img);
        let output_shape = patch.output_full_shape(out_channels);
        let output = ArrayD::<f32>::from_shape_fn(&*output_shape, |mut coords| {
            let k = coords[input_shape.c_axis()] / ker.shape()[3];
            let q = coords[input_shape.c_axis()] % ker.shape()[3];
            coords[input_shape.c_axis()] = k;
            println!("coords fix : {:?}", coords);
            println!("{:?}", visitor.at(coords.slice()).collect::<Vec<_>>());
            let mut it = visitor.at(coords.slice());
            let mut sum = 0.0f32;
            for di in 0..ker.shape()[0] {
                for dj in 0..ker.shape()[1] {
                    let vi = it.next().unwrap().unwrap_or(0.0);
                    let vk = ker[[di, dj, k, q]];
                    println!("{} {}", vi, vk);
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
            let img = self.fmt.shape(img);
            s.equals(&inputs[1].shape[2], &inputs[0].shape[img.c_axis()])?;
            s.equals(&outputs[0].shape[img.n_axis()], img.n_dim())?;
            let output_shape = self.padding.compute(
                img.hw_dims(),
                &ker[0..2],
                &self.dilations[img.hw_axes()],
                &self.strides[img.hw_axes()],
            );
            s.equals(&outputs[0].shape[img.h_axis()], output_shape.output[0])?;
            s.equals(&outputs[0].shape[img.h_axis() + 1], output_shape.output[1])?;
            let in_channels = ker[2].to_integer()?;
            let multiplier = ker[3].to_integer()?;
            s.equals(&outputs[0].shape[img.c_axis()], (in_channels * multiplier).to_dim())?;
            Ok(())
        })?;
        Ok(())
    }
}
