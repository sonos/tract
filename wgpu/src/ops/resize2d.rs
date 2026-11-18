use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

use crate::kernels::copy::wgpu_copy_nd_dispatch;
use crate::kernels::resize::wgpu_resize_2d_dispatch;

/// A resize over the two trailing axes in one launch, each output gathering
/// its `window_h x window_w` taps straight from the input. The per-axis form
/// costs a launch and an intermediate per axis, and re-uploads its plans every
/// frame; here the plans are inputs 1..=4 (row indices, row weights, column
/// indices, column weights), constants that live on the device. With
/// `concat_inputs`, input 5 is the second half of a channel concat the resize
/// reads in place. With `output_concat`, the resize writes its slab of a concat
/// and copies the trailing inputs into the rest of it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WgpuResize2d {
    pub window_h: usize,
    pub window_w: usize,
    pub output_shape: TVec<usize>,
    pub concat_inputs: bool,
    pub output_concat: Option<OutputConcat>,
}

/// The concat a resize writes into: along `axis`, before every resize axis
/// with only unit axes ahead of it, so the resize's slab is contiguous. The
/// last `siblings` inputs are the other concat inputs, in order; the resize
/// result sits after the first `position` of them.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OutputConcat {
    pub axis: usize,
    pub position: usize,
    pub siblings: usize,
}

impl Op for WgpuResize2d {
    fn name(&self) -> StaticName {
        "WgpuResize2d".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut info =
            vec![format!("windows {}x{} -> {:?}", self.window_h, self.window_w, self.output_shape)];
        if let Some(oc) = &self.output_concat {
            info.push(format!(
                "into concat axis {} at {} of {}",
                oc.axis,
                oc.position,
                oc.siblings + 1
            ));
        }
        Ok(info)
    }

    op_as_typed_op!();
}

impl EvalOp for WgpuResize2d {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let inputs =
            inputs.iter().map(|it| it.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;
        let input = inputs[0];
        let second = if self.concat_inputs { Some(inputs[5]) } else { None };
        let siblings = &inputs[5 + self.concat_inputs as usize..];
        let output = tract_gpu::turn_handler::make_tensor_for_node(
            ctx,
            input.datum_type(),
            &self.output_shape,
        )?;
        if output.len() == 0 {
            return Ok(tvec!(output.into_tensor().into_tvalue()));
        }
        let rank = self.output_shape.len();
        let plane = self.output_shape[rank - 2] * self.output_shape[rank - 1];
        let Some(oc) = &self.output_concat else {
            ensure!(siblings.is_empty());
            wgpu_resize_2d_dispatch(
                input,
                second,
                (inputs[1], inputs[2], self.window_h),
                (inputs[3], inputs[4], self.window_w),
                &output,
                None,
            )?;
            return Ok(tvec!(output.into_tensor().into_tvalue()));
        };
        ensure!(siblings.len() == oc.siblings && oc.position <= oc.siblings);
        let lens = siblings.iter().map(|s| s.shape()[oc.axis]).collect::<TVec<_>>();
        let resize_len = self.output_shape[oc.axis] - lens.iter().sum::<usize>();
        let resize_start = lens[..oc.position].iter().sum::<usize>();
        let slab = output.strides()[oc.axis] as usize;
        let mut start = 0;
        for (ix, sibling) in siblings.iter().enumerate() {
            if ix == oc.position {
                start += resize_len;
            }
            if sibling.len() > 0 {
                wgpu_copy_nd_dispatch(
                    sibling,
                    0,
                    sibling.strides(),
                    &output,
                    start * slab * output.datum_type().size_of(),
                    sibling.shape(),
                    output.strides(),
                )?;
            }
            start += lens[ix];
        }
        let planes = slab / plane;
        if resize_len > 0 {
            wgpu_resize_2d_dispatch(
                input,
                second,
                (inputs[1], inputs[2], self.window_h),
                (inputs[3], inputs[4], self.window_w),
                &output,
                Some((resize_start * planes, resize_len * planes)),
            )?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for WgpuResize2d {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let shape: TVec<TDim> = self.output_shape.iter().map(|d| d.to_dim()).collect();
            Ok(tvec!(facts[0].datum_type.fact(&shape)))
        })
        .with_context(|| "Error while computing facts for WgpuResize2d")
    }
}
