use crate::internal::*;
use crate::ops::cnn::Patch;
use crate::ops::cnn::patches::{Zone, ZoneScanner};
use crate::ops::nn::DataShape;

/// A group-1 NCHW f32 convolution computed as a filter along W, four output channels at a time,
/// instead of im2col and a matrix product. `Conv` builds it at codegen on hosts that have the
/// `conv_w4` kernel, and the `depthwise_w` one too when the output channels are not a multiple
/// of four.
#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct DirectSpatialConv {
    patch: Patch,
    input_shape: DataShape,
    output_shape: DataShape,
}

impl Op for DirectSpatialConv {
    fn name(&self) -> StaticName {
        "DirectSpatialConv".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.patch)])
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalOp for DirectSpatialConv {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (img, kernel, bias) = args_3!(inputs);
        ensure!(img.datum_type() == f32::datum_type(), "DirectSpatialConv is f32 only");
        let output = unsafe { self.eval_f32(&img, &kernel, &bias)? };
        Ok(tvec!(output.into_tvalue()))
    }
}

impl DirectSpatialConv {
    /// Every run of output points along W in `zone`: the offsets of its first input center and
    /// first output point, its length, and the input and output steps between two points.
    unsafe fn runs(&self, zone: &Zone, mut f: impl FnMut(isize, isize, usize, isize, isize)) {
        if zone.valid && zone.output_ranges.len() == 2 && *self.output_shape.w_stride() == 1 {
            let in_h = (*self.input_shape.h_stride() * self.patch.spec.strides[0]) as isize;
            let in_w = (*self.input_shape.w_stride() * self.patch.spec.strides[1]) as isize;
            let out_h = *self.output_shape.h_stride() as isize;
            let x0 = zone.output_ranges[1].start as isize;
            let len = zone.output_ranges[1].len();
            for y in zone.output_ranges[0].clone() {
                f(y as isize * in_h + x0 * in_w, y as isize * out_h + x0, len, in_w, 1);
            }
        } else {
            let mut visitor = ZoneScanner::new(zone, &self.patch);
            while !visitor.done {
                f(
                    visitor.input_center_offset,
                    visitor.output_offset,
                    visitor.inner_loop_len,
                    visitor.inner_loop_input_full_stride,
                    visitor.inner_loop_output_stride,
                );
                unsafe { visitor.next_non_inner_axis() };
            }
        }
    }

    unsafe fn eval_f32(&self, img: &Tensor, kernel: &Tensor, bias: &Tensor) -> TractResult<Tensor> {
        let conv_w4 = tract_linalg::routines::conv_w4_f32()
            .context("DirectSpatialConv needs a conv_w4 kernel")?;
        let ic = *self.input_shape.c();
        let oc = *self.output_shape.c();
        let conv_w1 = tract_linalg::routines::depthwise_w_f32();
        ensure!(
            oc.is_multiple_of(4) || conv_w1.is_some(),
            "DirectSpatialConv needs a depthwise_w kernel for its last channels"
        );
        let kvol = self.patch.spec.kernel_shape.iter().product::<usize>();
        let n = *self.input_shape.n().unwrap_or(&1);
        let n_stride_i = *self.input_shape.n_stride().unwrap_or(&0) as isize;
        let n_stride_o = *self.output_shape.n_stride().unwrap_or(&0) as isize;
        let c_stride_i = *self.input_shape.c_stride() as isize;
        let c_stride_o = *self.output_shape.c_stride() as isize;
        let mut output = unsafe { Tensor::uninitialized::<f32>(&self.output_shape.shape)? };
        let mut offsets = Vec::with_capacity(ic * kvol);
        let mut taps = Vec::with_capacity(4 * ic * kvol);
        unsafe {
            let iptr = img.as_ptr::<f32>()?;
            let kptr = kernel.as_ptr::<f32>()?;
            let bptr = bias.as_ptr::<f32>()?;
            let optr = output.as_ptr_mut::<f32>()?;
            for zone in &self.patch.zones {
                offsets.clear();
                for c in 0..ic as isize {
                    offsets.extend(zone.values_offsets.iter().map(|(_, off)| c * c_stride_i + off));
                }
                let n_taps = offsets.len();
                let mut o = 0;
                while o < oc {
                    let group = if o + 4 <= oc { 4 } else { 1 };
                    taps.clear();
                    for g in o..o + group {
                        for c in 0..ic {
                            let k = kptr.add((g * ic + c) * kvol);
                            taps.extend(zone.values_offsets.iter().map(|(ix, _)| *k.add(*ix)));
                        }
                    }
                    let b = std::slice::from_raw_parts(bptr.add(o), group);
                    for ni in 0..n as isize {
                        let iptr = iptr.offset(n_stride_i * ni);
                        let optr = optr.offset(n_stride_o * ni + c_stride_o * o as isize);
                        self.runs(zone, |center, out, len, in_stride, out_stride| {
                            let ip = iptr.offset(center);
                            let op = optr.offset(out);
                            if out_stride == 1 && in_stride >= 1 {
                                if group == 4 {
                                    let b4 = [b[0], b[1], b[2], b[3]];
                                    conv_w4(
                                        ip, op, &taps, &offsets, &b4, len, in_stride, c_stride_o,
                                    );
                                } else if let Some(conv_w1) = conv_w1 {
                                    conv_w1(ip, op, &taps, &offsets, b[0], len, in_stride);
                                }
                                return;
                            }
                            for g in 0..group {
                                let taps = &taps[g * n_taps..][..n_taps];
                                for i in 0..len as isize {
                                    let ip = ip.offset(i * in_stride);
                                    let sum = taps
                                        .iter()
                                        .zip(&offsets)
                                        .fold(b[g], |sum, (t, off)| sum + t * *ip.offset(*off));
                                    *op.offset(g as isize * c_stride_o + i * out_stride) = sum;
                                }
                            }
                        });
                    }
                    o += group;
                }
            }
        }
        Ok(output)
    }
}

impl TypedOp for DirectSpatialConv {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3);
        Ok(tvec!(inputs[0].datum_type.fact(&self.output_shape.shape)))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let [_input, kernel, _bias] = inputs else {
            bail!("DirectSpatialConv expects three inputs");
        };
        let n_output_points = self.patch.output_shape.iter().cloned().product::<usize>();
        Ok(tvec!((
            Cost::FMA(inputs[0].datum_type),
            kernel.shape.volume() * self.input_shape.n().unwrap_or(&1) * n_output_points
        )))
    }

    as_op!();
}
