use crate::internal::*;
use num_complex::Complex;
use rustfft::num_traits::{Float, FromPrimitive};
use rustfft::{FftDirection, FftNum};
use tract_data::itertools::Itertools;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Fft {
    pub axis: usize,
    pub inverse: bool,
}

impl Fft {
    fn eval_t<T: Datum + FftNum + FromPrimitive + Float>(
        &self,
        tensor: &mut Tensor,
    ) -> TractResult<()> {
        let mut iterator_shape: TVec<usize> = tensor.shape().into();
        iterator_shape.pop(); // last dim is [re, im]
        iterator_shape[self.axis] = 1;
        let len = tensor.shape()[self.axis];
        let direction = if self.inverse { FftDirection::Inverse } else { FftDirection::Forward };
        let fft = rustfft::FftPlanner::new().plan_fft(len, direction);
        let mut tensor_plain = tensor.try_as_plain_mut()?;
        let mut array = tensor_plain.to_array_view_mut::<T>()?;
        let mut v = Vec::with_capacity(len);
        for coords in tract_ndarray::indices(&*iterator_shape) {
            v.clear();
            let mut slice = array.slice_each_axis_mut(|ax| {
                if ax.axis.index() == self.axis || ax.stride == 1 {
                    // ax.stride == 1 => last dim
                    (..).into()
                } else {
                    let c = coords[ax.axis.index()] as isize;
                    (c..=c).into()
                }
            });
            v.extend(slice.iter().tuples().map(|(r, i)| Complex::new(*r, *i)));
            fft.process(&mut v);
            slice
                .iter_mut()
                .zip(v.iter().flat_map(|cmpl| [cmpl.re, cmpl.im].into_iter()))
                .for_each(|(s, v)| *s = v);
        }
        Ok(())
    }
}

impl Op for Fft {
    fn name(&self) -> StaticName {
        "Fft".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![if self.inverse { "inverse" } else { "forward" }.into()])
    }

    op_as_typed_op!();
}

impl EvalOp for Fft {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut tensor = args_1!(inputs).into_tensor();
        match tensor.datum_type() {
            DatumType::F16 => {
                let mut temp = tensor.cast_to::<f32>()?.into_owned();
                self.eval_t::<f32>(&mut temp)?;
                tensor = temp.cast_to::<f16>()?.into_owned();
            }
            DatumType::F32 => self.eval_t::<f32>(&mut tensor)?,
            DatumType::F64 => self.eval_t::<f64>(&mut tensor)?,
            _ => bail!("FFT not implemented for type {:?}", tensor.datum_type()),
        }
        Ok(tvec!(tensor.into_tvalue()))
    }
}

impl TypedOp for Fft {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        anyhow::ensure!(
            inputs[0].rank() >= 2,
            "Expect rank 2 (one for fft dimension, one for complex dimension"
        );
        anyhow::ensure!(
            inputs[0].shape.last().unwrap() == &2.to_dim(),
            "Fft operators expect inner (last) dimension to be 2 for real and imaginary part"
        );
        Ok(tvec!(inputs[0].without_value()))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        // Fft is rank-preserving but it is NOT axes-natural: two axes do
        // not map 1-to-1 from input to output and must be declared as a
        // separate input-only and output-only axis.
        //
        //   - the FFT axis (`self.axis`): every output sample along it
        //     depends on every input sample, so the axis cannot be
        //     sliced or streamed.
        //   - the trailing complex axis (`rank - 1`): the FFT mixes the
        //     real and imaginary parts, so re/im do not map 1-to-1.
        //
        // Splitting them is exactly what makes the generic pulse fallback
        // bail when asked to track a streaming axis through the FFT or
        // complex axis, while every genuine batch axis stays 1-to-1 and
        // is handled by the per-pulse `PulseWrappingOp`. No dedicated
        // `Fft` pulsifier is needed.
        let rank = inputs[0].rank();
        let complex_axis = rank - 1;
        let mut axes = tvec!();
        let mut alphabet = 'a'..;
        for i in 0..rank {
            if i == self.axis || i == complex_axis {
                axes.push(crate::axes::Axis::new(alphabet.next().unwrap(), 1, 1).input(0, i));
                axes.push(crate::axes::Axis::new(alphabet.next().unwrap(), 1, 1).output(0, i));
            } else {
                axes.push(
                    crate::axes::Axis::new(alphabet.next().unwrap(), 1, 1).input(0, i).output(0, i),
                );
            }
        }
        AxesMapping::new(1, 1, axes)
    }

    as_op!();
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Stft {
    pub axis: usize,
    pub frame: usize,
    pub stride: usize,
    pub window: Option<Arc<Tensor>>,
}

impl Stft {
    /// The window zero-padded to `frame` and centered, ready to scale a whole frame in one
    /// pass; `None` when the op carries no window.
    fn padded_window<T: Datum + Float>(&self) -> TractResult<Option<Vec<T>>> {
        let Some(window) = &self.window else { return Ok(None) };
        let window = window.try_as_plain()?.as_slice::<T>()?;
        ensure!(
            window.len() <= self.frame,
            "Stft window ({}) is longer than the frame ({})",
            window.len(),
            self.frame
        );
        let pad_left = (self.frame - window.len()) / 2;
        let mut padded = vec![T::zero(); self.frame];
        padded[pad_left..pad_left + window.len()].copy_from_slice(window);
        Ok(Some(padded))
    }

    fn eval_t<T: Datum + FftNum + FromPrimitive + Float>(
        &self,
        input: &Tensor,
    ) -> TractResult<Tensor> {
        let rank = input.rank();
        let frames = (input.shape()[self.axis] - self.frame) / self.stride + 1;
        let mut output_shape: TVec<usize> = input.shape().into();
        output_shape[self.axis] = frames;
        output_shape.insert(self.axis + 1, self.frame);
        let mut output = unsafe { Tensor::uninitialized::<T>(&output_shape)? };

        let mut iterator_shape: TVec<usize> = input.shape().into();
        iterator_shape.pop(); // last dim is [re, im]
        iterator_shape[self.axis] = 1;

        let in_strides: TVec<usize> = input.strides().iter().map(|s| *s as usize).collect();
        let out_strides: TVec<usize> = output.strides().iter().map(|s| *s as usize).collect();
        let time_stride = in_strides[self.axis];
        let frame_stride = out_strides[self.axis];
        let bin_stride = out_strides[self.axis + 1];

        let window = self.padded_window::<T>()?;
        let fft = rustfft::FftPlanner::new().plan_fft_forward(self.frame);

        let input_plain = input.try_as_plain()?;
        let data = input_plain.as_slice::<T>()?;
        let mut output_plain = output.try_as_plain_mut()?;
        let out = output_plain.as_slice_mut::<T>()?;

        let mut v = Vec::with_capacity(self.frame);
        for coords in tract_ndarray::indices(&*iterator_shape) {
            let mut in_base = 0;
            let mut out_base = 0;
            for i in (0..rank - 1).filter(|i| *i != self.axis) {
                in_base += coords[i] * in_strides[i];
                out_base += coords[i] * out_strides[i + usize::from(i > self.axis)];
            }
            for f in 0..frames {
                let src = in_base + f * self.stride * time_stride;
                v.clear();
                if time_stride == 2 {
                    v.extend(
                        data[src..src + 2 * self.frame]
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|c| Complex::new(c[0], c[1])),
                    );
                } else {
                    v.extend((0..self.frame).map(|k| {
                        let o = src + k * time_stride;
                        Complex::new(data[o], data[o + 1])
                    }));
                }
                if let Some(window) = &window {
                    v.iter_mut().zip(window).for_each(|(v, w)| {
                        v.re = v.re * *w;
                        v.im = v.im * *w;
                    });
                }
                fft.process(&mut v);
                let dst = out_base + f * frame_stride;
                if bin_stride == 2 {
                    out[dst..dst + 2 * self.frame]
                        .as_chunks_mut::<2>()
                        .0
                        .iter_mut()
                        .zip(&v)
                        .for_each(|(o, c)| {
                            o[0] = c.re;
                            o[1] = c.im;
                        });
                } else {
                    for (k, c) in v.iter().enumerate() {
                        let o = dst + k * bin_stride;
                        out[o] = c.re;
                        out[o + 1] = c.im;
                    }
                }
            }
        }
        Ok(output)
    }
}

impl Op for Stft {
    fn name(&self) -> StaticName {
        "STFT".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Stft {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let output = match input.datum_type() {
            DatumType::F16 => {
                let temp = input.cast_to::<f32>()?;
                self.eval_t::<f32>(&temp)?.cast_to::<f16>()?.into_owned()
            }
            DatumType::F32 => self.eval_t::<f32>(&input)?,
            DatumType::F64 => self.eval_t::<f64>(&input)?,
            _ => bail!("FFT not implemented for type {:?}", input.datum_type()),
        };
        Ok(tvec!(output.into_tvalue()))
    }
}

impl TypedOp for Stft {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        anyhow::ensure!(
            inputs[0].rank() >= 2,
            "Expect rank 2 (one for fft dimension, one for complex dimension"
        );
        anyhow::ensure!(
            inputs[0].shape.last().unwrap() == &2.to_dim(),
            "Fft operators expect inner (last) dimension to be 2 for real and imaginary part"
        );
        let mut shape = inputs[0].shape.to_tvec();
        let frames = (inputs[0].shape[self.axis].clone() - self.frame) / self.stride + 1;
        shape[self.axis] = frames;
        shape.insert(self.axis + 1, self.frame.to_dim());
        Ok(tvec!(inputs[0].datum_type.fact(shape)))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<crate::axes::AxesMapping> {
        // Stft is NOT rank-preserving: it inserts a frame axis at
        // `axis + 1`. The mapping is:
        //   - axes 0..self.axis (leading dims): 1-to-1 input <-> output.
        //   - input axis `self.axis` (the time axis) <-> output axis
        //     `self.axis` (now the n_frames axis -- same position, the
        //     dim shrinks from `T` to `(T - frame) / stride + 1`).
        //   - output axis `self.axis + 1` (the inserted frame axis):
        //     output-only, no input correspondence.
        //   - input axes `self.axis + 1..rank` (trailing dims incl.
        //     the complex pair) <-> output axes `self.axis + 2..rank+1`
        //     (shifted right by 1 to make room for the frame axis).
        //
        // Without this mapping the generic `PulseWrappingOp` fallback
        // bails with "could not track pulsing axis" the moment a user
        // streams a non-time axis through STFT (typical pattern: a
        // batched STFT pipeline that streams the batch axis).
        let in_rank = inputs[0].rank();
        let mut axes = tvec!();
        let mut alphabet = 'a'..;
        for i in 0..in_rank {
            let out_axis = if i <= self.axis { i } else { i + 1 };
            axes.push(
                crate::axes::Axis::new(alphabet.next().unwrap(), 1, 1)
                    .input(0, i)
                    .output(0, out_axis),
            );
        }
        // Inserted frame axis (output-only).
        axes.push(crate::axes::Axis::new(alphabet.next().unwrap(), 1, 1).output(0, self.axis + 1));
        crate::axes::AxesMapping::new(1, 1, axes)
    }

    as_op!();
}
