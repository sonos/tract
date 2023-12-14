use crate::internal::*;
use num_complex::Complex;
use rustfft::num_traits::{Float, FromPrimitive};
use rustfft::{FftDirection, FftNum};
use tract_data::itertools::Itertools;
use tract_ndarray::Axis;

#[derive(Clone, Debug, Hash)]
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
        let mut array = tensor.to_array_view_mut::<T>()?;
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
    fn name(&self) -> Cow<str> {
        "Fft".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![if self.inverse { "inverse" } else { "forward" }.into()])
    }

    op_as_typed_op!();
}

impl EvalOp for Fft {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
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

    as_op!();
}

#[derive(Clone, Debug, Hash)]
pub struct Stft {
    pub axis: usize,
    pub frame: usize,
    pub stride: usize,
    pub window: Option<Arc<Tensor>>,
}

impl Stft {
    fn eval_t<T: Datum + FftNum + FromPrimitive + Float>(
        &self,
        input: &Tensor,
    ) -> TractResult<Tensor> {
        let mut iterator_shape: TVec<usize> = input.shape().into();
        iterator_shape.pop(); // [re,im]
        iterator_shape[self.axis] = 1;
        let mut output_shape: TVec<usize> = input.shape().into();
        let frames = (input.shape()[self.axis] - self.frame) / self.stride + 1;
        output_shape.insert(self.axis, frames);
        output_shape[self.axis + 1] = self.frame;
        let mut output = unsafe { Tensor::uninitialized::<T>(&output_shape)? };
        let fft = rustfft::FftPlanner::new().plan_fft_forward(self.frame);
        let input = input.to_array_view::<T>()?;
        let mut oview = output.to_array_view_mut::<T>()?;
        let mut v = Vec::with_capacity(self.frame);
        for coords in tract_ndarray::indices(&*iterator_shape) {
            let islice = input.slice_each_axis(|ax| {
                if ax.axis.index() == self.axis || ax.stride == 1 {
                    (..).into()
                } else {
                    let c = coords[ax.axis.index()] as isize;
                    (c..=c).into()
                }
            });
            let mut oslice = oview.slice_each_axis_mut(|ax| {
                if ax.stride == 1 {
                    (..).into()
                } else if ax.axis.index() < self.axis {
                    let c = coords[ax.axis.index()] as isize;
                    (c..=c).into()
                } else if ax.axis.index() == self.axis || ax.axis.index() == self.axis + 1 {
                    (..).into()
                } else {
                    let c = coords[ax.axis.index() - 1] as isize;
                    (c..=c).into()
                }
            });
            for f in 0..frames {
                v.clear();
                v.extend(
                    islice
                        .iter()
                        .tuples()
                        .skip(self.stride * f)
                        .take(self.frame)
                        .map(|(re, im)| Complex::new(*re, *im)),
                );
                if let Some(win) = &self.window {
                    let win = win.as_slice::<T>()?;
                    v.iter_mut()
                        .zip(win.iter())
                        .for_each(|(v, w)| *v = *v * Complex::new(*w, T::zero()));
                }
                fft.process(&mut v);
                oslice
                    .index_axis_mut(Axis(self.axis), f)
                    .iter_mut()
                    .zip(v.iter().flat_map(|cmpl| [cmpl.re, cmpl.im].into_iter()))
                    .for_each(|(s, v)| *s = v);
            }
        }
        Ok(output)
    }
}

impl Op for Stft {
    fn name(&self) -> Cow<str> {
        "STFT".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Stft {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
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

    as_op!();
}
