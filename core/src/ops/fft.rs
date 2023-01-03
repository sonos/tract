use crate::internal::*;
use rustfft::num_traits::{Float, FromPrimitive};
use rustfft::{FftDirection, FftNum};
use std::ops::Mul;

#[derive(Clone, Debug, Hash)]
pub struct Fft {
    pub axis: usize,
    pub inverse: bool,
}

impl_dyn_hash!(Fft);

impl Fft {
    fn eval_t<T: Datum + FftNum + FromPrimitive + Float>(
        &self,
        tensor: &mut Tensor,
    ) -> TractResult<()>
    where
        Complex<T>: Datum + Mul<Complex<T>, Output = Complex<T>>,
    {
        let mut iterator_shape: TVec<usize> = tensor.shape().into();
        iterator_shape[self.axis] = 1;
        let len = tensor.shape()[self.axis];
        let direction = if self.inverse { FftDirection::Inverse } else { FftDirection::Forward };
        let fft = rustfft::FftPlanner::new().plan_fft(len, direction);
        let mut array = tensor.to_array_view_mut::<Complex<T>>()?;
        let mut v = Vec::with_capacity(len);
        for coords in tract_ndarray::indices(&*iterator_shape) {
            v.clear();
            let mut slice = array.slice_each_axis_mut(|ax| {
                if ax.axis.index() == self.axis {
                    (..).into()
                } else {
                    let c = coords[ax.axis.index()] as isize;
                    (c..=c).into()
                }
            });
            v.extend(slice.iter().copied());
            fft.process(&mut *v);
            slice.iter_mut().zip(v.iter()).for_each(|(s, v)| *s = *v);
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

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let mut tensor = args_1!(inputs).into_tensor();
        match tensor.datum_type() {
            DatumType::ComplexF32 => self.eval_t::<f32>(&mut tensor)?,
            DatumType::ComplexF64 => self.eval_t::<f64>(&mut tensor)?,
            _ => bail!("FFT not implemented for type {:?}", tensor.datum_type()),
        }
        Ok(tvec!(tensor.into_tvalue()))
    }
}

impl TypedOp for Fft {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if !inputs[0].datum_type.is_complex() {
            bail!("Fft operators expect input in complex form");
        }
        Ok(tvec!(inputs[0].without_value()))
    }

    as_op!();
}
