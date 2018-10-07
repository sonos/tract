use analyser::rules::prelude::*;
use ndarray::prelude::*;
use ops::prelude::*;

use num::cast::AsPrimitive;
use num::Float;

#[derive(Debug, Clone, new)]
pub struct Gemm {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

impl Gemm {
    fn eval_t<T: Datum + Float>(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>>
    where
        f32: AsPrimitive<T>,
    {
        let (a, b, c) = args_3!(inputs);
        let a = a.to_array_view::<T>()?.into_dimensionality()?;
        let at = if self.trans_a { a.t() } else { a };
        let b = b.to_array_view::<T>()?.into_dimensionality()?;
        let bt = if self.trans_b { b.t() } else { b };
        let mut c = if c.shape() == &[a.rows(), b.cols()] {
            c.into_array::<T>()?
            .into_dimensionality::<Ix2>()?
            .to_owned()
        } else {
            c.to_array_view::<T>()?.broadcast((a.cols(), b.rows()))
            .ok_or("Incompatible shape")?
            .to_owned()
        };
        ::ndarray::linalg::general_mat_mul(self.alpha.as_(), &at, &bt, self.beta.as_(), &mut c);
        Ok(tvec!(c.into()))
    }
}

impl Op for Gemm {
    fn name(&self) -> &str {
        "Gemm"
    }

    fn eval(&self, inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        dispatch_floatlike!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl InferenceRulesOp for Gemm {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[2].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape[0], &outputs[0].shape[0])?;
        s.equals(&inputs[0].shape[1], &inputs[1].shape[0])?;
        s.equals(&inputs[1].shape[1], &outputs[0].shape[1])?;
        Ok(())
    }
}
