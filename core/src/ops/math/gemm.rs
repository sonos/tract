use crate::ops::prelude::*;
use ndarray::prelude::*;

use num_traits::AsPrimitive;
use num_traits::Float;

#[derive(Debug, Clone, new)]
pub struct Gemm {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    have_c: bool,
}

impl Gemm {
    fn eval_t_3<T: Datum + Float>(
        &self,
        mut inputs: TVec<SharedTensor>,
    ) -> TractResult<TVec<SharedTensor>>
    where
        f32: AsPrimitive<T>,
    {
        let (a, b, c) = args_3!(inputs);
        let a = a.to_array_view::<T>()?.into_dimensionality()?;
        let at = if self.trans_a { a.t() } else { a };
        let b = b.to_array_view::<T>()?.into_dimensionality()?;
        let bt = if self.trans_b { b.t() } else { b };
        let c_shape = (at.rows(), bt.cols());
        let mut c = if c.shape() == &[c_shape.0, c_shape.1] {
            c.to_array::<T>()?.into_dimensionality::<Ix2>()?.to_owned()
        } else {
            c.to_array_view::<T>()?
                .broadcast(c_shape)
                .ok_or_else(|| format!("Incompatible broadcast: {:?} to {:?}", c.shape(), c_shape))?
                .to_owned()
        };
        ::ndarray::linalg::general_mat_mul(self.alpha.as_(), &at, &bt, self.beta.as_(), &mut c);
        Ok(tvec!(c.into()))
    }

    fn eval_t_2<T: Datum + Float>(
        &self,
        mut inputs: TVec<SharedTensor>,
    ) -> TractResult<TVec<SharedTensor>>
    where
        f32: AsPrimitive<T>,
    {
        let (a, b) = args_2!(inputs);
        let a = a.to_array_view::<T>()?.into_dimensionality()?;
        let at = if self.trans_a { a.t() } else { a };
        let b = b.to_array_view::<T>()?.into_dimensionality()?;
        let bt = if self.trans_b { b.t() } else { b };
        let c_shape = (at.rows(), bt.cols());
        let mut c = unsafe { Array::uninitialized((c_shape.0, c_shape.1)) };
        ::ndarray::linalg::general_mat_mul(self.alpha.as_(), &at, &bt, T::zero(), &mut c);
        Ok(tvec!(c.into()))
    }
}

impl Op for Gemm {
    fn name(&self) -> Cow<str> {
        "Gemm".into()
    }

    fn reduce(
        &self,
        inputs: TVec<&TensorFact>,
        _outputs: TVec<&TensorFact>,
        _phase: ReductionPhase,
    ) -> TractResult<Option<ReducedOpRewire>> {

        if self.have_c {
            if let (Some(b), Some(c)) = (inputs[1].concretize(), inputs[2].concretize()) {
                return Ok(Some(ReducedOpRewire::unary(GemmUnaryA {
                    alpha: self.alpha,
                    beta: self.beta,
                    trans_a: self.trans_a,
                    trans_b: self.trans_b,
                    b: b.to_tensor(),
                    c: c.to_tensor(),
                })));
            }

            if let (Some(a), Some(c)) = (inputs[0].concretize(), inputs[2].concretize()) {
                return Ok(Some(ReducedOpRewire::unary(GemmUnaryB {
                    alpha: self.alpha,
                    beta: self.beta,
                    trans_a: self.trans_a,
                    trans_b: self.trans_b,
                    a: a.to_tensor(),
                    c: c.to_tensor(),
                })));
            }
        } else {
            if let Some(b) = inputs[1].concretize() {
                return Ok(Some(ReducedOpRewire::unary(GemmUnaryA {
                    alpha: self.alpha,
                    beta: 0.0,
                    trans_a: self.trans_a,
                    trans_b: self.trans_b,
                    b: b.to_tensor(),
                    c: Tensor::from(0.0),
                })));
            }
            if let Some(a) = inputs[0].concretize() {
                return Ok(Some(ReducedOpRewire::unary(GemmUnaryB {
                    alpha: self.alpha,
                    beta: 0.0,
                    trans_a: self.trans_a,
                    trans_b: self.trans_b,
                    a: a.to_tensor(),
                    c: Tensor::from(0.0),
                })));
            }
        }

        Ok(None)
    }
}

impl StatelessOp for Gemm {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        if self.have_c {
            dispatch_floatlike!(Self::eval_t_3(inputs[0].datum_type())(self, inputs))
        } else {
            dispatch_floatlike!(Self::eval_t_2(inputs[0].datum_type())(self, inputs))
        }
    }
}

impl InferenceRulesOp for Gemm {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        if self.have_c {
            check_input_arity(&inputs, 3)?;
            s.equals(&inputs[2].datum_type, &outputs[0].datum_type)?;
        } else {
            check_input_arity(&inputs, 2)?;
        };
        s.equals(&inputs[0].rank, 2)?;
        s.equals(&inputs[1].rank, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].datum_type, &outputs[0].datum_type)?;
        let (ca, ra) = if self.trans_a { (0, 1) } else { (1, 0) };
        let (cb, rb) = if self.trans_b { (0, 1) } else { (1, 0) };
        s.equals(&inputs[0].shape[ra], &outputs[0].shape[0])?;
        s.equals(&inputs[0].shape[ca], &inputs[1].shape[rb])?;
        s.equals(&inputs[1].shape[cb], &outputs[0].shape[1])?;
        Ok(())
    }
}

#[derive(Debug, Clone, new)]
pub struct GemmUnaryA {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    b: Tensor,
    c: Tensor,
}

impl GemmUnaryA {
    fn eval_t<T: Datum + Float>(
        &self,
        mut inputs: TVec<SharedTensor>,
    ) -> TractResult<TVec<SharedTensor>>
    where
        f32: AsPrimitive<T>,
    {
        let a = args_1!(inputs);
        let a = a.to_array_view::<T>()?.into_dimensionality()?;
        let at = if self.trans_a { a.t() } else { a };
        let b = self.b.to_array_view::<T>()?.into_dimensionality()?;
        let bt = if self.trans_b { b.t() } else { b };
        let mut c = self
            .c
            .to_array_view::<T>()?
            .into_dimensionality()?
            .to_owned();
        ::ndarray::linalg::general_mat_mul(self.alpha.as_(), &at, &bt, self.beta.as_(), &mut c);
        Ok(tvec!(c.into()))
    }
}

impl Op for GemmUnaryA {
    fn name(&self) -> Cow<str> {
        "GemmUnaryA".into()
    }
}

impl StatelessOp for GemmUnaryA {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        dispatch_floatlike!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl InferenceRulesOp for GemmUnaryA {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        s.equals(&inputs[0].rank, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        let (ca, ra) = if self.trans_a { (0, 1) } else { (1, 0) };
        let (cb, rb) = if self.trans_b { (0, 1) } else { (1, 0) };
        s.equals(&inputs[0].shape[ra], &outputs[0].shape[0])?;
        s.equals(&inputs[0].shape[ca], self.b.shape()[rb].to_dim())?;
        s.equals(self.b.shape()[cb].to_dim(), &outputs[0].shape[1])?;
        Ok(())
    }
}

#[derive(Debug, Clone, new)]
pub struct GemmUnaryB {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    a: Tensor,
    c: Tensor,
}

impl GemmUnaryB {
    fn eval_t<T: Datum + Float>(
        &self,
        mut inputs: TVec<SharedTensor>,
    ) -> TractResult<TVec<SharedTensor>>
    where
        f32: AsPrimitive<T>,
    {
        let b = args_1!(inputs);
        let b = b.to_array_view::<T>()?.into_dimensionality()?;
        let a = self.a.to_array_view::<T>()?.into_dimensionality()?;
        let at = if self.trans_a { a.t() } else { a };
        let bt = if self.trans_b { b.t() } else { b };
        let mut c = self
            .c
            .to_array_view::<T>()?
            .into_dimensionality()?
            .to_owned();
        ::ndarray::linalg::general_mat_mul(self.alpha.as_(), &at, &bt, self.beta.as_(), &mut c);
        Ok(tvec!(c.into()))
    }
}

impl Op for GemmUnaryB {
    fn name(&self) -> Cow<str> {
        "GemmUnaryB".into()
    }
}

impl StatelessOp for GemmUnaryB {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        dispatch_floatlike!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl InferenceRulesOp for GemmUnaryB {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        s.equals(&inputs[0].rank, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].rank, 2)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        let (ca, ra) = if self.trans_a { (0, 1) } else { (1, 0) };
        let (cb, rb) = if self.trans_b { (0, 1) } else { (1, 0) };
        s.equals(self.a.shape()[ra].to_dim(), &outputs[0].shape[0])?;
        s.equals(self.a.shape()[ca].to_dim(), &inputs[0].shape[rb])?;
        s.equals(&inputs[0].shape[cb], &outputs[0].shape[1])?;
        Ok(())
    }
}
