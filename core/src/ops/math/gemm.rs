use crate::internal::*;
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
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>>
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
            c.into_tensor().into_array::<T>()?.into_dimensionality::<Ix2>()?.to_owned()
        } else {
            c.to_array_view::<T>()?
                .broadcast(c_shape)
                .ok_or_else(|| format!("Incompatible broadcast: {:?} to {:?}", c.shape(), c_shape))?
                .to_owned()
        };
        ::ndarray::linalg::general_mat_mul(self.alpha.as_(), &at, &bt, self.beta.as_(), &mut c);
        Ok(tvec!(c.into_arc_tensor()))
    }

    fn eval_t_2<T: Datum + Float>(
        &self,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>>
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
        Ok(tvec!(c.into_arc_tensor()))
    }
}

impl Op for Gemm {
    fn name(&self) -> Cow<str> {
        "Gemm".into()
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if self.have_c {
            if let (Some(b), Some(c)) = (inputs[1].konst.clone(), inputs[2].konst.clone()) {
                return Ok(Some(TypedModelPatch::replace_single_op(
                    model,
                    node,
                    &[node.inputs[0]],
                    GemmUnaryA {
                        alpha: self.alpha,
                        beta: self.beta,
                        trans_a: self.trans_a,
                        trans_b: self.trans_b,
                        b: b.clone(),
                        c: c.clone(),
                    },
                )?));
            }

            if let (Some(a), Some(c)) = (inputs[0].konst.clone(), inputs[2].konst.clone()) {
                return Ok(Some(TypedModelPatch::replace_single_op(
                    model,
                    node,
                    &[node.inputs[1]],
                    GemmUnaryB {
                        alpha: self.alpha,
                        beta: self.beta,
                        trans_a: self.trans_a,
                        trans_b: self.trans_b,
                        a,
                        c,
                    },
                )?));
            }
        } else {
            if let Some(b) = inputs[1].konst.clone() {
                return Ok(Some(TypedModelPatch::replace_single_op(
                    model,
                    node,
                    &[node.inputs[0]],
                    GemmUnaryA {
                        alpha: self.alpha,
                        beta: 0.0,
                        trans_a: self.trans_a,
                        trans_b: self.trans_b,
                        b,
                        c: Tensor::from(0.0).into(),
                    },
                )?));
            }
            if let Some(a) = inputs[0].konst.clone() {
                return Ok(Some(TypedModelPatch::replace_single_op(
                    model,
                    node,
                    &[node.inputs[1]],
                    GemmUnaryB {
                        alpha: self.alpha,
                        beta: 0.0,
                        trans_a: self.trans_a,
                        trans_b: self.trans_b,
                        a,
                        c: Tensor::from(0.0).into(),
                    },
                )?));
            }
        }

        Ok(None)
    }
}

impl StatelessOp for Gemm {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
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

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for Gemm {
    typed_op_as_op!();

    fn output_facts(&self, inputs: TVec<&NormalizedTensorInfo>) -> TractResult<TVec<NormalizedTensorInfo>> {
        let cols = inputs[1].shape.dim(if self.trans_b { 1 } else { 0 });
        let rows = inputs[0].shape.dim(if self.trans_a { 0 } else { 1 });
        Ok(tvec!(NormalizedTensorInfo::dt_shape(inputs[0].datum_type, [rows, cols].as_ref())?))
    }
}

#[derive(Debug, Clone, new)]
pub struct GemmUnaryA {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    b: Arc<Tensor>,
    c: Arc<Tensor>,
}

impl GemmUnaryA {
    fn eval_t<T: Datum + Float>(
        &self,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>>
    where
        f32: AsPrimitive<T>,
    {
        let a = args_1!(inputs);
        let a = a.to_array_view::<T>()?.into_dimensionality()?;
        let at = if self.trans_a { a.t() } else { a };
        let b = self.b.to_array_view::<T>()?.into_dimensionality()?;
        let bt = if self.trans_b { b.t() } else { b };
        let c_shape = (at.rows(), bt.cols());
        let mut c = if self.beta != 0.0 {
            if self.c.shape() == &[c_shape.0, c_shape.1] {
                self.c.to_array_view::<T>()?.into_dimensionality()?.to_owned()
            } else {
                self.c
                    .to_array_view::<T>()?
                    .broadcast(c_shape)
                    .ok_or_else(|| {
                        format!("Incompatible broadcast: {:?} to {:?}", self.c.shape(), c_shape)
                    })?
                    .to_owned()
            }
        } else {
            unsafe { Array::uninitialized((c_shape.0, c_shape.1)) }
        };
        ::ndarray::linalg::general_mat_mul(self.alpha.as_(), &at, &bt, self.beta.as_(), &mut c);
        Ok(tvec!(c.into_arc_tensor()))
    }
}

impl Op for GemmUnaryA {
    fn name(&self) -> Cow<str> {
        "GemmUnaryA".into()
    }
}

impl StatelessOp for GemmUnaryA {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        dispatch_floatlike!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl TypedOp for GemmUnaryA {
    typed_op_as_op!();

    fn output_facts(&self, inputs: TVec<&NormalizedTensorInfo>) -> TractResult<TVec<NormalizedTensorInfo>> {
        let cols = self.b.shape()[if self.trans_b { 1 } else { 0 }].to_dim();
        let rows = inputs[0].shape.dim(if self.trans_a { 0 } else { 1 });
        Ok(tvec!(NormalizedTensorInfo::dt_shape(inputs[0].datum_type, [rows, cols].as_ref())?))
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let fact = target.outlet_fact(input)?.clone();
        let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
        Ok(tvec!(OutletId::new(id, 0)))
    }
}

#[derive(Debug, Clone, new)]
pub struct GemmUnaryB {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    a: Arc<Tensor>,
    c: Arc<Tensor>,
}

impl GemmUnaryB {
    fn eval_t<T: Datum + Float>(
        &self,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>>
    where
        f32: AsPrimitive<T>,
    {
        let b = args_1!(inputs);
        let b = b.to_array_view::<T>()?.into_dimensionality()?;
        let a = self.a.to_array_view::<T>()?.into_dimensionality()?;
        let at = if self.trans_a { a.t() } else { a };
        let bt = if self.trans_b { b.t() } else { b };
        let c_shape = (at.rows(), bt.cols());
        let mut c = if self.beta != 0.0 {
            if self.c.shape() == &[c_shape.0, c_shape.1] {
                self.c.to_array_view::<T>()?.into_dimensionality()?.to_owned()
            } else {
                self.c
                    .to_array_view::<T>()?
                    .broadcast(c_shape)
                    .ok_or_else(|| {
                        format!("Incompatible broadcast: {:?} to {:?}", self.c.shape(), c_shape)
                    })?
                    .to_owned()
            }
        } else {
            unsafe { Array::uninitialized((c_shape.0, c_shape.1)) }
        };
        ::ndarray::linalg::general_mat_mul(self.alpha.as_(), &at, &bt, self.beta.as_(), &mut c);
        Ok(tvec!(c.into_arc_tensor()))
    }
}

impl Op for GemmUnaryB {
    fn name(&self) -> Cow<str> {
        "GemmUnaryB".into()
    }
}

impl StatelessOp for GemmUnaryB {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        dispatch_floatlike!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl TypedOp for GemmUnaryB {
    typed_op_as_op!();

    fn output_facts(&self, inputs: TVec<&NormalizedTensorInfo>) -> TractResult<TVec<NormalizedTensorInfo>> {
        let cols = inputs[0].shape.dim(if self.trans_b { 1 } else { 0 });
        let rows = self.a.shape()[if self.trans_a { 0 } else { 1 }].to_dim();
        Ok(tvec!(NormalizedTensorInfo::dt_shape(inputs[0].datum_type, [rows, cols].as_ref())?))
    }
}
