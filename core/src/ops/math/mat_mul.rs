use ndarray::*;
use ops::prelude::*;

fn make_slicer(
    coords: &[usize],
    shape: &[usize],
) -> TractResult<SliceInfo<Vec<SliceOrIndex>, IxDyn>> {
    let mut slice: Vec<SliceOrIndex> = coords
        .iter()
        .zip(shape.iter())
        .map(|(&c, &d)| {
            let a = if c < d { c } else { 0 };
            SliceOrIndex::Index(a as _)
        }).collect();
    slice.push(SliceOrIndex::from(..));
    slice.push(SliceOrIndex::from(..));
    Ok(SliceInfo::new(slice)?)
}

fn eval_t<T: Datum + LinalgScalar>(a: &Tensor, b: &Tensor) -> TractResult<Tensor> {
    let a = a.to_array_view::<T>()?;
    let b = b.to_array_view::<T>()?;
    let (ashape, bshape, cshape) = infer_shapes(a.shape().into(), b.shape().into())?;
    let a = a.into_shape(&*ashape)?;
    let b = b.into_shape(&*bshape)?;
    let mut c = unsafe { Array::uninitialized(&*cshape) };

    for prefix in indices(&cshape[..(cshape.len() - 2)]).into_iter() {
        let a_slice = make_slicer(&prefix.slice(), a.shape())?;
        let b_slice = make_slicer(&prefix.slice(), b.shape())?;
        let c_slice = make_slicer(&prefix.slice(), &*cshape)?;
        let a1: ArrayViewD<T> = a.slice(&a_slice.as_ref());
        let b1: ArrayViewD<T> = b.slice(&b_slice.as_ref());
        let c1: ArrayViewMutD<T> = c.slice_mut(&c_slice.as_ref());

        linalg::general_mat_mul(
            T::one(),
            &a1.into_dimensionality()?,
            &b1.into_dimensionality()?,
            T::zero(),
            &mut c1.into_dimensionality()?,
        );
    }
    Ok(c.into())
}

fn infer_shapes<D: DimLike>(
    mut ashape: TVec<D>,
    mut bshape: TVec<D>,
) -> TractResult<(TVec<D>, TVec<D>, TVec<D>)> {
    if ashape.len() < 2 {
        ashape.insert(0, D::one());
    }
    if bshape.len() < 2 {
        bshape.push(D::one());
    }
    while ashape.len() < bshape.len() {
        ashape.insert(0, D::one());
    }
    while bshape.len() < ashape.len() {
        bshape.insert(0, D::one());
    }
    let cshape_prefix = ::broadcast::multi_broadcast(&[
        &ashape[..(ashape.len() - 2)],
        &bshape[..(bshape.len() - 2)],
    ]).ok_or("Could not broadcast")?;
    let mut cshape: TVec<D> = cshape_prefix.clone();
    cshape.push(ashape[ashape.len() - 2]);
    cshape.push(bshape[bshape.len() - 1]);
    Ok((ashape, bshape, cshape))
}

#[derive(Debug, Clone, new, Default)]
pub struct MatMul {}

impl Op for MatMul {
    fn name(&self) -> &str {
        "MatMul"
    }
}

impl StatelessOp for MatMul {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (a, b) = args_2!(inputs);
        let c = dispatch_floatlike!(self::eval_t(a.datum_type())(a.as_tensor(), b.as_tensor()))?;
        Ok(tvec!(c.into()))
    }
}

impl InferenceRulesOp for MatMul {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 2)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].datum_type, &outputs[0].datum_type)?;
        s.given_2(
            &inputs[0].shape,
            &inputs[1].shape,
            move |s, ashape, bshape| {
                let (_, _, cshape) = infer_shapes(ashape, bshape)?;
                s.equals(&outputs[0].shape, cshape)
            },
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, new)]
pub struct MatMulUnaryA {
    b: Tensor,
}

impl Op for MatMulUnaryA {
    fn name(&self) -> &str {
        "MatMulUnaryA"
    }

    fn pulsify(&self, mut inputs: TVec<&PulsedTensorFact>) -> TractResult<Vec<PulsifiedOp>> {
        let input = args_1!(inputs);
        if input.axis >= input.shape.len() - 1 {
            bail!("Can not pulsify MatMulUnaryA on the most inner dimension (k)");
        }
        let (_, _, cshape_pulse) = infer_shapes(input.shape.clone(), self.b.shape().into())?;
        let (_, _, cshape_full) = infer_shapes(
            input.streaming_shape().into(),
            self.b.shape().iter().map(|d| d.to_dim()).collect(),
        )?;
        let mut fact = input.clone();
        fact.shape = cshape_pulse;
        fact.dim = cshape_full[fact.axis];
        Ok(vec![PulsifiedOp::new(Box::new(self.clone()), tvec!(fact))])
    }
}

impl StatelessOp for MatMulUnaryA {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let a = args_1!(inputs);
        let c = dispatch_floatlike!(self::eval_t(a.datum_type())(a.as_tensor(), &self.b))?;
        Ok(tvec!(c.into()))
    }
}

impl InferenceRulesOp for MatMulUnaryA {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.given(&inputs[0].shape, move |s, ashape| {
            let bshape: TVec<TDim> = self.b.shape().iter().map(|x| x.to_dim()).collect();
            let (_, _, cshape) = infer_shapes(ashape, bshape)?;
            s.equals(&outputs[0].shape, cshape)
        })?;
        Ok(())
    }
}

#[derive(Debug, Clone, new)]
pub struct MatMulUnaryB {
    a: Tensor,
}

impl Op for MatMulUnaryB {
    fn name(&self) -> &str {
        "MatMulUnaryB"
    }
}

impl StatelessOp for MatMulUnaryB {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let b = args_1!(inputs);
        let c = dispatch_floatlike!(self::eval_t(b.datum_type())(&self.a, b.as_tensor()))?;
        Ok(tvec!(c.into()))
    }
}

impl InferenceRulesOp for MatMulUnaryB {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.given(&inputs[0].shape, move |s, bshape| {
            let ashape: TVec<TDim> = self.a.shape().iter().map(|x| x.to_dim()).collect();
            let (_, _, cshape) = infer_shapes(ashape, bshape)?;
            s.equals(&outputs[0].shape, cshape)
        })?;
        Ok(())
    }
}
