use num_traits::Zero;
use std::ops::{Add, Mul};

use crate::ops::prelude::*;
use ndarray::*;

fn eval_t<T: Copy + Datum + LinalgScalar>(a: &Tensor, b: &Tensor) -> TractResult<Tensor> {
    let a = a.to_array_view::<T>()?;
    let b = b.to_array_view::<T>()?;
    let geo = Geo::<T>::new(a.shape(), b.shape())?;
    let a = a.into_shape(&*geo.bc_a_shape)?;
    let b = b.into_shape(&*geo.bc_b_shape)?;
    let mut c = unsafe { Array::uninitialized(&*geo.c_shape) };

    let mut pa = unsafe {
        Tensor::uninitialized_aligned::<T>(&[geo.mm.packed_a_len()], geo.mm.packed_a_alignment())?
    };
    let mut pb = unsafe {
        Tensor::uninitialized_aligned::<T>(&[geo.mm.packed_b_len()], geo.mm.packed_b_alignment())?
    };

    for prefix in indices(&*geo.c_shape_prefix).into_iter() {
        let mut a = a.view();
        let mut b = b.view();
        let mut c = c.view_mut();
        for (axis, &dim) in prefix.slice().iter().enumerate() {
            let d = dim.min(a.shape()[axis] - 1);
            a.slice_axis_inplace(Axis(axis), (d..=d).into());
            let d = dim.min(b.shape()[axis] - 1);
            b.slice_axis_inplace(Axis(axis), (d..=d).into());
            c.slice_axis_inplace(Axis(axis), (dim..=dim).into());
        }

        geo.mm.pack_a(
            pa.as_ptr_mut()?,
            a.as_ptr(),
            a.strides()[prefix.ndim()],
            a.strides()[prefix.ndim() + 1],
        );
        geo.mm.pack_b(
            pb.as_ptr_mut()?,
            b.as_ptr(),
            b.strides()[prefix.ndim()],
            b.strides()[prefix.ndim() + 1],
        );
        geo.mm.mat_mul_prepacked(
            pa.as_ptr()?,
            pb.as_ptr()?,
            c.as_mut_ptr(),
            c.strides()[prefix.ndim()],
            c.strides()[prefix.ndim() + 1],
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
    let cshape_prefix = crate::broadcast::multi_broadcast(&[
        &ashape[..(ashape.len() - 2)],
        &bshape[..(bshape.len() - 2)],
    ])
    .ok_or("Could not broadcast")?;
    let mut cshape: TVec<D> = cshape_prefix.clone();
    cshape.push(ashape[ashape.len() - 2]);
    cshape.push(bshape[bshape.len() - 1]);
    Ok((ashape, bshape, cshape))
}

#[derive(Debug, Clone)]
struct Geo<T: Copy + Datum + Add + Mul + Zero> {
    m: usize,
    k: usize,
    n: usize,
    mm: Box<tract_linalg::MatMul<T>>,
    a_shape: TVec<usize>,
    b_shape: TVec<usize>,
    bc_a_shape: TVec<usize>,
    bc_b_shape: TVec<usize>,
    c_shape: TVec<usize>,
    c_shape_prefix: TVec<usize>,
    a_stride_prefix: TVec<usize>,
    b_stride_prefix: TVec<usize>,
    c_stride_prefix: TVec<usize>,
}

impl<T: Copy + Datum + Add + Mul + Zero> Geo<T> {
    pub fn new(a_shape: &[usize], b_shape: &[usize]) -> TractResult<Geo<T>> {
        let (bc_a_shape, bc_b_shape, bc_c_shape) = infer_shapes(a_shape.into(), b_shape.into())?;
        let m = bc_a_shape[bc_a_shape.len() - 2];
        let k = bc_a_shape[bc_a_shape.len() - 1];
        let n = bc_b_shape[bc_b_shape.len() - 1];
        let mm = T::packed_mat_mul(m, k, n).ok_or_else(|| {
            format!("Can not perfom matmul on {:?} (not a linear algebra type)", T::datum_type())
        })?;
        let a_stride_prefix = bc_a_shape
            .iter()
            .rev()
            .scan(1, |stride, dim| {
                let s = Some(*stride);
                *stride *= dim;
                s
            })
            .skip(2)
            .collect();
        let b_stride_prefix = bc_b_shape
            .iter()
            .rev()
            .scan(1, |stride, dim| {
                let s = Some(*stride);
                *stride *= dim;
                s
            })
            .skip(2)
            .collect();
        let c_stride_prefix = bc_c_shape
            .iter()
            .rev()
            .scan(1, |stride, dim| {
                let s = Some(*stride);
                *stride *= dim;
                s
            })
            .skip(2)
            .collect();
        Ok(Geo {
            m,
            k,
            n,
            mm,
            c_shape_prefix: bc_c_shape[0..(bc_c_shape.len() - 2)].into(),
            bc_a_shape,
            bc_b_shape,
            a_shape: a_shape.into(),
            b_shape: b_shape.into(),
            c_shape: bc_c_shape.into(),
            a_stride_prefix,
            b_stride_prefix,
            c_stride_prefix,
        })
    }
}

#[derive(Debug, Clone, new, Default)]
pub struct MatMul {}

impl Op for MatMul {
    fn name(&self) -> Cow<str> {
        "MatMul".into()
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
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].datum_type, &outputs[0].datum_type)?;
        s.given_2(&inputs[0].shape, &inputs[1].shape, move |s, ashape, bshape| {
            let (_, _, cshape) = infer_shapes(ashape, bshape)?;
            s.equals(&outputs[0].shape, cshape)
        })?;
        Ok(())
    }
}

#[derive(Debug, Clone, new)]
pub struct MatMulUnaryA {
    b: Tensor,
}

impl MatMulUnaryA {
    pub fn codegen<T: Copy + Datum + Add + Mul + Zero>(
        &self,
        a_shape: &[usize],
    ) -> TractResult<Option<Box<Op>>> {
        if self.b.shape().len() == 2 {
            return Ok(Some(Box::new(MatMulUnaryImplASimpleB::<T>::new(
                a_shape,
                &self.b.to_array_view()?,
            )?)));
        } else {
            return Ok(Some(Box::new(MatMulUnaryImplA::<T>::new(
                a_shape,
                &self.b.to_array_view()?,
            )?)));
        }
    }
}

impl Op for MatMulUnaryA {
    fn name(&self) -> Cow<str> {
        "MatMulUnaryA".into()
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

    fn codegen(&self, model: &Model, node: &Node) -> TractResult<Option<ModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if let (Some(a_shape), Some(dt)) =
            (inputs[0].shape.as_concrete_finite()?, inputs[0].datum_type.concretize())
        {
            if let Some(op) = dispatch_floatlike!(Self::codegen(dt)(self, &*a_shape))? {
                return Ok(Some(ModelPatch::single_unary_op(model, node, op)?));
            }
        }
        Ok(None)
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
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.given(&inputs[0].shape, move |s, ashape| {
            let bshape: TVec<TDim> = self.b.shape().iter().map(|x| x.to_dim()).collect();
            let (_, _, cshape) = infer_shapes(ashape, bshape)?;
            s.equals(&outputs[0].shape, cshape)
        })?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MatMulUnaryImplASimpleB<T: Copy + Datum + Add + Mul + Zero> {
    geo: Geo<T>,
    packed_b: Tensor,
    a_shape: TVec<usize>,
    c_shape: TVec<usize>,
}

impl<T: Copy + Datum + Add + Mul + Zero> MatMulUnaryImplASimpleB<T> {
    pub fn new(a_shape: &[usize], b: &ArrayViewD<T>) -> TractResult<MatMulUnaryImplASimpleB<T>> {
        assert_eq!(b.ndim(), 2);
        let geo_ext = Geo::<T>::new(a_shape, b.shape())?;
        let c_shape = geo_ext.c_shape.into();

        let a_len = a_shape.iter().cloned().product::<usize>();
        let shape_a_internal = [a_len / geo_ext.k, geo_ext.k];
        let geo = Geo::new(&shape_a_internal, b.shape())?;
        let packed_b_len = geo.mm.packed_b_len();
        let mut packed_b = unsafe {
            Tensor::uninitialized_aligned::<T>(&[packed_b_len], geo.mm.packed_b_alignment())?
        };
        geo.mm.pack_b(packed_b.as_ptr_mut()?, b.as_ptr(), b.strides()[0], b.strides()[1]);
        Ok(MatMulUnaryImplASimpleB { geo, packed_b, c_shape, a_shape: a_shape.into() })
    }
}

impl<T: Copy + Datum + Add + Mul + Zero> Op for MatMulUnaryImplASimpleB<T> {
    fn name(&self) -> Cow<str> {
        "MatMulUnaryImplASimpleB".into()
    }

    fn info(&self) -> TractResult<Option<String>> {
        Ok(Some(format!("{:?}", self.geo.mm)))
    }
}

impl<T: Copy + Datum + Add + Mul + Zero> StatelessOp for MatMulUnaryImplASimpleB<T> {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let a = args_1!(inputs);
        let a = a.to_array_view::<T>()?;

        let mut c = unsafe { Array::uninitialized(&*self.c_shape) };

        let mut pa = unsafe {
            Tensor::uninitialized_aligned::<T>(
                &[self.geo.mm.packed_a_len()],
                self.geo.mm.packed_a_alignment(),
            )?
        };

        self.geo.mm.pack_a(pa.as_ptr_mut()?, a.as_ptr(), self.geo.k as isize, 1);
        self.geo.mm.mat_mul_prepacked(
            pa.as_ptr()?,
            self.packed_b.as_ptr()?,
            c.as_mut_ptr(),
            self.geo.n as isize,
            1,
        );

        Ok(tvec!(c.into()))
    }
}

impl<T: Copy + Datum + Add + Mul + Zero> InferenceRulesOp for MatMulUnaryImplASimpleB<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, T::datum_type())?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, ShapeFact::from(&*self.a_shape))?;
        s.equals(&outputs[0].shape, ShapeFact::from(&*self.c_shape))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MatMulUnaryImplA<T: Copy + Datum + Add + Mul + Zero> {
    geo: Geo<T>,
    packed_bs: Tensor,
}

impl<T: Copy + Datum + Add + Mul + Zero> MatMulUnaryImplA<T> {
    pub fn new(a_shape: &[usize], b: &ArrayViewD<T>) -> TractResult<MatMulUnaryImplA<T>> {
        let geo = Geo::new(a_shape, b.shape())?;
        let packed_b_len = geo.mm.packed_b_len();
        let mut packed_bs_shape = geo.bc_b_shape.clone();
        packed_bs_shape.pop();
        packed_bs_shape.pop();
        packed_bs_shape.push(packed_b_len);
        let mut packed_bs = unsafe {
            Tensor::uninitialized_aligned::<T>(&packed_bs_shape, geo.mm.packed_b_alignment())?
        };
        for (ix, prefix) in indices(&geo.b_shape[..geo.b_shape.len() - 2]).into_iter().enumerate() {
            let mut b = b.view();
            for (axis, &dim) in prefix.slice().iter().enumerate() {
                b.slice_axis_inplace(Axis(axis), (dim..=dim).into());
            }
            unsafe {
                geo.mm.pack_b(
                    packed_bs.as_ptr_mut::<T>()?.offset((ix * packed_b_len) as isize),
                    b.as_ptr(),
                    b.strides()[prefix.ndim()],
                    b.strides()[prefix.ndim() + 1],
                );
            }
        }
        Ok(MatMulUnaryImplA { geo, packed_bs })
    }
}

impl<T: Copy + Datum + Add + Mul + Zero> Op for MatMulUnaryImplA<T> {
    fn name(&self) -> Cow<str> {
        "MatMulUnaryImplA".into()
    }

    fn info(&self) -> TractResult<Option<String>> {
        Ok(Some(format!("{:?}", self.geo.mm)))
    }
}

impl<T: Copy + Datum + Add + Mul + Zero> StatelessOp for MatMulUnaryImplA<T> {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let a = args_1!(inputs);
        let a = a.to_array_view::<T>()?.into_shape(&*self.geo.bc_a_shape)?;

        let mut c = unsafe { Array::uninitialized(&*self.geo.c_shape) };

        let mut pa = unsafe {
            Tensor::uninitialized_aligned::<T>(
                &[self.geo.mm.packed_a_len()],
                self.geo.mm.packed_a_alignment(),
            )?
        };

        for prefix in indices(&*self.geo.c_shape_prefix).into_iter() {
            let mut a = a.view();
            let mut b = self.packed_bs.to_array_view::<T>()?;
            let mut c = c.view_mut();
            for (axis, &dim) in prefix.slice().iter().enumerate() {
                let d = dim.min(a.shape()[axis] - 1);
                a.slice_axis_inplace(Axis(axis), (d..=d).into());
                let d = dim.min(b.shape()[axis] - 1);
                b.slice_axis_inplace(Axis(axis), (d..=d).into());
                c.slice_axis_inplace(Axis(axis), (dim..=dim).into());
            }

            self.geo.mm.pack_a(
                pa.as_ptr_mut()?,
                a.as_ptr(),
                a.strides()[prefix.ndim()],
                a.strides()[prefix.ndim() + 1],
            );
            self.geo.mm.mat_mul_prepacked(
                pa.as_ptr_mut()?,
                b.as_ptr(),
                c.as_mut_ptr(),
                c.strides()[prefix.ndim()],
                c.strides()[prefix.ndim() + 1],
            );
        }
        Ok(tvec!(c.into()))
    }
}

impl<T: Copy + Datum + Add + Mul + Zero> InferenceRulesOp for MatMulUnaryImplA<T> {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, T::datum_type())?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, ShapeFact::from(&*self.geo.a_shape))?;
        s.equals(&outputs[0].shape, ShapeFact::from(&*self.geo.c_shape))?;
        Ok(())
    }
}

#[derive(Debug, Clone, new)]
pub struct MatMulUnaryB {
    a: Tensor,
}

impl Op for MatMulUnaryB {
    fn name(&self) -> Cow<str> {
        "MatMulUnaryB".into()
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
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.given(&inputs[0].shape, move |s, bshape| {
            let ashape: TVec<TDim> = self.a.shape().iter().map(|x| x.to_dim()).collect();
            let (_, _, cshape) = infer_shapes(ashape, bshape)?;
            s.equals(&outputs[0].shape, cshape)
        })?;
        Ok(())
    }
}
