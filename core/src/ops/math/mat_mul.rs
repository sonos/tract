use num_traits::Zero;
use std::ops::{Add, Mul};

use crate::internal::*;
use ndarray::*;

fn eval_t<T: Copy + Datum + LinalgScalar + FloatLike>(
    a: &Tensor,
    b: &Tensor,
) -> TractResult<Tensor> {
    let a = a.to_array_view::<T>()?;
    let b = b.to_array_view::<T>()?;
    let geo = Geo::<T>::new(a.shape(), b.shape())?;
    let a = a.into_shape(&*geo.bc_a_shape)?;
    let b = b.into_shape(&*geo.bc_b_shape)?;
    let mut c = unsafe { Array::uninitialized(&*geo.c_shape) };

    let b_pack = geo.mm.b_pack();

    let mut pa = unsafe {
        Tensor::uninitialized_aligned::<T>(&[geo.mm.packed_a_len()], geo.mm.packed_a_alignment())?
    };
    let mut pb =
        unsafe { Tensor::uninitialized_aligned::<T>(&[b_pack.len()], b_pack.alignment())? };

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
        b_pack.pack(
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
    Ok(c.into_tensor())
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
struct Geo<T: Copy + Datum + Add + Mul + Zero + FloatLike> {
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

impl<T: Copy + Datum + Add + Mul + Zero + FloatLike> Geo<T> {
    pub fn new(a_shape: &[usize], b_shape: &[usize]) -> TractResult<Geo<T>> {
        let (bc_a_shape, bc_b_shape, bc_c_shape) = infer_shapes(a_shape.into(), b_shape.into())?;
        let m = bc_a_shape[bc_a_shape.len() - 2];
        let k = bc_a_shape[bc_a_shape.len() - 1];
        let n = bc_b_shape[bc_b_shape.len() - 1];
        let mm = T::packed_mat_mul(m, k, n);
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

    fn cost(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let dt = inputs[0].datum_type;
        let (bc_a_shape, bc_b_shape, bc_c_shape) =
            infer_shapes(inputs[0].shape.iter().collect(), inputs[1].shape.iter().collect())?;
        let mul = bc_c_shape.iter().rev().skip(2).cloned().product::<TDim>();
        let m = bc_a_shape[bc_a_shape.len() - 2];
        let k = bc_a_shape[bc_a_shape.len() - 1];
        let n = bc_b_shape[bc_b_shape.len() - 1];
        Ok(tvec!((Cost::FMA(dt), (mul * m * k * n))))
    }
}

impl StatelessOp for MatMul {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (a, b) = args_2!(inputs);
        let c = dispatch_floatlike!(self::eval_t(a.datum_type())(&*a, &*b))?;
        Ok(tvec!(c.into_arc_tensor()))
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

    inference_op_as_op!();
}

#[derive(Debug, Clone, new)]
pub struct MatMulUnaryA {
    b: Tensor,
}

impl MatMulUnaryA {
    pub fn codegen<T: Copy + Datum + Add + Mul + Zero + FloatLike>(
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

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let mut fact = target.outlet_fact(input)?.clone();
        if fact.axis >= fact.shape.len() - 1 {
            bail!("Can not pulsify MatMulUnaryA on the most inner dimension (k)");
        }
        let (_, _, cshape_pulse) = infer_shapes(fact.shape.clone(), self.b.shape().into())?;
        let (_, _, cshape_full) = infer_shapes(
            fact.streaming_shape().into(),
            self.b.shape().iter().map(|d| d.to_dim()).collect(),
        )?;
        fact.shape = cshape_pulse;
        fact.dim = cshape_full[fact.axis];
        let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
        Ok(tvec!(OutletId::new(id, 0)))
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if let Some(a_shape) = inputs[0].shape.as_finite() {
            let dt = inputs[0].datum_type;
            if let Some(op) = dispatch_floatlike!(Self::codegen(dt)(self, &*a_shape))? {
                return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
            }
        }
        Ok(None)
    }
}

impl StatelessOp for MatMulUnaryA {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let a = args_1!(inputs);
        let c = dispatch_floatlike!(self::eval_t(a.datum_type())(&*a, &self.b))?;
        Ok(tvec!(c.into()))
    }
}

#[derive(Debug, Clone)]
pub struct MatMulUnaryImplASimpleB<T: Copy + Datum + Add + Mul + Zero + FloatLike> {
    geo: Geo<T>,
    packed_b: Tensor,
    a_shape: TVec<usize>,
    c_shape: TVec<usize>,
}

impl<T: Copy + Datum + Add + Mul + Zero + FloatLike> MatMulUnaryImplASimpleB<T> {
    pub fn new(a_shape: &[usize], b: &ArrayViewD<T>) -> TractResult<MatMulUnaryImplASimpleB<T>> {
        assert_eq!(b.ndim(), 2);
        let geo_ext = Geo::<T>::new(a_shape, b.shape())?;
        let c_shape = geo_ext.c_shape.into();

        let a_len = a_shape.iter().cloned().product::<usize>();
        let shape_a_internal = [a_len / geo_ext.k, geo_ext.k];
        let geo = Geo::new(&shape_a_internal, b.shape())?;
        let b_pack = geo.mm.b_pack();
        let mut packed_b =
            unsafe { Tensor::uninitialized_aligned::<T>(&[b_pack.len()], b_pack.alignment())? };
        b_pack.pack(packed_b.as_ptr_mut()?, b.as_ptr(), b.strides()[0], b.strides()[1]);
        Ok(MatMulUnaryImplASimpleB { geo, packed_b, c_shape, a_shape: a_shape.into() })
    }
}

impl<T: Copy + Datum + Add + Mul + Zero + FloatLike> Op for MatMulUnaryImplASimpleB<T> {
    fn name(&self) -> Cow<str> {
        "MatMulUnaryImplASimpleB".into()
    }

    fn info(&self) -> TractResult<Option<String>> {
        Ok(Some(format!("{:?}", self.geo.mm)))
    }

    fn cost(&self, _inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        Ok(tvec!((
            Cost::FMA(T::datum_type()),
            (self.geo.mm.m() * self.geo.mm.n() * self.geo.mm.k()).to_dim()
        )))
    }
}

impl<T: Copy + Datum + Add + Mul + Zero + FloatLike> StatelessOp for MatMulUnaryImplASimpleB<T> {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
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

        Ok(tvec!(c.into_arc_tensor()))
    }
}

#[derive(Debug, Clone)]
pub struct MatMulUnaryImplA<T: Copy + Datum + Add + Mul + Zero + FloatLike> {
    geo: Geo<T>,
    packed_bs: Tensor,
}

impl<T: Copy + Datum + Add + Mul + Zero + FloatLike> MatMulUnaryImplA<T> {
    pub fn new(a_shape: &[usize], b: &ArrayViewD<T>) -> TractResult<MatMulUnaryImplA<T>> {
        let geo = Geo::new(a_shape, b.shape())?;
        let b_pack = geo.mm.b_pack();
        let packed_b_len = b_pack.len();
        let mut packed_bs_shape = geo.bc_b_shape.clone();
        packed_bs_shape.pop();
        packed_bs_shape.pop();
        packed_bs_shape.push(packed_b_len);
        let mut packed_bs =
            unsafe { Tensor::uninitialized_aligned::<T>(&packed_bs_shape, b_pack.alignment())? };
        for (ix, prefix) in indices(&geo.b_shape[..geo.b_shape.len() - 2]).into_iter().enumerate() {
            let mut b = b.view();
            for (axis, &dim) in prefix.slice().iter().enumerate() {
                b.slice_axis_inplace(Axis(axis), (dim..=dim).into());
            }
            unsafe {
                b_pack.pack(
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

impl<T: Copy + Datum + Add + Mul + Zero + FloatLike> Op for MatMulUnaryImplA<T> {
    fn name(&self) -> Cow<str> {
        "MatMulUnaryImplA".into()
    }

    fn info(&self) -> TractResult<Option<String>> {
        Ok(Some(format!("{:?}", self.geo.mm)))
    }

    fn cost(&self, _inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let mul = self.geo.c_shape_prefix.iter().product::<usize>();
        Ok(tvec!((
            Cost::FMA(T::datum_type()),
            (self.geo.mm.m() * self.geo.mm.n() * self.geo.mm.k() * mul).to_dim()
        )))
    }
}

impl<T: Copy + Datum + Add + Mul + Zero + FloatLike> StatelessOp for MatMulUnaryImplA<T> {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
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
        Ok(tvec!(c.into_arc_tensor()))
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
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let b = args_1!(inputs);
        let c = dispatch_floatlike!(self::eval_t(b.datum_type())(&self.a, &*b))?;
        Ok(tvec!(c.into()))
    }
}
