use num_traits::{AsPrimitive, Zero};
use std::ops::{Add, Mul};

use crate::internal::*;
use ndarray::*;

use tract_linalg::mmm::{FusedSpec, MatMatMul};

fn eval_t<T: Copy + Datum + LinalgScalar + FloatLike>(
    a: &Tensor,
    b: &Tensor,
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
) -> TractResult<Tensor> {
    let a = a.to_array_view::<T>()?;
    let b = b.to_array_view::<T>()?;
    let geo = Geo::<T>::new(a.shape(), b.shape(), a_trans, b_trans, c_trans)?;
    let a = a.into_shape(&*geo.bc_a_shape)?;
    let b = b.into_shape(&*geo.bc_b_shape)?;
    let mut c = unsafe { Array::uninitialized(&*geo.c_shape) };

    let b_pack = geo.mm.b_pack();

    let mut pa = unsafe {
        Tensor::uninitialized_aligned::<T>(&[geo.mm.a_pack().len()], geo.mm.a_pack().alignment())?
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

        geo.mm.a_pack().pack(
            pa.as_ptr_mut()?,
            a.as_ptr(),
            a.strides()[prefix.ndim() + a_trans as usize],
            a.strides()[prefix.ndim() + !a_trans as usize],
        );
        b_pack.pack(
            pb.as_ptr_mut()?,
            b.as_ptr(),
            b.strides()[prefix.ndim() + b_trans as usize],
            b.strides()[prefix.ndim() + !b_trans as usize],
        );
        unsafe {
            geo.mm.run(
                &geo.mm.a_from_packed(pa.as_ptr()?),
                &geo.mm.b_from_packed(pb.as_ptr()?),
                &mut geo.mm.c_from_data_and_strides(
                    c.as_mut_ptr(),
                    c.strides()[prefix.ndim() + c_trans as usize],
                    c.strides()[prefix.ndim() + !c_trans as usize],
                ),
                &[],
            );
        }
    }
    Ok(c.into_tensor())
}

fn infer_shapes<D: DimLike>(
    mut ashape: TVec<D>,
    mut bshape: TVec<D>,
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
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
    let (mut m, mut ka) = (ashape[ashape.len() - 2].clone(), ashape[ashape.len() - 1].clone());
    let (mut kb, mut n) = (bshape[bshape.len() - 2].clone(), bshape[bshape.len() - 1].clone());
    if a_trans {
        std::mem::swap(&mut m, &mut ka);
    }
    if b_trans {
        std::mem::swap(&mut kb, &mut n);
    }
    assert_eq!(ka, kb);
    if c_trans {
        cshape.push(n);
        cshape.push(m);
    } else {
        cshape.push(m);
        cshape.push(n);
    }
    Ok((ashape, bshape, cshape))
}

#[derive(Debug, Clone)]
struct Geo<T: Copy + Datum + Add + Mul + Zero + FloatLike> {
    m: usize,
    k: usize,
    n: usize,
    mm: Box<dyn MatMatMul<T>>,
    a_shape: TVec<usize>,
    a_trans: bool,
    b_shape: TVec<usize>,
    b_trans: bool,
    bc_a_shape: TVec<usize>,
    bc_b_shape: TVec<usize>,
    c_shape: TVec<usize>,
    c_trans: bool,
    c_shape_prefix: TVec<usize>,
    a_stride_prefix: TVec<usize>,
    b_stride_prefix: TVec<usize>,
    c_stride_prefix: TVec<usize>,
}

impl<T: Copy + Datum + Add + Mul + Zero + FloatLike> Geo<T> {
    pub fn new(
        a_shape: &[usize],
        b_shape: &[usize],
        a_trans: bool,
        b_trans: bool,
        c_trans: bool,
    ) -> TractResult<Geo<T>> {
        let (bc_a_shape, bc_b_shape, bc_c_shape) =
            infer_shapes(a_shape.into(), b_shape.into(), a_trans, b_trans, c_trans)?;
        let m = bc_a_shape[bc_a_shape.len() - 2 + a_trans as usize];
        let k = bc_a_shape[bc_a_shape.len() - 1 - a_trans as usize];
        let n = bc_b_shape[bc_b_shape.len() - 1 - b_trans as usize];
        let mm = T::mmm(m, k, n);
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
            a_trans,
            b_trans,
            c_trans,
        })
    }
}

#[derive(Debug, Clone, new, Default)]
pub struct MatMul {
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
}

impl Op for MatMul {
    fn name(&self) -> Cow<str> {
        "MatMul".into()
    }

    fn cost(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let dt = inputs[0].datum_type;
        let (bc_a_shape, bc_b_shape, bc_c_shape) = infer_shapes(
            inputs[0].shape.iter().collect(),
            inputs[1].shape.iter().collect(),
            self.a_trans,
            self.b_trans,
            self.c_trans,
        )?;
        let mul = bc_c_shape.iter().rev().skip(2).cloned().product::<TDim>();
        let m = &bc_a_shape[bc_a_shape.len() - 2 + self.a_trans as usize];
        let k = &bc_a_shape[bc_a_shape.len() - 1 - self.a_trans as usize];
        let n = &bc_b_shape[bc_b_shape.len() - 1 - self.b_trans as usize];
        Ok(tvec!((Cost::FMA(dt), (mul * m * k * n))))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(ref a) = model.outlet_fact(node.inputs[0])?.konst {
            let patch = TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs[1..2],
                MatMulUnary::new(a.clone(), self.a_trans, self.b_trans, self.c_trans),
            )?;
            return Ok(Some(patch));
        } else if let Some(ref b) = model.outlet_fact(node.inputs[1])?.konst {
            let patch = TypedModelPatch::replace_single_op(
                model,
                node,
                &node.inputs[0..1],
                MatMulUnary::new(b.clone(), !self.b_trans, !self.a_trans, !self.c_trans),
            )?;
            return Ok(Some(patch));
        }
        Ok(None)
    }

    op_as_typed_op!();
}

impl StatelessOp for MatMul {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (a, b) = args_2!(inputs);
        let c = dispatch_floatlike!(self::eval_t(a.datum_type())(
            &*a,
            &*b,
            self.a_trans,
            self.b_trans,
            self.c_trans
        ))?;
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
            let (_, _, cshape) =
                infer_shapes(ashape, bshape, self.a_trans, self.b_trans, self.c_trans)?;
            s.equals(&outputs[0].shape, cshape)
        })?;
        Ok(())
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for MatMul {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(TypedTensorInfo::dt_shape(
            inputs[0].datum_type,
            &*infer_shapes(
                inputs[0].shape.to_tvec(),
                inputs[1].shape.to_tvec(),
                self.a_trans,
                self.b_trans,
                self.c_trans,
            )?
            .2
        )?))
    }
}

#[derive(Debug, Clone, new)]
pub struct MatMulUnary {
    a: Arc<Tensor>,
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
}

impl Op for MatMulUnary {
    fn name(&self) -> Cow<str> {
        "MatMulUnary".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![
            format!(
                "a_trans:{:?} b_trans:{:?} c_trans:{:?}",
                self.a_trans, self.b_trans, self.c_trans
            ),
            format!("A: {:?}", self.a),
        ])
    }

    fn axes_info(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<AxesInfo> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        if input_fact.shape.rank() != node.outputs[0].fact.shape.rank() {
            return Ok(AxesInfo::none());
        }
        let mut broadcasted_a_shape: TVec<_> = self.a.shape().into();
        while broadcasted_a_shape.len() < input_fact.shape.rank() {
            broadcasted_a_shape.insert(0, 1);
        }
        let mut invars = broadcasted_a_shape[..broadcasted_a_shape.len() - 2]
            .into_iter()
            .enumerate()
            .map(|(axis, &period)| AxisInfo::simple(axis).with_period(period))
            .collect::<Vec<_>>();
        if self.b_trans && self.c_trans {
            invars.push(AxisInfo::simple(input_fact.shape.rank() - 2))
        }
        if !self.b_trans && !self.c_trans {
            invars.push(AxisInfo::simple(input_fact.shape.rank() - 1))
        };
        Ok(invars.into_iter().collect())
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let b = args_1!(model.node_input_facts(node.id)?);
        if let Some(b_shape) = b.shape.as_finite() {
            let op = dispatch_floatlike!(self::new_mat_mul_unary_finite(b.datum_type)(
                self.a.clone(),
                b_shape,
                self.a_trans,
                self.b_trans,
                self.c_trans
            ))?;
            let patch = TypedModelPatch::replace_single_op(model, node, &node.inputs[0..1], op)?;
            return Ok(Some(patch));
        }
        Ok(None)
    }

    canonic!();
    op_as_typed_op!();
}

impl StatelessOp for MatMulUnary {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let b = args_1!(inputs);
        let c = dispatch_floatlike!(self::eval_t(b.datum_type())(
            &self.a,
            &*b,
            self.a_trans,
            self.b_trans,
            self.c_trans
        ))?;
        Ok(tvec!(c.into()))
    }
}

impl TypedOp for MatMulUnary {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(TypedTensorInfo::dt_shape(
            inputs[0].datum_type,
            &*infer_shapes(
                self.a.shape().into_iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
                inputs[0].shape.to_tvec(),
                self.a_trans,
                self.b_trans,
                self.c_trans,
            )?
            .2
        )?))
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
        let mut fact = target.outlet_fact(input)?.clone();
        if fact.axis >= fact.shape.len() - 1 {
            bail!("Can not pulsify MatMulUnaryA on the most inner dimension (k)");
        }
        let (_, _, cshape_pulse) =
            infer_shapes(fact.shape.clone(), self.a.shape().into(), false, false, false)?;
        let (_, _, cshape_full) = infer_shapes(
            fact.streaming_shape().into(),
            self.a.shape().iter().map(|d| d.to_dim()).collect(),
            false,
            false,
            false,
        )?;
        fact.shape = cshape_pulse;
        fact.dim = cshape_full[fact.axis].clone();
        let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
        Ok(tvec!(OutletId::new(id, 0)))
    }
}

#[derive(Debug, Clone)]
pub struct MatMulUnaryFinite<T>
where
    T: Copy + Datum + Add + Mul + Zero + FloatLike,
    f32: ::num_traits::AsPrimitive<T>,
{
    packed_as: ArrayD<Tensor>,
    geo: Geo<T>,
    non_linear: Vec<FusedSpec<T>>,
}

fn new_mat_mul_unary_finite<T>(
    a: Arc<Tensor>,
    b_shape: &[usize],
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
) -> TractResult<Box<dyn TypedOp>>
where
    T: Copy + Datum + Add + Mul + Zero + FloatLike,
    f32: ::num_traits::AsPrimitive<T>,
{
    let geo = Geo::<T>::new(a.shape(), b_shape, a_trans, b_trans, c_trans)?;
    let a = a.to_array_view::<T>()?;
    let a = a.into_shape(&*geo.bc_a_shape)?;
    let packed_as = Array::from_shape_fn(&a.shape()[0..a.ndim() - 2], |a_prefix| {
        let mut a = a.view();
        for x in a_prefix.slice() {
            a.index_axis_inplace(Axis(0), *x);
        }
        let mut pa = unsafe {
            Tensor::uninitialized_aligned::<T>(
                &[geo.mm.a_pack().len()],
                geo.mm.a_pack().alignment(),
            )
            .unwrap()
        };
        geo.mm.a_pack().pack(
            pa.as_ptr_mut().unwrap(),
            a.as_ptr(),
            a.strides()[a_trans as usize],
            a.strides()[!a_trans as usize],
        );
        pa
    });
    Ok(Box::new(MatMulUnaryFinite { packed_as, geo, non_linear: vec![] }))
}

impl<T> Op for MatMulUnaryFinite<T>
where
    T: Copy + Datum + Add + Mul + Zero + FloatLike,
    f32: AsPrimitive<T>,
{
    fn name(&self) -> Cow<str> {
        "MatMulUnaryFinite".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut infos = vec![format!(
            "a: {:?} m:{} k:{} n:{}",
            self.geo.a_shape, self.geo.m, self.geo.k, self.geo.n
        )];
        if self.non_linear.len() > 0 {
            infos.push(format!("{:?}", self.non_linear))
        }
        Ok(infos)
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops;
        if let Some(succ) = model.single_succ(node.id)? {
            let fused_micro_op = (|| -> TractResult<Option<TVec<FusedSpec<T>>>> {
                if let Some(op) = succ.op_as::<ops::binary::UnaryOp>() {
                    if op.a.shape() == &[self.geo.m] && self.geo.c_trans {
                        if op.mini_op.is::<ops::math::Mul>() {
                            return Ok(Some(tvec!(FusedSpec::PerRowMul(
                                op.a.as_slice::<T>()?.to_vec(),
                            ))));
                        } else if op.mini_op.is::<ops::math::Add>() {
                            return Ok(Some(tvec!(FusedSpec::PerRowAdd(
                                op.a.as_slice::<T>()?.to_vec(),
                            ))));
                        }
                    }
                } else if let Some(op) = succ.op_as::<ops::math::ScalarMax>() {
                    return Ok(Some(tvec!(FusedSpec::Max(op.max.as_()))));
                } else if let Some(op) = succ.op_as::<ops::math::ScalarMin>() {
                    return Ok(Some(tvec!(FusedSpec::Min(op.min.as_()))));
                } else if let Some(op) = succ.op_as::<ops::math::ScalarMinMax>() {
                    return Ok(Some(tvec!(
                        FusedSpec::Min(op.min.as_()),
                        FusedSpec::Max(op.max.as_()),
                    )));
                }
                Ok(None)
            })()?;
            if let Some(op) = fused_micro_op {
                let mut ops = self.non_linear.clone();
                ops.extend(op.into_iter());
                return Ok(Some(TypedModelPatch::fuse_with_next(
                    model,
                    &node,
                    Self { non_linear: ops, ..self.clone() },
                )?));
            }
        }
        Ok(None)
    }

    op_as_typed_op!();
}

impl<T> StatelessOp for MatMulUnaryFinite<T>
where
    T: Copy + Datum + Add + Mul + Zero + FloatLike,
    f32: ::num_traits::AsPrimitive<T>,
{
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let b = args_1!(inputs);
        let b = b.to_array_view::<T>()?;
        let mut c = unsafe { Array::uninitialized(&*self.geo.c_shape) };
        let b = b.into_shape(&*self.geo.bc_b_shape)?;
        if self.geo.n == 1 {
            for prefix in indices(&*self.geo.c_shape_prefix).into_iter() {
                let mut a = self.packed_as.view();
                let mut b = b.view();
                for &dim in prefix.slice() {
                    let d = dim.min(a.shape()[0] - 1);
                    a.index_axis_inplace(Axis(0), d);
                    let d = dim.min(b.shape()[0] - 1);
                    b.index_axis_inplace(Axis(0), d);
                    c.index_axis_inplace(Axis(0), dim);
                }
                debug_assert_eq!(a.ndim(), 0);
                debug_assert_eq!(b.ndim(), 2);
                debug_assert_eq!(c.ndim(), 2);
                let pa: &Tensor = a.iter().next().unwrap();
                unsafe {
                    self.geo.mm.run(
                        &self.geo.mm.a_from_packed(pa.as_ptr()?),
                        &self.geo.mm.b_vec_from_data_and_stride(
                            b.as_ptr(),
                            b.strides()[self.geo.b_trans as usize],
                        ),
                        &mut self.geo.mm.c_vec_from_data_and_stride(
                            c.as_mut_ptr(),
                            c.strides()[self.geo.c_trans as usize],
                        ),
                        &self.non_linear,
                    );
                }
            }
        } else {
            let b_pack = self.geo.mm.b_pack();
            let mut pb =
                unsafe { Tensor::uninitialized_aligned::<T>(&[b_pack.len()], b_pack.alignment())? };

            for prefix in indices(&*self.geo.c_shape_prefix).into_iter() {
                let mut a = self.packed_as.view();
                let mut b = b.view();
                let mut c = c.view_mut();
                for &dim in prefix.slice() {
                    let d = dim.min(a.shape()[0] - 1);
                    a.index_axis_inplace(Axis(0), d);
                    let d = dim.min(b.shape()[0] - 1);
                    b.index_axis_inplace(Axis(0), d);
                    c.index_axis_inplace(Axis(0), dim);
                }
                debug_assert_eq!(a.ndim(), 0);
                debug_assert_eq!(b.ndim(), 2);
                debug_assert_eq!(c.ndim(), 2);
                b_pack.pack(
                    pb.as_ptr_mut()?,
                    b.as_ptr(),
                    b.strides()[self.geo.b_trans as usize],
                    b.strides()[!self.geo.b_trans as usize],
                );
                let pa: &Tensor = a.iter().next().unwrap();
                unsafe {
                    self.geo.mm.run(
                        &self.geo.mm.a_from_packed(pa.as_ptr()?),
                        &self.geo.mm.b_from_packed(pb.as_ptr()?),
                        &mut self.geo.mm.c_from_data_and_strides(
                            c.as_mut_ptr(),
                            c.strides()[self.geo.c_trans as usize],
                            c.strides()[!self.geo.c_trans as usize],
                        ),
                        &self.non_linear,
                    );
                }
            }
        }

        Ok(tvec!(c.into_arc_tensor()))
    }
}

impl<T> TypedOp for MatMulUnaryFinite<T>
where
    T: Copy + Datum + Add + Mul + Zero + FloatLike,
    f32: ::num_traits::AsPrimitive<T>,
{
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(TypedTensorInfo::dt_shape(inputs[0].datum_type, &*self.geo.c_shape)?))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bin() {
        let a = rctensor2(&[[0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let b = rctensor2(&[[0f32], [1.0], [2.0]]);
        let c = rctensor2(&[[5f32], [14.0]]);
        let op = MatMul::new(false, false, false);
        let c_found = op.eval(tvec!(a, b)).unwrap().pop().unwrap();
        c.close_enough(&c_found, true).unwrap();
    }

    #[test]
    fn bin_transpose() {
        let a = rctensor2(&[[0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let b = rctensor2(&[[0f32], [1.0], [2.0]]);
        let c = rctensor2(&[[5f32], [14.0]]);
        let op = MatMul::new(true, true, true);
        let c_found = op.eval(tvec!(b, a)).unwrap().pop().unwrap();
        c.close_enough(&c_found, true).unwrap();
    }
}
