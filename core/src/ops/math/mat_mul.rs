use num_traits::Zero;
use std::fmt;
use std::ops::{Add, Deref, Mul};

use crate::internal::*;
use crate::ops::math::mat_mat_mul::{MatMatMulPackB, MatMatMulUnaryFinite};
use ndarray::*;

use tract_linalg::mmm::MatMatMul;
use tract_linalg::mmm::QMatMatMul;

fn eval(
    a: &Tensor,
    b: &Tensor,
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
    zero_point_a: Option<&Tensor>,
    zero_point_b: Option<&Tensor>,
) -> TractResult<TVec<Arc<Tensor>>> {
    let c = if (a.datum_type(), b.datum_type()) == (f32::datum_type(), f32::datum_type()) {
        eval_t(a, b, a_trans, b_trans, c_trans, zero_point_a, zero_point_b, &|m, k, n| {
            MMMWrapper::Plain((tract_linalg::ops().smmm)(m, k, n))
        })?
    } else if (a.datum_type(), b.datum_type()) == (i8::datum_type(), i8::datum_type()) {
        eval_t(a, b, a_trans, b_trans, c_trans, zero_point_a, zero_point_b, &|m, k, n| {
            MMMWrapper::Quant((tract_linalg::ops().qmmm_i8_i32)(m, k, n))
        })?
    } else if (a.datum_type(), b.datum_type()) == (u8::datum_type(), u8::datum_type()) {
        eval_t(a, b, a_trans, b_trans, c_trans, zero_point_a, zero_point_b, &|m, k, n| {
            MMMWrapper::Quant((tract_linalg::ops().qmmm_u8_i32)(m, k, n))
        })?
    } else {
        bail!(
            "Unsupported combination for MatMul (a: {:?}, b:{:?})",
            a.datum_type(),
            b.datum_type()
        );
    };
    Ok(tvec!(c.into()))
}

fn eval_t<TA, TB, TC, TI>(
    a: &Tensor,
    b: &Tensor,
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
    zero_point_a: Option<&Tensor>,
    zero_point_b: Option<&Tensor>,
    mmm: impl Fn(usize, usize, usize) -> MMMWrapper<TA, TB, TC, TI>,
) -> TractResult<Tensor>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy + Zero + fmt::Debug ,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    let a = a.to_array_view::<TA>()?;
    let b = b.to_array_view::<TB>()?;
    let mut geo = Geo::<TA, TB, TC, TI>::new(a.shape(), b.shape(), a_trans, b_trans, c_trans, mmm)?;
    unsafe {
        geo.mm.as_mmm_mut().c_from_data_and_strides(
            if c_trans { 1 } else { *geo.c_shape.last().unwrap() as isize },
            if !c_trans { 1 } else { *geo.c_shape.last().unwrap() as isize },
        );
        if let Some(ref t) = zero_point_a {
            geo.mm.set_zero_point_a(t)?;
        }
        if let Some(ref t) = zero_point_b {
            geo.mm.set_zero_point_b(t)?;
        }
    }
    let a = a.into_shape(&*geo.bc_a_shape)?;
    let b = b.into_shape(&*geo.bc_b_shape)?;
    let mut c = unsafe { Array::uninitialized(&*geo.c_shape) };

    let b_pack = geo.mm.as_mmm().b_pack();

    let mut pa = unsafe {
        Tensor::uninitialized_aligned::<TA>(
            &[geo.mm.as_mmm().a_pack().len()],
            geo.mm.as_mmm().a_pack().alignment(),
        )?
    };
    let mut pb =
        unsafe { Tensor::uninitialized_aligned::<TB>(&[b_pack.len()], b_pack.alignment())? };

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

        geo.mm.as_mmm().a_pack().pack(
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
            geo.mm.run(pa.as_ptr()?, pb.as_ptr()?, c.as_mut_ptr());
        }
    }
    Ok(c.into_tensor())
}

pub fn infer_shapes<D: DimLike>(
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

#[derive(Clone, Debug)]
pub enum MMMWrapper<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    Plain(Box<dyn MatMatMul<TA, TB, TC, TI>>),
    Quant(Box<dyn QMatMatMul<TA, TB, TC, TI>>),
}

impl<TA, TB, TC, TI> MMMWrapper<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    pub fn as_mmm(&self) -> &dyn MatMatMul<TA, TB, TC, TI> {
        match self {
            MMMWrapper::Plain(a) => a.as_ref(),
            MMMWrapper::Quant(a) => a.as_mmm(),
        }
    }

    pub fn as_mmm_mut(&mut self) -> &mut dyn MatMatMul<TA, TB, TC, TI> {
        match self {
            MMMWrapper::Plain(a) => a.as_mut(),
            MMMWrapper::Quant(a) => a.as_mmm_mut(),
        }
    }

    pub fn as_quant(&self) -> Option<&dyn QMatMatMul<TA, TB, TC, TI>> {
        match self {
            MMMWrapper::Plain(_) => None,
            MMMWrapper::Quant(a) => Some(a.deref()),
        }
    }

    pub fn as_quant_mut(&mut self) -> Option<&mut dyn QMatMatMul<TA, TB, TC, TI>> {
        match self {
            MMMWrapper::Plain(_) => None,
            MMMWrapper::Quant(ref mut a) => Some(a.as_mut()),
        }
    }

    pub unsafe fn run(&self, a: *const TA, b: *const TB, c: *mut TC) {
        match self {
            MMMWrapper::Plain(p) => p.run(a, b, c),
            MMMWrapper::Quant(q) => q.run(a, b, c),
        }
    }

    pub fn set_zero_point_a(&mut self, t: &Tensor) -> TractResult<()> {
        let q = self.as_quant_mut().ok_or("try to zero_point on a float mat mul")?;
        unsafe {
            if t.rank() == 0 {
                q.set_zero_point_a_scalar(*t.to_scalar()?)
            } else {
                q.set_zero_point_a_vector(t.as_slice()?.to_vec())
            }
        }
        Ok(())
    }

    pub fn set_zero_point_b(&mut self, t: &Tensor) -> TractResult<()> {
        let q = self.as_quant_mut().ok_or("try to zero_point on a float mat mul")?;
        unsafe {
            if t.rank() == 0 {
                q.set_zero_point_b_scalar(*t.to_scalar()?)
            } else {
                q.set_zero_point_b_vector(t.as_slice()?.to_vec())
            }
        }
        Ok(())
    }
}

impl<TA, TB, TC, TI> fmt::Display for MMMWrapper<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MMMWrapper::Plain(a) => write!(fmt, "{}", a),
            MMMWrapper::Quant(a) => write!(fmt, "{}", a),
        }
    }
}

#[derive(Debug, Clone)]
struct Geo<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    m: usize,
    k: usize,
    n: usize,
    mm: MMMWrapper<TA, TB, TC, TI>,
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

impl<TA, TB, TC, TI> Geo<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    pub fn new(
        a_shape: &[usize],
        b_shape: &[usize],
        a_trans: bool,
        b_trans: bool,
        c_trans: bool,
        mmm: impl Fn(usize, usize, usize) -> MMMWrapper<TA, TB, TC, TI>,
    ) -> TractResult<Geo<TA, TB, TC, TI>> {
        let (bc_a_shape, bc_b_shape, bc_c_shape) =
            infer_shapes(a_shape.into(), b_shape.into(), a_trans, b_trans, c_trans)?;
        let m = bc_a_shape[bc_a_shape.len() - 2 + a_trans as usize];
        let k = bc_a_shape[bc_a_shape.len() - 1 - a_trans as usize];
        let n = bc_b_shape[bc_b_shape.len() - 1 - b_trans as usize];
        let mm = mmm(m, k, n);
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

#[derive(Debug, Clone, Default)]
pub struct MatMul {
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
    zero_point_a: Option<Arc<Tensor>>,
    zero_point_b: Option<Arc<Tensor>>,
}

impl MatMul {
    pub fn with_a_trans(self, a_trans: bool) -> MatMul {
        MatMul { a_trans, ..self }
    }

    pub fn with_b_trans(self, b_trans: bool) -> MatMul {
        MatMul { b_trans, ..self }
    }

    pub fn with_c_trans(self, c_trans: bool) -> MatMul {
        MatMul { c_trans, ..self }
    }

    pub fn with_zero_point_a(self, zero_point: &Arc<Tensor>) -> MatMul {
        MatMul { zero_point_a: Some(zero_point.clone()), ..self }
    }

    pub fn with_zero_point_b(self, zero_point: &Arc<Tensor>) -> MatMul {
        MatMul { zero_point_b: Some(zero_point.clone()), ..self }
    }
}

impl Op for MatMul {
    fn name(&self) -> Cow<str> {
        "MatMul".into()
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for MatMul {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        eval(
            &inputs[0],
            &inputs[1],
            self.a_trans,
            self.b_trans,
            self.c_trans,
            self.zero_point_a.as_ref().map(|t| t.as_ref()),
            self.zero_point_b.as_ref().map(|t| t.as_ref()),
        )
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
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(
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

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let konst_ix = if model.outlet_fact(node.inputs[0])?.konst.is_some() {
            0
        } else if model.outlet_fact(node.inputs[1])?.konst.is_some() {
            1
        } else {
            return Ok(None);
        };

        let var_ix = 1 - konst_ix;
        let flip = konst_ix == 1;
        let t_konst = [self.a_trans, self.b_trans][konst_ix] ^ flip;
        let t_var = [self.b_trans, self.a_trans][konst_ix] ^ flip;
        let konst = model.outlet_fact(node.inputs[konst_ix])?.konst.clone().unwrap();
        let patch = TypedModelPatch::replace_single_op(
            model,
            node,
            &node.inputs[var_ix..][..1],
            MatMulUnary::new(
                konst,
                t_konst,
                t_var,
                self.c_trans ^ flip,
                self.zero_point_a.clone(),
                self.zero_point_b.clone(),
            ),
        )?;
        return Ok(Some(patch));
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        cost(
            &inputs[0].shape.to_tvec(),
            &inputs[1].shape.to_tvec(),
            inputs[0].datum_type,
            self.a_trans,
            self.b_trans,
        )
    }

    typed_op_as_op!();
}

#[derive(Debug, Clone, new)]
pub struct MatMulUnary {
    a: Arc<Tensor>,
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
    zero_point_a: Option<Arc<Tensor>>,
    zero_point_b: Option<Arc<Tensor>>,
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

    canonic!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for MatMulUnary {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        eval(
            &self.a,
            &inputs[0],
            self.a_trans,
            self.b_trans,
            self.c_trans,
            self.zero_point_a.as_ref().map(|t| t.as_ref()),
            self.zero_point_b.as_ref().map(|t| t.as_ref()),
        )
    }
}

impl TypedOp for MatMulUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(
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

    fn axes_info(&self, model: &TypedModel, node: &TypedNode) -> TractResult<AxesInfo> {
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

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        cost(
            self.a.shape(),
            &inputs[0].shape.to_tvec(),
            self.a.datum_type(),
            self.a_trans,
            self.b_trans,
        )
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
        let fact = target.outlet_fact(input)?;
        if fact.axis >= fact.shape.len() - 1 {
            bail!("Can not pulsify MatMulUnaryA on the most inner dimension (k)");
        }
        target.wire_node(&*node.name, self.clone(), &[input])
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let b = args_1!(model.node_input_facts(node.id)?);
        if let Some(b_shape) = b.shape.as_finite() {
            let patch =
                if (self.a.datum_type(), b.datum_type) == (f32::datum_type(), f32::datum_type()) {
                    new_mat_mul_unary_finite(
                        model,
                        node,
                        self.a.clone(),
                        b_shape,
                        self.a_trans,
                        self.b_trans,
                        self.c_trans,
                        self.zero_point_a.as_ref().map(|t| t.as_ref()),
                        self.zero_point_b.as_ref().map(|t| t.as_ref()),
                        &|m, k, n| MMMWrapper::Plain((tract_linalg::ops().smmm)(m, k, n)),
                    )?
                } else {
                    bail!(
                        "Unsupported combination for MatMul (a: {:?}, b:{:?})",
                        self.a.datum_type(),
                        b.datum_type
                    );
                };
            return Ok(Some(patch));
        }
        Ok(None)
    }

    typed_op_as_op!();
}

impl PulsedOp for MatMulUnary {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape = infer_shapes(
            self.a.shape().into_iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
            inputs[0].shape.iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
            self.a_trans,
            self.b_trans,
            self.c_trans,
        )?
        .2
        .iter()
        .map(|d| d.to_integer().unwrap() as usize)
        .collect::<TVec<_>>();
        Ok(tvec!(fact))
    }

    pulsed_op_as_op!();
    pulsed_op_to_typed_op!();
}

fn new_mat_mul_unary_finite<TA, TB, TC, TI>(
    model: &TypedModel,
    node: &TypedNode,
    a: Arc<Tensor>,
    b_shape: &[usize],
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
    zero_point_a: Option<&Tensor>,
    zero_point_b: Option<&Tensor>,
    mmm: &impl Fn(usize, usize, usize) -> MMMWrapper<TA, TB, TC, TI>,
) -> TractResult<TypedModelPatch>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    let mut patch = TypedModelPatch::default();
    let mut wire = patch.tap_model(model, node.inputs[0])?;
    let mut geo = Geo::<TA, TB, TC, TI>::new(a.shape(), b_shape, a_trans, b_trans, c_trans, mmm)?;
    let a = a.to_array_view::<TA>()?;
    let a = a.into_shape(&*geo.bc_a_shape)?;
    let packed_as = Array::from_shape_fn(&a.shape()[0..a.ndim() - 2], |a_prefix| {
        let mut a = a.view();
        for x in a_prefix.slice() {
            a.index_axis_inplace(Axis(0), *x);
        }
        let mut pa = unsafe {
            Tensor::uninitialized_aligned::<TA>(
                &[geo.mm.as_mmm().a_pack().len()],
                geo.mm.as_mmm().a_pack().alignment(),
            )
            .unwrap()
        };
        geo.mm.as_mmm().a_pack().pack(
            pa.as_ptr_mut().unwrap(),
            a.as_ptr(),
            a.strides()[a_trans as usize],
            a.strides()[!a_trans as usize],
        );
        pa.into_arc_tensor()
    });
    unsafe {
        if geo.n == 1 {
            geo.mm.as_mmm_mut().b_vec_from_data_and_stride(if b_trans {
                1
            } else {
                *geo.b_shape.last().unwrap() as isize
            });
            geo.mm.as_mmm_mut().c_vec_from_data_and_stride(if c_trans {
                1
            } else {
                *geo.c_shape.last().unwrap() as isize
            });
        } else {
            geo.mm.as_mmm_mut().c_from_data_and_strides(
                if c_trans { 1 } else { *geo.c_shape.last().unwrap() as isize },
                if !c_trans { 1 } else { *geo.c_shape.last().unwrap() as isize },
            );
        };
        if let Some(ref t) = zero_point_a {
            geo.mm.set_zero_point_a(t)?;
        }
        if let Some(ref t) = zero_point_b {
            geo.mm.set_zero_point_b(t)?;
        }
    }
    if geo.n > 1 {
        let mut packed_b_shape: TVec<usize> = b_shape[..b_shape.len() - 2].into();
        packed_b_shape.push(geo.mm.as_mmm().b_pack().len());
        wire = patch.wire_node(
            format!("{}-pack", &*node.name),
            MatMatMulPackB {
                pack_b: geo.mm.as_mmm().b_pack().clone(),
                col_stride: if b_trans { *b_shape.last().unwrap() as isize } else { 1 },
                row_stride: if b_trans { 1 } else { *b_shape.last().unwrap() as isize },
                output_shape: packed_b_shape,
            },
            &[wire],
        )?[0];
    }
    let c_prefix_dim_and_stride = if geo.c_shape_prefix.iter().any(|d| *d > 1) {
        let c_prefix_strides: TVec<isize> = geo
            .c_shape
            .iter()
            .rev()
            .scan(1isize, |s, &d| {
                let now: isize = *s;
                *s *= d as isize;
                Some(now)
            })
            .collect::<TVec<_>>()
            .into_iter()
            .skip(2)
            .rev()
            .collect::<TVec<_>>();
        Some((geo.c_shape_prefix.clone(), c_prefix_strides))
    } else {
        None
    };
    wire = patch.wire_node(
        format!("{}-matmatmul", &*node.name),
        MatMatMulUnaryFinite {
            c_shape: geo.c_shape,
            c_prefix_dim_and_stride,
            packed_as,
            mmm: geo.mm,
        },
        &[wire],
    )?[0];
    patch.shunt_outside(OutletId::new(node.id, 0), wire)?;
    Ok(patch)
}

fn cost<A: ToDim + Clone, B: ToDim + Clone>(
    a: &[A],
    b: &[B],
    dt: DatumType,
    a_trans: bool,
    b_trans: bool,
) -> TractResult<TVec<(Cost, TDim)>> {
    let (bc_a_shape, bc_b_shape, bc_c_shape) = infer_shapes(
        a.iter().map(|d| d.clone().to_dim()).collect(),
        b.iter().map(|d| d.clone().to_dim()).collect(),
        a_trans,
        b_trans,
        false,
    )?;
    let mul = bc_c_shape.iter().rev().skip(2).cloned().product::<TDim>();
    let m = &bc_a_shape[bc_a_shape.len() - 2 + a_trans as usize];
    let k = &bc_a_shape[bc_a_shape.len() - 1 - a_trans as usize];
    let n = &bc_b_shape[bc_b_shape.len() - 1 - b_trans as usize];
    Ok(tvec!((Cost::FMA(dt), (mul * m * k * n))))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bin() {
        let a = rctensor2(&[[0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let b = rctensor2(&[[0f32], [1.0], [2.0]]);
        let c = rctensor2(&[[5f32], [14.0]]);
        let op = MatMul::default();
        let c_found = op.eval(tvec!(a, b)).unwrap().pop().unwrap();
        c.close_enough(&c_found, true).unwrap();
    }

    #[test]
    fn bin_transpose() {
        let a = rctensor2(&[[0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let b = rctensor2(&[[0f32], [1.0], [2.0]]);
        let c = rctensor2(&[[5f32], [14.0]]);
        let op = MatMul::default().with_a_trans(true).with_b_trans(true).with_c_trans(true);
        let c_found = op.eval(tvec!(b, a)).unwrap().pop().unwrap();
        c.close_enough(&c_found, true).unwrap();
    }
}
