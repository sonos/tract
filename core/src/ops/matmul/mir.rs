use num_traits::Zero;
use std::fmt;
use std::ops::{Add, Mul};

use crate::internal::*;
use crate::ops::matmul::*;
use crate::ops::quant::QParams;
use ndarray::*;

use itertools::Itertools;

fn eval(
    a: &Tensor,
    b: &Tensor,
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
    q_params: Option<&QParams>,
) -> TractResult<Tensor> {
    if let Some(q) = q_params {
        if (a.datum_type(), b.datum_type()) == (i8::datum_type(), i8::datum_type()) {
            if q.c_datum_type == i32::datum_type() {
                return eval_t(a, b, a_trans, b_trans, c_trans, q_params, &|m, k, n| {
                    MMMWrapper::Quant((tract_linalg::ops().qmmm_i8_i32)(m, k, n))
                });
            } else if q.c_datum_type == i8::datum_type() {
                return eval_t(a, b, a_trans, b_trans, c_trans, q_params, &|m, k, n| {
                    MMMWrapper::Quant((tract_linalg::ops().qmmm_i8_i8)(m, k, n))
                });
            }
        } else if (a.datum_type(), b.datum_type()) == (u8::datum_type(), u8::datum_type()) {
            if q.c_datum_type == i32::datum_type() {
                return eval_t(a, b, a_trans, b_trans, c_trans, q_params, &|m, k, n| {
                    MMMWrapper::Quant((tract_linalg::ops().qmmm_u8_i32)(m, k, n))
                });
            } else if q.c_datum_type == u8::datum_type() {
                return eval_t(a, b, a_trans, b_trans, c_trans, q_params, &|m, k, n| {
                    MMMWrapper::Quant((tract_linalg::ops().qmmm_u8_u8)(m, k, n))
                });
            }
        }
    } else if (a.datum_type(), b.datum_type()) == (f32::datum_type(), f32::datum_type()) {
        return eval_t(a, b, a_trans, b_trans, c_trans, q_params, &|m, k, n| {
            MMMWrapper::Plain((tract_linalg::ops().mmm_f32)(m, k, n))
        });
    }
    bail!(
        "Unsupported combination for MatMul eval (a: {:?}, b:{:?} q:{:?})",
        a.datum_type(),
        b.datum_type(),
        q_params
    );
}

fn eval_t<TA, TB, TC, TI>(
    a: &Tensor,
    b: &Tensor,
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
    q_params: Option<&QParams>,
    mmm: impl Fn(usize, usize, usize) -> MMMWrapper<TA, TB, TC, TI>,
) -> TractResult<Tensor>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy + Zero + fmt::Debug,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    let rank = a.rank();
    let a = a.to_array_view::<TA>()?;
    let b = b.to_array_view::<TB>()?;
    let m = a.shape()[a.shape().len() - 2 + a_trans as usize];
    let k = a.shape()[a.shape().len() - 1 - a_trans as usize];
    let n = b.shape()[b.shape().len() - 1 - b_trans as usize];
    let mut mm = mmm(m, k, n);
    let c_shape = compute_shape(a.shape(), b.shape(), a_trans, b_trans, c_trans)?;
    unsafe {
        mm.as_mmm_mut().c_from_data_and_strides(
            if c_trans { 1 } else { c_shape[rank - 1] as isize },
            if !c_trans { 1 } else { c_shape[rank - 1] as isize },
        );
        if let Some(q) = q_params {
            mm.set_quant_params(q)?;
        }
    }
    let mut c = unsafe { Array::<TC, IxDyn>::uninitialized(&*c_shape) };

    let b_pack = mm.as_mmm().b_pack();

    let mut pa = unsafe {
        Tensor::uninitialized_aligned::<TA>(
            &[mm.as_mmm().a_pack().len()],
            mm.as_mmm().a_pack().alignment(),
        )?
    };
    let mut pb =
        unsafe { Tensor::uninitialized_aligned::<TB>(&[b_pack.len()], b_pack.alignment())? };

    for prefix in indices(&c_shape[..rank - 2]).into_iter() {
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
        mm.as_mmm().a_pack().pack(
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
            mm.run(pa.as_ptr()?, pb.as_ptr()?, c.as_mut_ptr(), &[]);
        }
    }
    Ok(c.into_tensor())
}

pub fn compute_shape<D: DimLike>(
    ashape: &[D],
    bshape: &[D],
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
) -> TractResult<TVec<D>> {
    let mut c_shape = crate::broadcast::multi_broadcast(&[
        &ashape[..(ashape.len() - 2)],
        &bshape[..(bshape.len() - 2)],
    ])
    .ok_or_else(|| format_err!("Could not broadcast"))?;
    let (mut m, mut ka) = (ashape[ashape.len() - 2].clone(), ashape[ashape.len() - 1].clone());
    let (mut kb, mut n) = (bshape[bshape.len() - 2].clone(), bshape[bshape.len() - 1].clone());
    if a_trans {
        std::mem::swap(&mut m, &mut ka);
    }
    if b_trans {
        std::mem::swap(&mut kb, &mut n);
    }
    if ka != kb {
        bail!(
            "Inconsistent matmul: a: {} b: {}, a_trans: {} b_trans: {} c_trans: {}",
            ashape.iter().join("x"),
            bshape.iter().join("x"),
            a_trans,
            b_trans,
            c_trans
        );
    }
    if c_trans {
        c_shape.push(n.clone());
        c_shape.push(m.clone());
    } else {
        c_shape.push(m.clone());
        c_shape.push(n.clone());
    }
    Ok(c_shape)
}

#[derive(Debug, Clone, Default, Hash)]
pub struct MatMul {
    pub a_trans: bool,
    pub b_trans: bool,
    pub c_trans: bool,
    pub q_params: Option<QParams>,
}

tract_data::impl_dyn_hash!(MatMul);

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

    pub fn with_q_params(self, q_params: QParams) -> MatMul {
        MatMul { q_params: Some(q_params), ..self }
    }
}

impl Op for MatMul {
    fn name(&self) -> Cow<str> {
        "MatMul".into()
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for MatMul {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        assert_eq!(&inputs[0].rank(), &inputs[1].rank());
        let t = eval(
            &inputs[0],
            &inputs[1],
            self.a_trans,
            self.b_trans,
            self.c_trans,
            self.q_params.as_ref(),
        )?;
        Ok(tvec!(t.into_arc_tensor()))
    }
}

impl TypedOp for MatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].rank() != inputs[1].rank() {
            bail!(
                "Inconsistent matmul between {:?} and {:?} (rank mismatch)",
                inputs[0],
                inputs[1]
            );
        }
        let dt = self.q_params.as_ref().map(|qp| qp.c_datum_type).unwrap_or(inputs[0].datum_type);
        Ok(tvec!(TypedFact::dt_shape(
            dt,
            &*compute_shape(
                &inputs[0].shape,
                &inputs[1].shape,
                self.a_trans,
                self.b_trans,
                self.c_trans,
            )?
        )?))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let a_fact = model.outlet_fact(node.inputs[0])?;
        let b_fact = model.outlet_fact(node.inputs[1])?;
        let konst_ix = if a_fact.konst.is_some() {
            0
        } else if b_fact.konst.is_some() {
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
            MatMulUnary::new(konst, t_konst, t_var, self.c_trans ^ flip, self.q_params.clone()),
        )?
        .with_context("to unary");
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

    as_op!();
}

#[derive(Debug, Clone, new, Hash)]
pub struct MatMulUnary {
    pub a: Arc<Tensor>,
    pub a_trans: bool,
    pub b_trans: bool,
    pub c_trans: bool,
    pub q_params: Option<QParams>,
}

tract_data::impl_dyn_hash!(MatMulUnary);

impl Op for MatMulUnary {
    fn name(&self) -> Cow<str> {
        "MatMulUnary".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut v = vec![
            format!(
                "a_trans:{:?} b_trans:{:?} c_trans:{:?}",
                self.a_trans, self.b_trans, self.c_trans
            ),
            format!("A: {:?}", self.a),
        ];
        if let Some(qp) = &self.q_params {
            v.push(format!("{:?}", qp));
        }
        Ok(v)
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for MatMulUnary {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let t = eval(
            &self.a,
            &inputs[0],
            self.a_trans,
            self.b_trans,
            self.c_trans,
            self.q_params.as_ref(),
        )?;
        Ok(tvec!(t.into_arc_tensor()))
    }
}

impl TypedOp for MatMulUnary {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].rank() != self.a.rank() {
            bail!(
                "Inconsistent matmul between input {:?} and attribute {:?} (rank mismatch)",
                inputs[0],
                self.a
            );
        }
        Ok(tvec!(TypedFact::dt_shape(
            self.q_params.as_ref().map(|qp| qp.c_datum_type).unwrap_or(inputs[0].datum_type),
            &*compute_shape(
                &self.a.shape().iter().map(|d| d.to_dim()).collect::<TVec<_>>(),
                &*inputs[0].shape,
                self.a_trans,
                self.b_trans,
                self.c_trans,
            )?
        )?))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        if input_fact.shape.rank() != node.outputs[0].fact.shape.rank() {
            return Ok(Invariants::none());
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
        if self.b_trans && self.c_trans && input_fact.rank() >= 2 {
            invars.push(AxisInfo::simple(input_fact.shape.rank() - 2))
        }
        if !self.b_trans && !self.c_trans {
            invars.push(AxisInfo::simple(input_fact.shape.rank() - 1))
        };
        Ok(invars.into_iter().collect())
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let b = &model.outlet_fact(node.inputs[0])?;
        match change {
            AxisOp::Move(from, to) => {
                if *from == b.rank() - 2 && *to == b.rank() - 1 {
                    let op = MatMulUnary {
                        b_trans: !self.b_trans,
                        c_trans: !self.c_trans,
                        ..self.clone()
                    };
                    Ok(Some(AxisChangeConsequence::new(model, node, Some(Box::new(op)), change)))
                } else {
                    Ok(None)
                }
            }
            AxisOp::Add(axis) if *axis < b.rank() - 1 => {
                let mut a = self.a.clone().into_tensor();
                a.insert_axis(*axis)?;
                let op =
                    Some(Box::new(MatMulUnary { a: a.into_arc_tensor(), ..self.clone() }) as _);
                Ok(Some(AxisChangeConsequence::new(model, node, op, change)))
            }
            // b is [.. 1, n], can add axis to the right and transpose
            AxisOp::Add(axis) if *axis == b.rank() && b.shape[b.rank() - 2] == 1.to_dim() => {
                let mut a = self.a.clone().into_tensor();
                a.insert_axis(*axis - 2)?;
                let op = MatMulUnary {
                    b_trans: !self.b_trans,
                    c_trans: !self.c_trans,
                    a: a.into_arc_tensor(),
                    ..self.clone()
                };
                Ok(Some(AxisChangeConsequence::new(model, node, Some(Box::new(op)), change)))
            }
            AxisOp::Rm(axis) if b.rank() - axis > 2 => {
                let mut a = self.a.clone().into_tensor();
                a.remove_axis(*axis)?;
                let op =
                    Some(Box::new(MatMulUnary { a: a.into_arc_tensor(), ..self.clone() }) as _);
                Ok(Some(AxisChangeConsequence::new(model, node, op, change)))
            }
            _ => return Ok(None),
        }
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        use crate::ops::array::concat::ConcatSlice;
        use crate::ops::array::TypedConcat;
        let input_fact = model.outlet_fact(node.inputs[0])?;
        if let Some(concat) = model.nodes()[node.inputs[0].node].op().downcast_ref::<TypedConcat>()
        {
            let mut patch = TypedModelPatch::new("split over k-concatenated input");
            let k_axis = self.a.rank() - 1 - self.a_trans as usize;
            if concat.axis == input_fact.shape.rank() - 1 && self.b_trans {
                let mut input = 0;
                let concat_node = model.node(node.inputs[0].node);
                let offsets = concat
                    .offsets(&model.node_input_facts(concat_node.id)?)?
                    .iter()
                    .map(|x| x.to_usize())
                    .collect::<TractResult<Vec<usize>>>()?;
                let mut wires = vec![];
                for (ix, slice) in concat.slices.iter().enumerate() {
                    let wire = match slice {
                        ConcatSlice::Const(t) => patch.add_const(
                            format!("{}.const-{}", node.name, ix),
                            t.clone().into_arc_tensor(),
                        )?,
                        ConcatSlice::Var => {
                            input += 1;
                            patch.tap_model(model, concat_node.inputs[input - 1])?
                        }
                    };
                    let mut a = self.a.slice(k_axis, offsets[ix], offsets[ix + 1])?;
                    while a.rank() > 0 && a.shape()[0] == 1 {
                        a.remove_axis(0)?;
                    }
                    let wire = patch.wire_node(
                        format!("{}.k-{}-{}", node.name, offsets[ix], offsets[ix + 1]),
                        MatMulUnary { a: a.into_arc_tensor(), ..self.clone() },
                        &[wire],
                    )?[0];
                    wires.push(wire)
                }
                let mut wire = wires[0];
                for (ix, w) in wires[1..].iter().enumerate() {
                    wire = patch.wire_node(
                        format!("{}.k-add-{}", node.name, ix),
                        crate::ops::binary::TypedBinOp(Box::new(crate::ops::math::Add)),
                        &[wire, *w],
                    )?[0];
                }
                patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }

    fn slice_output(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: &mut TypedModelPatch,
        _output_slot: usize,
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<OutletId>> {
        let b_fact = model.outlet_fact(node.inputs[0])?;
        let c_fact = &self.output_facts(&[b_fact])?[0];
        if axis + self.c_trans as usize == c_fact.shape.rank() {
            let a_split_axis = self.a.rank() - 1 - !self.a_trans as usize;
            let a = self.a.slice(a_split_axis, start, end)?.into_arc_tensor();
            let wire = patch.tap_model(model, node.inputs[0])?;
            return Ok(Some(
                patch.wire_node(
                    format!("{}.sliced-m-{}-{}", node.name, start, end),
                    Self { a, ..self.clone() },
                    &[wire],
                )?[0],
            ));
        }
        return Ok(None);
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let mut cost = cost(
            self.a.shape(),
            &inputs[0].shape.to_tvec(),
            self.a.datum_type(),
            self.a_trans,
            self.b_trans,
        )?;
        cost.push((Cost::Params(self.a.datum_type()), self.a.len().to_dim()));
        Ok(cost)
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
                        &b_shape,
                        self.a_trans,
                        self.b_trans,
                        self.c_trans,
                        self.q_params.as_ref(),
                        &|m, k, n| MMMWrapper::Plain((tract_linalg::ops().mmm_f32)(m, k, n)),
                    )?
                } else if (
                    self.a.datum_type(),
                    b.datum_type,
                    self.q_params.as_ref().map(|q| q.c_datum_type),
                ) == (i8::datum_type(), i8::datum_type(), Some(i8::datum_type()))
                {
                    new_mat_mul_unary_finite(
                        model,
                        node,
                        self.a.clone(),
                        &b_shape,
                        self.a_trans,
                        self.b_trans,
                        self.c_trans,
                        self.q_params.as_ref(),
                        &|m, k, n| MMMWrapper::Quant((tract_linalg::ops().qmmm_i8_i8)(m, k, n)),
                    )?
                } else if (
                    self.a.datum_type(),
                    b.datum_type,
                    self.q_params.as_ref().map(|q| q.c_datum_type),
                ) == (i8::datum_type(), i8::datum_type(), Some(i32::datum_type()))
                {
                    new_mat_mul_unary_finite(
                        model,
                        node,
                        self.a.clone(),
                        &b_shape,
                        self.a_trans,
                        self.b_trans,
                        self.c_trans,
                        self.q_params.as_ref(),
                        &|m, k, n| MMMWrapper::Quant((tract_linalg::ops().qmmm_i8_i32)(m, k, n)),
                    )?
                } else {
                    bail!(
                        "Unsupported combination for MatMul codegen (a: {:?}, b:{:?}, q: {:?})",
                        self.a.datum_type(),
                        b.datum_type,
                        self.q_params
                    );
                };
            return Ok(Some(patch));
        }
        Ok(None)
    }

    as_op!();
}

fn new_mat_mul_unary_finite<TA, TB, TC, TI>(
    model: &TypedModel,
    node: &TypedNode,
    a: Arc<Tensor>,
    b_shape: &[usize],
    a_trans: bool,
    b_trans: bool,
    c_trans: bool,
    q_params: Option<&QParams>,
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

    let m = a.shape()[a.rank() - 2 + a_trans as usize];
    let k = a.shape()[a.rank() - 1 - a_trans as usize];
    let n = b_shape[b_shape.len() - 1 - b_trans as usize];

    let mut mm = mmm(m, k, n);
    let c_shape = compute_shape(&a.shape(), b_shape, a_trans, b_trans, c_trans)?;
    let a = a.to_array_view::<TA>()?;
    let packed_as = Array::from_shape_fn(&a.shape()[0..a.ndim() - 2], |a_prefix| {
        let mut a = a.view();
        for x in a_prefix.slice() {
            a.index_axis_inplace(Axis(0), *x);
        }
        let mut pa = unsafe {
            Tensor::uninitialized_aligned::<TA>(
                &[mm.as_mmm().a_pack().len()],
                mm.as_mmm().a_pack().alignment(),
            )
            .unwrap()
        };
        mm.as_mmm().a_pack().pack(
            pa.as_ptr_mut().unwrap(),
            a.as_ptr(),
            a.strides()[a_trans as usize],
            a.strides()[!a_trans as usize],
        );
        pa.into_arc_tensor()
    });
    unsafe {
        if n == 1 {
            mm.as_mmm_mut().b_vec_from_data_and_stride(if b_trans {
                1
            } else {
                *b_shape.last().unwrap() as isize
            });
            mm.as_mmm_mut().c_vec_from_data_and_stride(if c_trans {
                1
            } else {
                *c_shape.last().unwrap() as isize
            });
        } else {
            mm.as_mmm_mut().c_from_data_and_strides(
                if c_trans { 1 } else { *c_shape.last().unwrap() as isize },
                if !c_trans { 1 } else { *c_shape.last().unwrap() as isize },
            );
        };
        if let Some(q) = q_params {
            mm.set_quant_params(q)?;
        }
    }
    let rank = c_shape.len();
    if n > 1 {
        let mut packed_b_shape: TVec<usize> = b_shape[..b_shape.len() - 2].into();
        packed_b_shape.push(mm.as_mmm().b_pack().len());
        wire = patch.wire_node(
            format!("{}.pack", &*node.name),
            super::MatMatMulPackB {
                pack_b: mm.as_mmm().b_pack().clone(),
                col_stride: if b_trans { *b_shape.last().unwrap() as isize } else { 1 },
                row_stride: if b_trans { 1 } else { *b_shape.last().unwrap() as isize },
                output_shape: packed_b_shape,
            },
            &[wire],
        )?[0];
    }
    let c_prefix_dim_and_stride = if c_shape[..rank - 2].iter().any(|d| *d > 1) {
        let c_prefix_strides: TVec<isize> = c_shape
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
        Some((c_shape[..rank - 2].into(), c_prefix_strides))
    } else {
        None
    };
    wire = patch.wire_node(
        format!("{}.matmatmul", &*node.name),
        lir::MatMatMulUnaryFinite {
            c_trans,
            c_fact: TypedFact::dt_shape(TC::datum_type(), &*c_shape)?,
            bc_c_shape: c_shape,
            c_prefix_dim_and_stride,
            packed_as,
            fused_ops: None,
            mmm: mm,
        },
        &[wire],
    )?[0];
    patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
    Ok(patch)
}

fn cost<A: DimLike + Clone, B: DimLike + Clone>(
    a: &[A],
    b: &[B],
    dt: DatumType,
    a_trans: bool,
    b_trans: bool,
) -> TractResult<TVec<(Cost, TDim)>> {
    let c_shape = compute_shape(
        &a.iter().map(|d| d.clone().to_dim()).collect::<TVec<_>>(),
        &b.iter().map(|d| d.clone().to_dim()).collect::<TVec<_>>(),
        a_trans,
        b_trans,
        false,
    )?;
    let mul = c_shape.iter().rev().skip(2).cloned().maybe_product()?;
    let m = a[a.len() - 2 + a_trans as usize].to_dim();
    let k = a[a.len() - 1 - a_trans as usize].to_dim();
    let n = b[b.len() - 1 - b_trans as usize].to_dim();
    Ok(tvec!((Cost::FMA(dt), [mul, m.to_dim(), k.to_dim(), n.to_dim()].iter().maybe_product()?)))
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

    #[test]
    fn batch_input() -> TractResult<()> {
        crate::setup_test_logger();
        let (batch, len, ci, co) = (2, 3, 4, 5);
        let mut model = TypedModel::default();
        let input_shape = tvec!(batch, len, ci);
        let mut wire =
            tvec!(model.add_source("s", TypedFact::dt_shape(f32::datum_type(), &*input_shape)?)?);
        let a = unsafe { Tensor::uninitialized::<f32>(&[1, ci, co])?.into_arc_tensor() };
        wire = model.wire_node(
            "m",
            MatMulUnary { a, a_trans: true, b_trans: true, c_trans: true, q_params: None },
            &wire,
        )?;
        let b = unsafe { Tensor::uninitialized::<f32>(&[1, 1, co])?.into_arc_tensor() };
        wire = model.wire_node("a", crate::ops::math::add::unary(b), &wire)?;
        model.set_output_outlets(&wire)?;
        let input = unsafe { Tensor::uninitialized::<f32>(&input_shape)? };
        trace!("running mir");
        model.clone().into_runnable()?.run(tvec!(input.clone()))?;
        trace!("running optimized");
        model.declutter()?.optimize()?.into_runnable()?.run(tvec!(input))?;
        Ok(())
    }
}
