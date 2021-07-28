use anyhow::ensure;
use ops::binary::wire_with_rank_broadcast;

use crate::internal::*;
use crate::ops;
use crate::ops::matmul::*;
use crate::ops::quant::offset_u8_as_i8_elementwise;

use super::mir_quant_unary::QMatMulUnary;

#[derive(Debug, Clone, Hash, PartialEq)]
pub enum QParamKind {
    Attr(Arc<Tensor>),
    FromInput(usize),
    FromQType,
}

impl QParamKind {
    fn remove_input(&mut self, ix: usize) {
        if let QParamKind::FromInput(slot) = self {
            *slot = *slot - (*slot > ix) as usize;
        }
    }

    fn insert_input(&mut self, ix: usize) {
        if let QParamKind::FromInput(slot) = self {
            *slot = *slot + (*slot >= ix) as usize;
        }
    }

    pub fn as_static(&self) -> Option<&Arc<Tensor>> {
        match self {
            QParamKind::Attr(t) => Some(t),
            QParamKind::FromInput(_) => None,
            QParamKind::FromQType => None,
        }
    }

    pub fn offset_u8_as_i8(&self, model: &TypedModel, inputs: &[OutletId]) -> TractResult<Self> {
        let tensor = match self {
            QParamKind::Attr(t) => t,
            QParamKind::FromInput(i) => model
                .outlet_fact(inputs[*i])?
                .konst
                .as_ref()
                .ok_or(format_err!("Expected static quantization parameter"))?,
            QParamKind::FromQType => return Ok(QParamKind::FromQType),
        };
        match tensor.datum_type().unquantized() {
            DatumType::U8 => Ok(QParamKind::Attr(
                tensor.to_array_view()?.mapv(offset_u8_as_i8_elementwise).into_arc_tensor(),
            )),
            DatumType::I32 => Ok(QParamKind::Attr(
                tensor.to_array_view()?.mapv(|i: i32| i - 128).into_arc_tensor(),
            )),
            _ => Ok(self.clone()),
        }
    }
}

impl From<Tensor> for QParamKind {
    fn from(t: Tensor) -> Self {
        QParamKind::Attr(t.into_arc_tensor())
    }
}

impl From<usize> for QParamKind {
    fn from(o: usize) -> Self {
        QParamKind::FromInput(o)
    }
}

impl From<AttrOrInput> for QParamKind {
    fn from(at: AttrOrInput) -> Self {
        match at {
            AttrOrInput::Attr(t) => QParamKind::Attr(t),
            AttrOrInput::Input(o) => QParamKind::FromInput(o),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq)]
pub struct MatMulQParams {
    pub a0: QParamKind,
    pub a_scale: QParamKind,
    pub b0: QParamKind,
    pub b_scale: QParamKind,
    pub c0: QParamKind,
    pub c_scale: QParamKind,
}

impl MatMulQParams {
    pub fn noop_static(dt: DatumType) -> MatMulQParams {
        MatMulQParams {
            a0: QParamKind::Attr(Tensor::zero_scalar_dt(dt).unwrap().into_arc_tensor()),
            a_scale: QParamKind::Attr(rctensor0(1f32)),
            b0: QParamKind::Attr(Tensor::zero_scalar_dt(dt).unwrap().into_arc_tensor()),
            b_scale: QParamKind::Attr(rctensor0(1f32)),
            c0: QParamKind::Attr(Tensor::zero_scalar_dt(dt).unwrap().into_arc_tensor()),
            c_scale: QParamKind::Attr(rctensor0(1f32)),
        }
    }

    pub fn all_dynamic(offset: usize) -> MatMulQParams {
        MatMulQParams {
            a0: QParamKind::FromInput(offset),
            a_scale: QParamKind::FromInput(offset + 1),
            b0: QParamKind::FromInput(offset + 2),
            b_scale: QParamKind::FromInput(offset + 3),
            c0: QParamKind::FromInput(offset + 4),
            c_scale: QParamKind::FromInput(offset + 5),
        }
    }

    pub fn all_from_qtype() -> MatMulQParams {
        MatMulQParams {
            a0: QParamKind::FromQType,
            a_scale: QParamKind::FromQType,
            b0: QParamKind::FromQType,
            b_scale: QParamKind::FromQType,
            c0: QParamKind::FromQType,
            c_scale: QParamKind::FromQType,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &QParamKind)> {
        vec![
            ("a0", &self.a0),
            ("a_scale", &self.a_scale),
            ("b0", &self.b0),
            ("b_scale", &self.b_scale),
            ("c0", &self.c0),
            ("c_scale", &self.c_scale),
        ]
        .into_iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut QParamKind)> {
        vec![
            ("a0", &mut self.a0),
            ("a_scale", &mut self.a_scale),
            ("b0", &mut self.b0),
            ("b_scale", &mut self.b_scale),
            ("c0", &mut self.c0),
            ("c_scale", &mut self.c_scale),
        ]
        .into_iter()
    }

    pub fn inline_static(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<(Vec<OutletId>, MatMulQParams)>> {
        let mut new = self.clone();
        let mut inputs = vec![];
        for (ix, input) in node.inputs.iter().enumerate() {
            if let (Some(position), Some(k)) = (
                self.iter().position(|qp| &QParamKind::FromInput(ix) == qp.1),
                model.outlet_fact(*input)?.konst.as_ref(),
            ) {
                *new.iter_mut().nth(position).unwrap().1 = QParamKind::Attr(k.clone());
                new.remove_input(inputs.len());
            } else {
                inputs.push(*input)
            }
        }
        Ok(Some((inputs, new)).filter(|pair| &pair.1 != self))
    }

    pub fn remove_input(&mut self, ix: usize) {
        for (_, qp) in self.iter_mut() {
            qp.remove_input(ix);
        }
    }

    pub fn insert_input(&mut self, ix: usize) {
        for (_, qp) in self.iter_mut() {
            qp.insert_input(ix)
        }
    }

    pub fn input_count(&self) -> usize {
        self.iter().filter(|qp| matches!(qp.1, QParamKind::FromInput(_))).count()
    }

    pub fn as_outlet_ids(
        &self,
        model: &mut TypedModel,
        node_name: &str,
        inputs_wires: &[OutletId],
        a_dt: DatumType,
        b_dt: DatumType,
        c_dt: DatumType,
    ) -> TractResult<TVec<OutletId>> {
        let mut params_outlets = tvec!();
        for (mut params, dt) in self.iter().chunks(2).into_iter().zip([a_dt, b_dt, c_dt].iter()) {
            if let Some(qp) = dt.qparams() {
                let (x0_name, x0) = params.next().unwrap();
                let (x_scale_name, x_scale) = params.next().unwrap();
                ensure!(
                    matches!(x0, QParamKind::FromQType) && matches!(x_scale, QParamKind::FromQType),
                    "Quantization cannot be specified both in the type and in params"
                );
                let (zp, scale) = qp.zp_scale();
                let zp = tensor0(zp);
                let zp = model.add_const(format!("{}.{}", node_name, x0_name), zp)?;
                let scale = tensor0(scale);
                let scale = model.add_const(format!("{}.{}", node_name, x_scale_name), scale)?;
                params_outlets.push(zp);
                params_outlets.push(scale)
            } else {
                for (param_name, param) in params {
                    match param {
                        QParamKind::Attr(t) => params_outlets.push(
                            model.add_const(format!("{}.{}", node_name, param_name), t.clone())?,
                        ),
                        QParamKind::FromInput(i) => params_outlets.push(inputs_wires[*i]),
                        QParamKind::FromQType => {
                            bail!("Param {} has no quantization parameters", param_name)
                        }
                    }
                }
            }
        }
        Ok(params_outlets)
    }
}

#[derive(Debug, Clone, new, Hash)]
pub struct QMatMul {
    pub a_trans: bool,
    pub b_trans: bool,
    pub c_trans: bool,
    pub output_type: DatumType,
    pub params: MatMulQParams,
}

impl_dyn_hash!(QMatMul);

impl QMatMul {
    pub fn with_a_trans(self, a_trans: bool) -> QMatMul {
        QMatMul { a_trans, ..self }
    }

    pub fn with_b_trans(self, b_trans: bool) -> QMatMul {
        QMatMul { b_trans, ..self }
    }

    pub fn with_c_trans(self, c_trans: bool) -> QMatMul {
        QMatMul { c_trans, ..self }
    }
}

impl Op for QMatMul {
    fn name(&self) -> Cow<str> {
        "QMatMul".into()
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for QMatMul {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        if &inputs[0].rank() != &inputs[1].rank() {
            bail!("Rank mismatch {:?} vs {:?}", inputs[0], inputs[1]);
        }

        let mut model = TypedModel::default();
        let a = model.add_const("source_a", inputs[0].clone())?;
        let b = model.add_const("source_b", inputs[1].clone())?;
        let bias = model.add_const("source_bias", inputs[2].clone())?;

        let mut input_outlets = tvec![a, b, bias];
        for (i, t) in inputs.iter().enumerate().skip(3) {
            input_outlets.push(model.add_const(format!("source_{}", i), t.clone())?)
        }

        let mut params = self.params.as_outlet_ids(
            &mut model,
            "qmatmul_unary",
            &input_outlets,
            inputs[0].datum_type(),
            inputs[1].datum_type(),
            self.output_type,
        )?;

        let a = wire_offset_u8_as_i8(&mut model, "adhoc", a, "a", &mut params[0], "a0")?;
        let b = wire_offset_u8_as_i8(&mut model, "adhoc", b, "b", &mut params[2], "b0")?;

        let new_op = MatMul { a_trans: self.a_trans, b_trans: self.b_trans, c_trans: self.c_trans };
        let result = model.wire_node("adhoc.matmul", new_op, &[a, b])?[0];
        let result = wire_matmul_quant(
            &mut model,
            "adhoc",
            a,
            self.a_trans,
            b,
            self.b_trans,
            Some(bias),
            self.c_trans,
            result,
            self.output_type,
            &params,
        )?;
        model.set_output_outlets(&[result])?;
        model.into_runnable()?.run(tvec![])
    }
}

impl TypedOp for QMatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs.len() != 3 + self.params.input_count() {
            bail!(
                "Inconsistent q matmul. expects {} inputs, got {}",
                3 + self.params.input_count(),
                inputs.len()
            );
        }
        if inputs[0].rank() != inputs[1].rank() {
            bail!(
                "Inconsistent matmul between {:?} and {:?} (rank mismatch)",
                inputs[0],
                inputs[1]
            );
        }
        let (_m, _k, _n, c_shape) = compute_shape(
            &inputs[0].shape,
            &inputs[1].shape,
            self.a_trans,
            self.b_trans,
            self.c_trans,
        )?;

        if inputs[2].rank() == 2 {
            let expected_bias_shape: [TDim; 2] = if self.c_trans {
                [1.to_dim(), c_shape[c_shape.len() - 1].clone()]
            } else {
                [c_shape[c_shape.len() - 2].clone(), 1.to_dim()]
            };
            anyhow::ensure!(*inputs[2].shape == expected_bias_shape);
        } else {
            anyhow::ensure!(inputs[2].shape.iter().product::<TDim>() == 1.to_dim());
        };

        Ok(tvec!(TypedFact::dt_shape(self.output_type, c_shape)))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let a_fact = model.outlet_fact(node.inputs[0])?;
        let b_fact = model.outlet_fact(node.inputs[1])?;
        let bias_fact = model.outlet_fact(node.inputs[2])?;

        if bias_fact.konst.is_none() {
            return Ok(None);
        }

        let konst_ix = if a_fact.konst.is_some() {
            0
        } else if b_fact.konst.is_some() {
            1
        } else {
            return Ok(None);
        };

        let flip = konst_ix == 1;
        let t_konst = [self.a_trans, self.b_trans][konst_ix] ^ flip;
        let t_var = [self.b_trans, self.a_trans][konst_ix] ^ flip;
        let konst = model.outlet_fact(node.inputs[konst_ix])?.konst.as_ref().unwrap();
        let bias = model.outlet_fact(node.inputs[2])?.konst.clone().unwrap();

        let inputs: Vec<_> = node
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(i, out_id)| if i == konst_ix || i == 2 { None } else { Some(*out_id) })
            .collect();

        let new_params = {
            let mut qp = self.params.clone();
            //compensate for the removed parameter
            for (_, a) in qp.iter_mut() {
                if let QParamKind::FromInput(i) = a {
                    *i -= 2
                }
            }
            if flip {
                MatMulQParams {
                    a0: qp.b0,
                    a_scale: qp.b_scale,
                    b0: qp.a0,
                    b_scale: qp.a_scale,
                    ..qp
                }
            } else {
                qp
            }
        };

        TypedModelPatch::replace_single_op(
            model,
            node,
            &inputs,
            QMatMulUnary::new(
                konst.clone(),
                // if bias is uniformly zero, it can be discarded
                Some(bias).filter(|b| {
                    b.as_uniform()
                        .map(|b| b.cast_to_scalar::<f32>().unwrap() != 0.0)
                        .unwrap_or(true)
                }),
                t_konst,
                t_var,
                self.c_trans ^ flip,
                self.output_type,
                new_params,
            ),
        )
        .map(Some)
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

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut patch = TypedModelPatch::default();

        if let Some((inputs, qp)) = self.params.inline_static(model, node)? {
            let mut patch = TypedModelPatch::new("inlining matmul quantized params");
            let inputs: Vec<OutletId> =
                inputs.iter().map(|i| patch.tap_model(model, *i)).collect::<TractResult<_>>()?;
            let op = Self { params: qp, ..self.clone() };
            let wire = patch.wire_node(&node.name, op, &inputs)?;
            patch.shunt_outside(model, node.id.into(), wire[0])?;
            return Ok(Some(patch));
        }

        let a = patch.tap_model(model, node.inputs[0])?;
        let b = patch.tap_model(model, node.inputs[1])?;
        let bias = patch.tap_model(model, node.inputs[2])?;

        let mut input_outlets = tvec![a, b, bias];
        for i in node.inputs.iter().skip(3) {
            input_outlets.push(patch.tap_model(model, *i)?)
        }
        let mut params = self.params.as_outlet_ids(
            &mut patch,
            &*node.name,
            &input_outlets,
            model.node_input_facts(node.id)?[0].datum_type,
            model.node_input_facts(node.id)?[1].datum_type,
            self.output_type,
        )?;

        let a = wire_offset_u8_as_i8(&mut patch, &node.name, a, "a", &mut params[0], "a0")?;
        let b = wire_offset_u8_as_i8(&mut patch, &node.name, b, "b", &mut params[2], "b0")?;

        let new_op = MatMul { a_trans: self.a_trans, b_trans: self.b_trans, c_trans: self.c_trans };
        let result = patch.wire_node(format!("{}.matmul", &node.name), new_op, &[a, b])?[0];
        let result = wire_matmul_quant(
            &mut patch,
            &node.name,
            a,
            self.a_trans,
            b,
            self.b_trans,
            Some(bias),
            self.c_trans,
            result,
            self.output_type,
            &params,
        )?;
        patch.shunt_outside(model, node.id.into(), result)?;
        Ok(Some(patch))
    }

    as_op!();
}

/// Wires the offsetting of a matrix and zero point node.
///
/// Only wires nodes of u8 type and leaves nodes of different type untouched.
pub(crate) fn wire_offset_u8_as_i8(
    model: &mut TypedModel,
    model_name: &str,
    matrix: OutletId,
    matrix_name: &str,
    zero_point: &mut OutletId,
    zero_point_name: &str,
) -> TractResult<OutletId> {
    if let DatumType::U8 = model.outlet_fact(matrix)?.datum_type.unquantized() {
        match model.outlet_fact(*zero_point)?.datum_type.unquantized() {
            DatumType::U8 => {
                *zero_point = model.wire_node(
                    format!("{}.offset_{}_as_i8", model_name, zero_point_name),
                    ops::quant::offset_u8_as_i8(),
                    &[*zero_point],
                )?[0];
            }
            DatumType::I32 => {
                *zero_point = model.wire_node(
                    format!("{}.offset_{}_as_i8", model_name, zero_point_name),
                    ops::math::add::unary(rctensor0(-128i32)),
                    &[*zero_point],
                )?[0];
            }
            _ => (),
        }
        let ret = Ok(model.wire_node(
            format!("{}.offset_{}_as_i8", model_name, matrix_name),
            ops::quant::offset_u8_as_i8(),
            &[matrix],
        )?[0]);
        ret
    } else {
        Ok(matrix)
    }
}

pub(crate) fn wire_matmul_quant(
    model: &mut TypedModel,
    name: &str,
    a: OutletId,
    a_trans: bool,
    b: OutletId,
    b_trans: bool,
    bias: Option<OutletId>,
    c_trans: bool,
    mut result: OutletId,
    output_type: DatumType,
    params: &[OutletId],
) -> TractResult<OutletId> {
    let a_fact = model.outlet_fact(a)?.clone();
    let rank = a_fact.rank();
    let m_axis = rank - 2 + c_trans as usize;
    let n_axis = rank - 1 - c_trans as usize;

    if let Some(bias) = bias {
        result = wire_with_rank_broadcast(
            &format!("{}.add_bias", &name),
            model,
            ops::math::add::bin_typed(),
            &[result, bias],
        )?[0];
    }

    let k = model.outlet_fact(a)?.shape[rank - 2 + !a_trans as usize].clone();

    let abc_scale = combine_scales(model, name, params[1], params[3], params[5])?;

    let a_i32 =
        model.wire_node(format!("{}.a_as_i32", name), ops::cast::cast(i32::datum_type()), &[a])?[0];
    let b_i32 =
        model.wire_node(format!("{}.b_as_i32", name), ops::cast::cast(i32::datum_type()), &[b])?[0];
    let a_k_axis = rank - 2 + !a_trans as usize;
    let sum_a = model.wire_node(
        format!("{}.sum_a", name),
        ops::nn::Reduce::new(tvec!(a_k_axis), ops::nn::Reducer::Sum),
        &[a_i32],
    )?[0];
    let sum_a =
        model.wire_node(format!("{}.sum_a_reduced", name), AxisOp::Rm(a_k_axis), &[sum_a])?[0];
    let b_k_axis = rank - 2 + b_trans as usize;
    let sum_b = model.wire_node(
        format!("{}.sum_b", name),
        ops::nn::Reduce::new(tvec!(b_k_axis), ops::nn::Reducer::Sum),
        &[b_i32],
    )?[0];
    let sum_b =
        model.wire_node(format!("{}.sum_b_reduced", name), AxisOp::Rm(b_k_axis), &[sum_b])?[0];
    let result = compensate_zero_points(
        model, name, result, k, params[0], params[2], sum_a, sum_b, m_axis, n_axis,
    )?;
    requant(model, name, result, output_type, abc_scale, params[4])
}

pub(crate) fn combine_scales(
    model: &mut TypedModel,
    name: &str,
    a_scale: OutletId,
    b_scale: OutletId,
    c_scale: OutletId,
) -> TractResult<OutletId> {
    let ab_scale = wire_with_rank_broadcast(
        &format!("{}.ab_scale", name),
        model,
        ops::math::mul::bin_typed(),
        &[a_scale, b_scale],
    )?[0];
    let abc_scale = wire_with_rank_broadcast(
        &format!("{}.abc_scales", name),
        model,
        ops::math::div::bin_typed(),
        &[ab_scale, c_scale],
    )?[0];
    Ok(abc_scale)
}

pub(crate) fn compensate_zero_points(
    model: &mut TypedModel,
    name: &str,
    result: OutletId,
    k: TDim,
    a0: OutletId,
    b0: OutletId,
    sum_a: OutletId,
    sum_b: OutletId,
    m_axis: usize,
    n_axis: usize,
) -> TractResult<OutletId> {
    let input_shape = model.outlet_fact(result)?.shape.clone();
    let rank = model.outlet_fact(result)?.rank();

    debug_assert_eq!(model.outlet_fact(sum_a)?.rank(), rank - 1);
    debug_assert_eq!(model.outlet_fact(sum_b)?.rank(), rank - 1);

    // make sum_a into from a 1D vector to a vertical matrix, sum_b horizontal
    // switch shapes if c_trans
    let sum_a =
        model.wire_node(format!("{}.reshape_sum_a", name), AxisOp::Add(n_axis), &[sum_a])?[0];

    let sum_b =
        model.wire_node(format!("{}.reshape_sum_b", name), AxisOp::Add(m_axis), &[sum_b])?[0];

    debug_assert_eq!(
        model.outlet_fact(sum_a)?.shape[m_axis],
        model.outlet_fact(result)?.shape[m_axis]
    );
    debug_assert_eq!(
        model.outlet_fact(sum_b)?.shape[n_axis],
        model.outlet_fact(result)?.shape[n_axis]
    );

    let a0 =
        model.wire_node(format!("{}.cast_a0", name), ops::cast::cast(i32::datum_type()), &[a0])?[0];

    let b0 =
        model.wire_node(format!("{}.cast_b0", name), ops::cast::cast(i32::datum_type()), &[b0])?[0];

    let k = model.add_const(format!("{}.k", name), rctensor0(k.clone()))?;
    let k =
        model.wire_node(format!("{}.cast_k", name), ops::cast::cast(i32::datum_type()), &[k])?[0];

    let a0_sum_b = wire_with_rank_broadcast(
        &format!("{}.a0_sum_b", name),
        model,
        ops::math::mul::bin_typed(),
        &[a0, sum_b],
    )?[0];

    let b0_sum_a = wire_with_rank_broadcast(
        &format!("{}.b0_sum_a", name),
        model,
        ops::math::mul::bin_typed(),
        &[b0, sum_a],
    )?[0];

    let a0_k = wire_with_rank_broadcast(
        &format!("{}.a0_k", name),
        model,
        ops::math::mul::bin_typed(),
        &[a0, k],
    )?[0];

    let a0_k_b0 = wire_with_rank_broadcast(
        &format!("{}.a0_k_b0", name),
        model,
        ops::math::mul::bin_typed(),
        &[a0_k, b0],
    )?[0];

    let result = wire_with_rank_broadcast(
        &format!("{}.minus_a0_B", &name),
        model,
        ops::math::sub::bin_typed(),
        &[result, a0_sum_b],
    )?[0];
    let result = wire_with_rank_broadcast(
        &format!("{}.minus_b0_A", &name),
        model,
        ops::math::sub::bin_typed(),
        &[result, b0_sum_a],
    )?[0];

    let result = wire_with_rank_broadcast(
        &format!("{}.plus_a0_k_b0", &name),
        model,
        ops::math::add::bin_typed(),
        &[result, a0_k_b0],
    )?[0];

    debug_assert_eq!(model.outlet_fact(result)?.shape, input_shape);
    Ok(result)
}

pub(crate) fn requant(
    model: &mut TypedModel,
    name: &str,
    wire: OutletId,
    dt: DatumType,
    scale: OutletId,
    zero_point: OutletId,
) -> TractResult<OutletId> {
    let wire = wire_with_rank_broadcast(
        &format!("{}.scale", name),
        model,
        ops::quant::scale::bin_typed(),
        &[scale, wire],
    )?[0];

    let zero_point = model.wire_node(
        format!("{}.cast_c0", name),
        ops::cast::cast(i32::datum_type()),
        &[zero_point],
    )?[0];

    let wire = wire_with_rank_broadcast(
        &format!("{}.zeropoint", name),
        model,
        ops::math::add::bin_typed(),
        &[wire, zero_point],
    )?[0];

    clamp_and_cast_to(model, name, dt, wire)
}

pub(crate) fn clamp_and_cast_to(
    model: &mut TypedModel,
    name: &str,
    dt: DatumType,
    wire: OutletId,
) -> TractResult<OutletId> {
    if dt == i32::datum_type() {
        return Ok(wire);
    }
    let rank = model.outlet_fact(wire)?.rank();
    let inf = dt
        .min_value()
        .cast_to_dt(DatumType::I32)?
        .into_owned()
        .broadcast_into_rank(rank)?
        .into_arc_tensor();
    let sup = dt
        .max_value()
        .cast_to_dt(DatumType::I32)?
        .into_owned()
        .broadcast_into_rank(rank)?
        .into_arc_tensor();
    let wire = model.wire_node(format!("{}.min", name), ops::math::min::unary(sup), &[wire])?;
    let wire = model.wire_node(format!("{}.max", name), ops::math::max::unary(inf), &wire)?;
    let wire = model.wire_node(format!("{}.cast", name), ops::cast::cast(dt), &wire)?;
    Ok(wire[0])
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_ndarray::prelude::*;

    proptest! {
        #[test]
        fn prop_i8_i8_i8(pb in any::<QMatMulProblemI8I8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_i8_i8_u8(pb in any::<QMatMulProblemI8I8U8>()) {
            pb.check();
        }

        #[test]
        fn prop_i8_u8_i8(pb in any::<QMatMulProblemI8U8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_i8_i8(pb in any::<QMatMulProblemU8I8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_i8_u8_u8(pb in any::<QMatMulProblemI8U8U8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_i8_u8(pb in any::<QMatMulProblemU8I8U8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_u8_i8(pb in any::<QMatMulProblemU8U8I8>()) {
            pb.check();
        }

        #[test]
        fn prop_u8_u8_u8(pb in any::<QMatMulProblemU8U8U8>()) {
            pb.check();
        }
    }

    #[test]
    fn c0() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[0]]),
            b: arr2(&[[0]]),
            bias: tensor0(0i32),
            a0: 0,
            b0: 0,
            c0: 1,
            a_scale: 1.0,
            b_scale: 1.0,
            c_scale: 1.0,
        }
        .check();
    }

    #[test]
    fn b_scale() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[0]]),
            b: arr2(&[[0]]),
            bias: tensor0(0i32),
            a0: 0,
            b0: 0,
            c0: 1,
            a_scale: 1.0,
            b_scale: 2.0,
            c_scale: 1.0,
        }
        .check();
    }

    #[test]
    fn sat() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[0]]),
            b: arr2(&[[34]]),
            bias: tensor0(0i32),
            a0: -17,
            b0: 1,
            c0: 0,
            a_scale: 1.0,
            b_scale: 0.05,
            c_scale: 0.25,
        }
        .check();
    }

    #[test]
    fn rounding() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[26]]),
            b: arr2(&[[0]]),
            bias: tensor0(0i32),
            a0: 27,
            b0: -1,
            c0: 1,
            a_scale: 1.0,
            b_scale: 0.05,
            c_scale: 1.0,
        }
        .check();
    }

    #[test]
    fn neg_rounding() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[-23]]),
            b: arr2(&[[-2]]),
            bias: tensor0(0i32),
            a0: -11,
            b0: -45,
            c0: 0,
            a_scale: 0.1,
            b_scale: 1.0,
            c_scale: 1.0,
        }
        .check();
    }

    #[test]
    fn rounding_ties_2() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[47], [0]]),
            b: arr2(&[[1, 0, 30]]),
            bias: tensor0(0i32),
            a0: 86,
            b0: 19,
            c0: 0,
            a_scale: 0.1,
            b_scale: 1.0,
            c_scale: 0.6,
        }
        .check();
    }

    #[test]
    fn rounding_ties_3() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[-30]]),
            b: arr2(&[[0, 107, 0]]),
            bias: tensor0(0i32),
            a0: -59,
            b0: 117,
            c0: 0,
            a_scale: 1.0,
            b_scale: 0.15,
            c_scale: 0.6,
        }
        .check();
    }

    #[test]
    fn onnx_test_matmulinteger() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[11, 7, 3], [10, 6, 2], [9, 5, 1], [8, 4, 0]]),
            b: arr2(&[[1, 4], [2, 5], [3, 6]]),
            bias: tensor0(0i32),
            a0: 12,
            b0: 0,
            c0: 0,
            a_scale: 1.0,
            b_scale: 1.0,
            c_scale: 1.0,
        }
        .check();
    }

    fn round_ties_to_right(x: f32) -> i32 {
        (x + 0.5).floor() as i32
    }

    fn scale() -> BoxedStrategy<f32> {
        prop_oneof![Just(1.0), (1i32..=20).prop_map(|x| x as f32 / 20.0)].boxed()
    }

    macro_rules! impl_qmmp {
        ($name:ident, $a:ty, $b:ty, $c:ty $(,)?) => {
            #[derive(Debug)]
            struct $name {
                a: Array2<$a>,
                b: Array2<$b>,
                bias: Tensor,
                a0: $a,
                b0: $b,
                c0: $c,
                a_scale: f32,
                b_scale: f32,
                c_scale: f32,
            }

            impl $name {
                fn check(&self) {
                    let check_with = |r: &Array2<$c>, opt: bool, qp: bool| {
                        let t = self.tract(opt, qp);
                        assert!(
                            r.iter().zip(t.iter()).all(|(r, t)| r.max(t) - r.min(t) <= 1),
                            "mismatch! optimized plan: {}, dynamic qparams: {}, reference: {:?}, tract: {:?}",
                            opt,
                            qp,
                            r,
                            t,
                        );
                    };

                    let r = self.reference();
                    check_with(&r, false, false);
                    check_with(&r, false, true);
                    check_with(&r, true, false);
                    check_with(&r, true, true);
                }

                fn reference(&self) -> Array2<$c> {
                    let a = self.a.map(|&x| (x as f32 - self.a0 as f32) * self.a_scale);
                    let b = self.b.map(|&x| (x as f32 - self.b0 as f32) * self.b_scale);
                    let c = a.dot(&b);
                    let c = c.map(|&x| round_ties_to_right(x / self.c_scale) + self.c0 as i32);
                    c.map(|&x| x.max(<$c>::MIN as i32).min(<$c>::MAX as i32) as $c)
                }

                fn tract(&self, opt: bool, qp: bool) -> Array2<$c> {
                    let mut model = TypedModel::default();
                    let mut inputs = tvec![];
                    inputs.push(
                        model
                            .add_source(
                                "a",
                                TypedFact::dt_shape(
                                    <$a>::datum_type(),
                                    &[self.a.nrows(), self.a.ncols()],
                                ),
                            )
                            .unwrap(),
                    );
                    inputs.push(
                        model
                            .add_source(
                                "b",
                                TypedFact::dt_shape(
                                    <$b>::datum_type(),
                                    &[self.b.nrows(), self.b.ncols()],
                                ),
                            )
                            .unwrap(),
                    );
                    inputs.push(
                        model
                            .add_source(
                                "bias",
                                TypedFact::dt_shape(i32::datum_type(), self.bias.shape()),
                            )
                            .unwrap(),
                    );
                    let qparams = if qp {
                        inputs.push(model.add_source("a0", TypedFact::scalar::<$a>()).unwrap());
                        inputs
                            .push(model.add_source("a_scale", TypedFact::scalar::<f32>()).unwrap());
                        inputs.push(model.add_source("b0", TypedFact::scalar::<$b>()).unwrap());
                        inputs
                            .push(model.add_source("b_scale", TypedFact::scalar::<f32>()).unwrap());
                        inputs.push(model.add_source("c0", TypedFact::scalar::<$c>()).unwrap());
                        inputs
                            .push(model.add_source("c_scale", TypedFact::scalar::<f32>()).unwrap());
                        MatMulQParams::all_dynamic(3)
                    } else {
                        MatMulQParams {
                            a0: QParamKind::Attr(rctensor0::<$a>(self.a0)),
                            a_scale: QParamKind::Attr(rctensor0::<f32>(self.a_scale)),
                            b0: QParamKind::Attr(rctensor0::<$b>(self.b0)),
                            b_scale: QParamKind::Attr(rctensor0::<f32>(self.b_scale)),
                            c0: QParamKind::Attr(rctensor0::<$c>(self.c0)),
                            c_scale: QParamKind::Attr(rctensor0::<f32>(self.c_scale)),
                        }
                    };
                    let result = model
                        .wire_node(
                            "qmm",
                            QMatMul::new(false, false, false, <$c>::datum_type(), qparams),
                            &inputs,
                        )
                        .unwrap();
                    model.set_output_outlets(&result).unwrap();

                    let inputs = if qp {
                        tvec![
                            self.a.clone().into_tensor(),
                            self.b.clone().into_tensor(),
                            self.bias.clone(),
                            self.a0.into(),
                            self.a_scale.into(),
                            self.b0.into(),
                            self.b_scale.into(),
                            self.c0.into(),
                            self.c_scale.into(),
                        ]
                    } else {
                        tvec![
                            self.a.clone().into_tensor(),
                            self.b.clone().into_tensor(),
                            self.bias.clone(),
                        ]
                    };
                    let mut outputs = if opt { model.into_optimized().unwrap() } else { model }
                        .into_runnable()
                        .unwrap()
                        .run(inputs)
                        .unwrap();
                    outputs
                        .remove(0)
                        .into_tensor()
                        .into_array::<$c>()
                        .unwrap()
                        .into_dimensionality()
                        .unwrap()
                }
            }

            impl Arbitrary for $name {
                type Parameters = ();
                type Strategy = BoxedStrategy<$name>;
                fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
                    (1usize..=4, 1usize..=4, 1usize..=4)
                        .prop_flat_map(|(m, k, n)| {
                            (
                                Just((m, k, n)),
                                vec(any::<$a>(), m * k..=m * k),
                                vec(any::<$b>(), k * n..=k * n),
                                any::<$a>(),
                                any::<$b>(),
                                any::<$c>(),
                                scale(),
                                scale(),
                                scale(),
                            )
                        })
                        .prop_map(|((m, k, n), a, b, a0, b0, c0, a_scale, b_scale, c_scale)| {
                            $name {
                                a: Array2::from_shape_vec((m, k), a).unwrap(),
                                b: Array2::from_shape_vec((k, n), b).unwrap(),
                                bias: tensor0(0i32),
                                a0,
                                b0,
                                c0,
                                a_scale,
                                b_scale,
                                c_scale,
                            }
                        })
                        .boxed()
                }
            }
        };
    }

    impl_qmmp! { QMatMulProblemI8I8I8, i8, i8, i8 }
    impl_qmmp! { QMatMulProblemI8I8U8, i8, i8, u8 }
    impl_qmmp! { QMatMulProblemI8U8I8, i8, u8, i8 }
    impl_qmmp! { QMatMulProblemU8I8I8, u8, i8, i8 }
    impl_qmmp! { QMatMulProblemI8U8U8, i8, u8, u8 }
    impl_qmmp! { QMatMulProblemU8I8U8, u8, i8, u8 }
    impl_qmmp! { QMatMulProblemU8U8I8, u8, u8, i8 }
    impl_qmmp! { QMatMulProblemU8U8U8, u8, u8, u8 }

    #[test]
    fn test_qmmp_i8_i8_i8() {
        QMatMulProblemI8I8I8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 0,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmp_i8_i8_u8() {
        QMatMulProblemI8I8U8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 0,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    #[test]
    fn test_qmmp_i8_u8_i8() {
        QMatMulProblemI8U8I8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 127,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmp_u8_i8_i8() {
        QMatMulProblemU8I8I8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 0,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmp_i8_u8_u8() {
        QMatMulProblemI8U8U8 {
            a: arr2(&[[76, 76, 76], [127, -127, 102]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 51,
            b0: 127,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    #[test]
    fn test_qmmp_u8_i8_u8() {
        QMatMulProblemU8I8U8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[25, 51, 76, 102, 127], [-51, -25, 0, 25, 51], [-25, -51, -76, -102, -127]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 0,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    #[test]
    fn test_qmmp_u8_u8_i8() {
        QMatMulProblemU8U8I8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 127,
            c0: -31,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[-52, -41, -31, -21, -10], [127, 64, 0, -62, -126]]
    }

    #[test]
    fn test_qmmp_u8_u8_u8() {
        QMatMulProblemU8U8U8 {
            a: arr2(&[[204, 204, 204], [255, 1, 230]]),
            b: arr2(&[[152, 178, 203, 229, 254], [76, 102, 127, 152, 178], [102, 76, 51, 25, 0]]),
            bias: tensor0(0i32),
            a0: 179,
            b0: 127,
            c0: 96,
            a_scale: 0.039215688,
            b_scale: 0.039215688,
            c_scale: 0.09411765,
        }
        .check(); // c: [[75, 86, 96, 106, 117], [255, 191, 127, 65, 1]]
    }

    fn setup_qparams(inputs: [usize; 6]) -> ([OutletId; 9], MatMulQParams, TypedModel, OutletId) {
        let mut model = TypedModel::default();
        let ids = [
            model.add_source("a", TypedFact::dt_shape(i8::datum_type(), &[2, 3])).unwrap(),
            model.add_source("b", TypedFact::dt_shape(i8::datum_type(), &[3, 4])).unwrap(),
            model.add_const("bias", Tensor::zero_scalar::<i32>().unwrap()).unwrap(),
            model.add_const("a0", Tensor::zero_scalar::<i8>().unwrap()).unwrap(),
            model.add_const("a_scale", Tensor::zero_scalar::<f32>().unwrap()).unwrap(),
            model.add_source("b0", TypedFact::dt_scalar(i8::datum_type())).unwrap(),
            model.add_source("b_scale", TypedFact::dt_scalar(f32::datum_type())).unwrap(),
            model.add_const("c0", Tensor::zero_scalar::<i8>().unwrap()).unwrap(),
            model.add_const("c_scale", Tensor::zero_scalar::<f32>().unwrap()).unwrap(),
        ];
        let indices = inputs
            .iter()
            .enumerate()
            .sorted_by(|(_, this), (_, other)| this.cmp(other))
            .map(|(idx, _)| idx + 3)
            .collect::<Vec<_>>();
        let qparams = MatMulQParams {
            a0: QParamKind::FromInput(indices[0]),
            a_scale: QParamKind::FromInput(indices[1]),
            b0: QParamKind::FromInput(indices[2]),
            b_scale: QParamKind::FromInput(indices[3]),
            c0: QParamKind::FromInput(indices[4]),
            c_scale: QParamKind::FromInput(indices[5]),
        };
        let op = QMatMul {
            a_trans: false,
            b_trans: false,
            c_trans: false,
            output_type: i8::datum_type(),
            params: qparams.clone(),
        };
        let node = model
            .wire_node(
                "qmatmul",
                op,
                &[
                    ids[0],
                    ids[1],
                    ids[2],
                    ids[inputs[0]],
                    ids[inputs[1]],
                    ids[inputs[2]],
                    ids[inputs[3]],
                    ids[inputs[4]],
                    ids[inputs[5]],
                ],
            )
            .unwrap()[0];

        (ids, qparams, model, node)
    }

    #[test]
    fn test_qparams_inline_ascending() {
        let (ids, qparams, model, node) = setup_qparams([3, 4, 5, 6, 7, 8]);
        let (new_ids, new_qparams) =
            qparams.inline_static(&model, model.node(node.node)).unwrap().unwrap();
        assert_eq!(new_ids, [ids[0], ids[1], ids[2], ids[5], ids[6]]);
        assert!(matches!(
            new_qparams,
            MatMulQParams {
                a0: QParamKind::Attr(_),
                a_scale: QParamKind::Attr(_),
                b0: QParamKind::FromInput(3),
                b_scale: QParamKind::FromInput(4),
                c0: QParamKind::Attr(_),
                c_scale: QParamKind::Attr(_),
            },
        ));
    }

    #[test]
    fn test_qparams_inline_descending() {
        let (ids, qparams, model, node) = setup_qparams([8, 7, 6, 5, 4, 3]);
        let (new_ids, new_qparams) =
            qparams.inline_static(&model, model.node(node.node)).unwrap().unwrap();
        assert_eq!(new_ids, [ids[0], ids[1], ids[2], ids[6], ids[5]]);
        assert!(matches!(
            new_qparams,
            MatMulQParams {
                a0: QParamKind::Attr(_),
                a_scale: QParamKind::Attr(_),
                b0: QParamKind::FromInput(4),
                b_scale: QParamKind::FromInput(3),
                c0: QParamKind::Attr(_),
                c_scale: QParamKind::Attr(_),
            },
        ));
    }

    #[test]
    fn test_qparams_inline_mixed() {
        let (ids, qparams, model, node) = setup_qparams([5, 3, 8, 4, 7, 6]);
        let (new_ids, new_qparams) =
            qparams.inline_static(&model, model.node(node.node)).unwrap().unwrap();
        assert_eq!(new_ids, [ids[0], ids[1], ids[2], ids[5], ids[6]]);
        assert!(matches!(
            new_qparams,
            MatMulQParams {
                a0: QParamKind::Attr(_),
                a_scale: QParamKind::Attr(_),
                b0: QParamKind::FromInput(3),
                b_scale: QParamKind::FromInput(4),
                c0: QParamKind::Attr(_),
                c_scale: QParamKind::Attr(_),
            },
        ));
    }
}
