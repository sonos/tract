use crate::internal::*;
use crate::ops::matmul::*;
use crate::ops::quant::{QParams, QParamsInputKind};

pub(super) fn q_params_from_inputs(
    q_params: &Option<QParams>,
    inputs: &TVec<Arc<Tensor>>,
) -> TractResult<Option<QParams>> {
    q_params
        .as_ref()
        .and_then(|q_params| {
            q_params.inputs_kind.as_ref().and_then(|inputs_kind| {
                let q_params = q_params.clone();

                Some(inputs_kind.iter().try_fold(q_params, |mut q_params, kind| {
                    match kind {
                        QParamsInputKind::ZeroPointA(ix) => {
                            q_params.set_zero_point_a(&inputs[*ix]);
                        }
                        QParamsInputKind::ZeroPointB(ix) => {
                            q_params.set_zero_point_b(&inputs[*ix].clone());
                        }
                        QParamsInputKind::ZeroPointC(ix) => {
                            q_params.set_zero_point_c(&inputs[*ix].clone());
                        }
                        QParamsInputKind::ScaleABC(a_ix, b_ix, c_ix) => {
                            let scale = *inputs[*a_ix].to_scalar::<f32>()?
                                * *inputs[*b_ix].to_scalar::<f32>()?
                                / *inputs[*c_ix].to_scalar::<f32>()?;

                            q_params.set_scale_factor(scale);
                        }
                    };
                    Ok(q_params)
                }))
            })
        })
        .transpose()
}

/// The binary op. It will declutter to MatMulUnary if either A or B is constant.
///
/// TODO: implemnent TypedOp fully to play nice with optimizer.
/// TODO: codegen fails if A and B are variable inputs.
#[derive(Debug, Clone, Default, Hash)]
pub struct QMatMul {
    pub a_trans: bool,
    pub b_trans: bool,
    pub c_trans: bool,
    pub q_params: Option<QParams>,
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

    pub fn with_q_params(self, q_params: QParams) -> QMatMul {
        QMatMul { q_params: Some(q_params), ..self }
    }
}

impl Op for QMatMul {
    fn name(&self) -> Cow<str> {
        "MatMul".into()
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

        let q_params = q_params_from_inputs(&self.q_params, &inputs)?;
        let q_params = q_params.as_ref().or(self.q_params.as_ref());

        let t = eval(&inputs[0], &inputs[1], self.a_trans, self.b_trans, self.c_trans)?;
        Ok(tvec!(t.into_arc_tensor()))
    }
}

impl TypedOp for QMatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].rank() != inputs[1].rank() {
            bail!(
                "Inconsistent matmul between {:?} and {:?} (rank mismatch)",
                inputs[0],
                inputs[1]
            );
        }
        let dt = self.q_params.as_ref().map(|qp| qp.c_datum_type).unwrap_or(inputs[0].datum_type);
        let (_m, _k, _n, c_shape) = compute_shape(
            &inputs[0].shape,
            &inputs[1].shape,
            self.a_trans,
            self.b_trans,
            self.c_trans,
        )?;
        Ok(tvec!(TypedFact::dt_shape(dt, c_shape)))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let a_fact = model.outlet_fact(node.inputs[0])?;
        let b_fact = model.outlet_fact(node.inputs[1])?;
        use crate::ops;
        use QParamsInputKind::*;
        // Quant: saturate and cast
        if self.q_params.as_ref().map(|qp| qp.c_datum_type.size_of() == 8).unwrap_or(false) {
            let dt = self.q_params.as_ref().unwrap().c_datum_type;
            let mut patch = TypedModelPatch::new("Saturate and cast to Q type");
            let mut new_op = self.clone();
            new_op.q_params.as_mut().unwrap().c_datum_type = DatumType::I32;
            let wire = patch.tap_model(model, node.inputs[0])?;
            let wire = patch.wire_node(&node.name, new_op, &[wire])?;
            let inf = rctensor0(if dt == DatumType::I8 { -128i32 } else { 0 });
            let sup = rctensor0(if dt == DatumType::I8 { 127i32 } else { 255 });
            let wire =
                patch.wire_node(format!("{}.min", node.name), ops::math::min::unary(sup), &wire)?;
            let wire =
                patch.wire_node(format!("{}.max", node.name), ops::math::max::unary(inf), &wire)?;
            let wire =
                patch.wire_node(format!("{}.cast", node.name), ops::cast::cast(dt), &wire)?;
            patch.shunt_outside(model, node.id.into(), wire[0])?;
            return Ok(Some(patch));
        }
        // Quant: zero points
        if self
            .q_params
            .as_ref()
            .map(|qp| {
                qp.zero_point_a.is_some()
                    || qp.zero_point_b.is_some()
                    || qp
                        .inputs_kind
                        .as_ref()
                        .map(|inputs| {
                            inputs
                                .iter()
                                .any(|i| matches!(i, ZeroPointA(_)) || matches!(i, ZeroPointB(_)))
                        })
                        .unwrap_or(false)
            })
            .unwrap_or(false)
        {
            let qp = self.q_params.as_ref().unwrap();
            let mut patch = TypedModelPatch::new("Zeropoints expansion");
            let a = patch.tap_model(model, node.inputs[0])?;
            let b = patch.tap_model(model, node.inputs[1])?;
            let a0: OutletId = if let Some(k) = &qp.zero_point_a {
                patch.add_const(format!("{}.a0", node.name), k.clone())?
            } else if let Some(i) = qp
                .inputs_kind
                .as_ref()
                .unwrap_or(&tvec!())
                .iter()
                .filter_map(
                    |i| if let QParamsInputKind::ZeroPointA(i) = i { Some(i) } else { None },
                )
                .next()
            {
                patch.tap_model(model, node.inputs[*i])?
            } else {
                patch.add_const(format!("{}.a0", node.name), rctensor0(0.0f32))?
            };
            let a0 = patch.wire_node(
                format!("{}.cast_a0", node.name),
                ops::cast::cast(qp.c_datum_type),
                &[a0],
            )?[0];
            /*
            let k = a_fact.shape.last().unwrap();
            let k = patch.add_const(format!("{}.k", node.name), rctensor0(k.clone()))?;
            let k = patch.wire_node(
                format!("{}.cast_k", node.name),
                ops::cast::cast(qp.c_datum_type),
                &[k],
            )?[0];
            let wires = crate::ops::binary::wire_rank_broadcast(
                &format!("{}.a0_k", node.name),
                &mut patch,
                &[a0, k],
            )?;
            let a0_k = patch.wire_node(
                format!("{}.aO_k", node.name),
                ops::math::mul::bin_typed(),
                &wires,
            )?[0];
            let sum_a_over_k = patch.wire_node(
                format!("{}.sum_a_over_k", node.name),
                ops::nn::Reduce::new(
                    tvec!(a_fact.rank() - 1 - !self.a_trans as usize),
                    ops::nn::Reducer::Sum,
                ),
                &[a],
            )?[0];
            */
            let b_i32 = patch.wire_node(
                format!("{}.b_as_i32", node.name),
                ops::cast::cast(qp.c_datum_type),
                &[b],
            )?[0];
            let sum_b_over_k = patch.wire_node(
                format!("{}.sum_b_over_k", node.name),
                ops::nn::Reduce::new(
                    tvec!(b_fact.rank() - 1 - !self.b_trans as usize),
                    ops::nn::Reducer::Sum,
                ),
                &[b_i32],
            )?[0];
            let wires = crate::ops::binary::wire_rank_broadcast(
                &format!("{}.a0_sum_b", node.name),
                &mut patch,
                &[a0, sum_b_over_k],
            )?;
            let a0_k_sum_b_over_k = patch.wire_node(
                format!("{}.a0_k_sum_b", node.name),
                ops::math::mul::bin_typed(),
                &wires,
            )?[0];

            let mut new_op = self.clone();
            new_op.q_params.as_mut().unwrap().zero_point_a = None;
            new_op.q_params.as_mut().unwrap().zero_point_b = None;
            new_op.q_params.as_mut().unwrap().inputs_kind.as_mut().map(|kinds| {
                kinds.retain(|i| !matches!(i, ZeroPointA(_)) && !matches!(i, ZeroPointB(_)))
            });
            dbg!(&new_op);
            let inputs = &node
                .inputs
                .iter()
                .map(|i| patch.tap_model(model, *i))
                .collect::<TractResult<TVec<OutletId>>>()?;
            let product = patch.wire_node(format!("{}.matmul", &node.name), new_op, &inputs)?[0];
            let result = patch.wire_node(
                &format!("{}.minus_a0_B", &node.name),
                ops::math::sub::bin_typed(),
                &[product, a0_k_sum_b_over_k],
            )?[0];
            patch.shunt_outside(model, node.id.into(), result)?;
            return Ok(Some(patch));
        }
        return Ok(None);
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

pub(super) fn cost<A: DimLike + Clone, B: DimLike + Clone>(
    a: &[A],
    b: &[B],
    dt: DatumType,
    a_trans: bool,
    b_trans: bool,
) -> TractResult<TVec<(Cost, TDim)>> {
    let (m, k, n, c_shape) = compute_shape(
        &a.iter().map(|d| d.clone().to_dim()).collect::<TVec<_>>(),
        &b.iter().map(|d| d.clone().to_dim()).collect::<TVec<_>>(),
        a_trans,
        b_trans,
        false,
    )?;
    let mul = c_shape.iter().rev().skip(2).cloned().maybe_product()?;
    Ok(tvec!((Cost::FMA(dt), [mul, m.to_dim(), k.to_dim(), n.to_dim()].iter().maybe_product()?)))
}
