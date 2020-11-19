use crate::internal::*;
use crate::ops::matmul::*;
use crate::ops::quant::{QParams, QParamsInputKind};

pub(super) fn q_params_from_inputs(
    q_params: &Option<QParams>,
    inputs: &TVec<TensorVar>,
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
                            q_params.set_zero_point_b(&inputs[*ix]);
                        }
                        QParamsInputKind::ZeroPointC(ix) => {
                            q_params.set_zero_point_c(&inputs[*ix]);
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
pub struct MatMul {
    pub a_trans: bool,
    pub b_trans: bool,
    pub c_trans: bool,
    pub q_params: Option<QParams>,
}

impl_dyn_hash!(MatMul);

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

    fn eval(&self, inputs: TVec<TensorVar>) -> TractResult<TVec<Tensor>> {
        assert_eq!(&inputs[0].rank(), &inputs[1].rank());

        let q_params = q_params_from_inputs(&self.q_params, &inputs)?;
        let q_params = q_params.as_ref().or(self.q_params.as_ref());

        let t = eval(&inputs[0], &inputs[1], self.a_trans, self.b_trans, self.c_trans, q_params)?;
        Ok(tvec!(t))
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bin() {
        //           0
        //           1
        //           2
        //
        // 0 1 2     5
        // 3 4 5    14
        let a = tensor2(&[[0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let b = tensor2(&[[0f32], [1.0], [2.0]]);
        let c = tensor2(&[[5f32], [14.0]]);
        let op = MatMul::default();
        let c_found = op.eval(tvec!(a.into(), b.into())).unwrap().pop().unwrap();
        c.close_enough(&c_found, true).unwrap();
    }

    #[test]
    fn bin_transpose() {
        let a = tensor2(&[[0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let b = tensor2(&[[0f32], [1.0], [2.0]]);
        let c = tensor2(&[[5f32], [14.0]]);
        let op = MatMul::default().with_a_trans(true).with_b_trans(true).with_c_trans(true);
        let c_found = op.eval(tvec!(b.into(), a.into())).unwrap().pop().unwrap();
        c.close_enough(&c_found, true).unwrap();
    }

    #[test]
    fn batch_input() -> TractResult<()> {
        crate::setup_test_logger();
        let (batch, len, ci, co) = (2, 3, 4, 5);
        let mut model = TypedModel::default();
        let input_shape = tvec!(batch, len, ci);
        let mut wire =
            tvec!(model.add_source("s", TypedFact::dt_shape(f32::datum_type(), &*input_shape))?);
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
        model.clone().into_runnable()?.run(tvec!((&input).into()))?;
        trace!("running optimized");
        model.declutter()?.optimize()?.into_runnable()?.run(tvec!(input.into()))?;
        Ok(())
    }
}
