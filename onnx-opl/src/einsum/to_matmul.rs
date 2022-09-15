use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::matmul::MatMul;

use crate::einsum::EinSum;

pub fn declutter(
    op: &super::EinSum,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    // 2 input, exactly one axis sum
    // FIXME: no projection (axis present more than once in one input)
    if op.expr.n_inputs() != 2 || op.expr.sum.len() != 1 || op.expr.output_rank() < 2 {
        return Ok(None);
    }
    let k_axis = op.expr.sum[0].clone();
    assert!(k_axis.result.is_none());
    assert!(k_axis.inputs.iter().all(|pos| pos.len() == 1));
    let inputs = model.node_input_facts(node.id)?;
    // summing axis is either last or last before last
    if k_axis.inputs[0][0] + 2 < inputs[0].rank() || k_axis.inputs[1][0] + 2 < inputs[1].rank() {
        trace!("Not decluttering, k_axis");
        return Ok(None);
    }
    let a_rank = inputs[0].rank();
    let b_rank = inputs[1].rank();
    let c_rank = op.expr.output_rank();
    let m_axis = op
        .expr
        .input_axis(0, if k_axis.inputs[0][0] == a_rank - 1 { a_rank - 2 } else { a_rank - 1 })
        .unwrap();
    let n_axis = op
        .expr
        .input_axis(1, if k_axis.inputs[1][0] == b_rank - 1 { b_rank - 2 } else { b_rank - 1 })
        .unwrap();
    if m_axis.result != Some(c_rank - 2) && m_axis.result != Some(c_rank - 1) {
        trace!("Not decluttering, m_axis");
        return Ok(None);
    }
    if n_axis.result != Some(c_rank - 2) && n_axis.result != Some(c_rank - 1) {
        trace!("Not decluttering, n_axis");
        return Ok(None);
    }
    // add broadcasting axes if required
    for axis in &op.expr.index {
        if axis == n_axis || axis == m_axis {
            continue;
        }
        if axis.inputs[0].len() == 0 {
            let mut new_expr = op.expr.clone();
            new_expr.insert_input_axis(axis.repr, 0, 0);
            let mut patch = TypedModelPatch::default();
            let a = patch.tap_model(model, node.inputs[0])?;
            let b = patch.tap_model(model, node.inputs[1])?;
            let add = patch.wire_node(
                format!("{}.add_bc_axis.a.{}", &node.name, a_rank),
                AxisOp::Add(0),
                &[a],
            )?;
            let sum = patch.wire_node(&node.name, EinSum::new(new_expr), &[add[0], b])?;
            patch.shunt_outside(model, node.id.into(), sum[0])?;
            return Ok(Some(patch))
        }
        if axis.inputs[1].len() == 0 {
            let mut new_expr = op.expr.clone();
            new_expr.insert_input_axis(axis.repr, 1, 0);
            let mut patch = TypedModelPatch::default();
            let a = patch.tap_model(model, node.inputs[0])?;
            let b = patch.tap_model(model, node.inputs[1])?;
            let add = patch.wire_node(
                format!("{}.add_bc_axis.b.{}", &node.name, b_rank),
                AxisOp::Add(0),
                &[b],
            )?;
            let sum = patch.wire_node(&node.name, EinSum::new(new_expr), &[a, add[0]])?;
            patch.shunt_outside(model, node.id.into(), sum[0])?;
            return Ok(Some(patch))
        }
    }
    let op = MatMul {
        a_trans: k_axis.inputs[0][0] == a_rank - 2,
        b_trans: k_axis.inputs[1][0] == b_rank - 1,
        c_trans: m_axis.result == Some(c_rank - 1),
    };
    Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, op)?))
}

#[cfg(test)]
mod test {
    use super::super::EinSum;
    use super::*;

    struct EinSumProblem {
        a: Tensor,
        b: Tensor,
        expr: String,
    }

    impl EinSumProblem {
        fn new(a: Tensor, b: Tensor, expr: &str) -> EinSumProblem {
            EinSumProblem { a, b, expr: expr.to_string() }
        }

        fn check(&self) {
            let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
            let mut model = TypedModel::default();
            let a = model.add_source("a", TypedFact::from(&self.a).without_value()).unwrap();
            let b = model.add_source("b", TypedFact::from(&self.b).without_value()).unwrap();
            let c = model
                .wire_node("c,", EinSum { expr: self.expr.parse().unwrap() }, &[a, b])
                .unwrap();
            model.set_output_outlets(&*c).unwrap();
            let expect = model
                .clone()
                .into_runnable()
                .unwrap()
                .run(tvec!(self.a.clone(), self.b.clone()))
                .unwrap();
            let model = model.into_decluttered().unwrap();
            assert!(model.nodes.iter().all(|n| !n.op_is::<EinSum>()));
            let found =
                model.into_runnable().unwrap().run(tvec!(self.a.clone(), self.b.clone())).unwrap();
            assert_eq!(found, expect);
        }
    }

    fn t(shape: &[usize]) -> Tensor {
        let mut t = Tensor::zero::<f32>(shape).unwrap();
        t.as_slice_mut::<f32>().unwrap().iter_mut().enumerate().for_each(|(i, x)| *x = i as f32);
        t
    }

    #[test]
    fn simple() {
        EinSumProblem::new(t(&[2, 3]), t(&[3, 4]), "mk,kn->mn").check()
    }

    #[test]
    fn a_trans() {
        EinSumProblem::new(t(&[3, 2]), t(&[3, 4]), "km,kn->mn").check()
    }

    #[test]
    fn b_trans() {
        EinSumProblem::new(t(&[2, 3]), t(&[4, 3]), "mk,nk->mn").check()
    }

    #[test]
    fn c_trans() {
        EinSumProblem::new(t(&[2, 3]), t(&[3, 4]), "mk,kn->nm").check()
    }

    #[test]
    fn prefix() {
        EinSumProblem::new(t(&[5, 3, 2]), t(&[5, 3, 4]), "akm,akn->amn").check()
    }

    #[test]
    fn prefix_broadcast_a() {
        EinSumProblem::new(t(&[1, 3, 2]), t(&[5, 3, 4]), "akm,akn->amn").check()
    }

    #[test]
    fn prefix_broadcast_b() {
        EinSumProblem::new(t(&[5, 3, 2]), t(&[1, 3, 4]), "akm,akn->amn").check()
    }

    #[test]
    fn rank_broadcast_a() {
        EinSumProblem::new(t(&[2, 3]), t(&[5, 3, 4]), "mk,akn->anm").check()
    }

    #[test]
    fn rank_broadcast_b() {
        EinSumProblem::new(t(&[5, 2, 3]), t(&[3, 4]), "amk,kn->anm").check()
    }
}
