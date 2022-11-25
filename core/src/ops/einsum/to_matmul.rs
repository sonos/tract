use super::EinSum;
use crate::internal::*;
use crate::ops::einsum::Axis;
use crate::ops::matmul::{MatMul, MatMulAxes};

pub fn declutter(
    op: &super::EinSum,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    // 2 input, exactly one axis sum
    // FIXME: no projection (axis present more than once in one input)
    if op.expr.n_inputs() != 2 || op.expr.sum.len() != 1 || op.expr.output_rank() < 1 {
        return Ok(None);
    }
    let k_axis = op.expr.sum[0].clone();
    assert!(k_axis.result.is_none());
    assert!(k_axis.inputs.iter().all(|pos| pos.len() == 1));
    let inputs = model.node_input_facts(node.id)?;
    //    eprintln!("{}", op.expr);
    // summing axis is either last or last before last
    if k_axis.inputs[0][0] + 2 < inputs[0].rank() || k_axis.inputs[1][0] + 2 < inputs[1].rank() {
        trace!("Not decluttering, k_axis");
        return Ok(None);
    }
    let a_rank = inputs[0].rank();
    let b_rank = inputs[1].rank();
    let c_rank = op.expr.output_rank();

    let mut patch = TypedModelPatch::default();
    let a = patch.tap_model(model, node.inputs[0])?;
    let b = patch.tap_model(model, node.inputs[1])?;

    // TODO: replace the >= c_rank - 2 criteria by max
    let m_axis = op
        .expr
        .input_axes(0)
        .find(|ax| ax.inputs[1].len() == 0 && ax.result.unwrap() >= c_rank.saturating_sub(2));
    let n_axis = op
        .expr
        .input_axes(1)
        .find(|ax| ax.inputs[0].len() == 0 && ax.result.unwrap() >= c_rank.saturating_sub(2));

    if m_axis.is_none() {
        // FIXME remove me when matmul get smarter
        if k_axis.inputs[0][0] == a_rank - 1 {
            let repr = ('m'..).find(|c| op.expr.axis_by_repr(*c).is_none()).unwrap();
            let mut new_expr = op.expr.clone();
            new_expr.index.push(Axis::new(repr).input(0, a_rank).result(c_rank));
            let add =
                patch.wire_node(format!("{}.add_m_axis", &node.name), AxisOp::Add(a_rank), &[a])?;
            let sum = patch.wire_node(
                format!("{}.mm", &node.name),
                EinSum::new(new_expr),
                &[add[0], b],
            )?;
            let rm = patch.wire_node(&node.name, AxisOp::Rm(c_rank), &sum)?;
            patch.shunt_outside(model, node.id.into(), rm[0])?;
            return Ok(Some(patch));
        } else {
            // TODO: permute k to end
            return Ok(None);
        }
    }
    let m_axis = m_axis.unwrap();

    if n_axis.is_none() {
        // FIXME remove me when matmul get smarter
        if k_axis.inputs[1][0] == b_rank - 1 {
            let repr = ('n'..).find(|c| op.expr.axis_by_repr(*c).is_none()).unwrap();
            let mut new_expr = op.expr.clone();
            new_expr.index.push(Axis::new(repr).input(1, b_rank).result(c_rank));
            let add =
                patch.wire_node(format!("{}.add_n_axis", &node.name), AxisOp::Add(b_rank), &[b])?;
            let sum = patch.wire_node(
                format!("{}.mm", &node.name),
                EinSum::new(new_expr),
                &[a, add[0]],
            )?;
            let rm = patch.wire_node(&node.name, AxisOp::Rm(c_rank), &sum)?;
            patch.shunt_outside(model, node.id.into(), rm[0])?;
            return Ok(Some(patch));
        } else {
            // TODO: permute k to end
            return Ok(None);
        }
    }
    let n_axis = n_axis.unwrap();

    // add broadcasting axes if required
    for axis in &op.expr.index {
        if axis == n_axis || axis == m_axis {
            continue;
        }
        if axis.inputs[0].len() == 0 {
            let mut new_expr = op.expr.clone();
            new_expr.insert_input_axis(axis.repr, 0, 0);
            //eprintln!("{}", &new_expr);
            let add = patch.wire_node(
                format!("{}.add_bc_axis.a.{}", &node.name, a_rank),
                AxisOp::Add(0),
                &[a],
            )?;
            let sum = patch.wire_node(&node.name, EinSum::new(new_expr), &[add[0], b])?;
            patch.shunt_outside(model, node.id.into(), sum[0])?;
            return Ok(Some(patch));
        }
        if axis.inputs[1].len() == 0 {
            let mut new_expr = op.expr.clone();
            new_expr.insert_input_axis(axis.repr, 1, 0);
            //eprintln!("{}", &new_expr);
            let add = patch.wire_node(
                format!("{}.add_bc_axis.b.{}", &node.name, b_rank),
                AxisOp::Add(0),
                &[b],
            )?;
            let sum = patch.wire_node(&node.name, EinSum::new(new_expr), &[a, add[0]])?;
            patch.shunt_outside(model, node.id.into(), sum[0])?;
            return Ok(Some(patch));
        }
    }
    // FIXME
    let axes = MatMulAxes::default_for_ranks(a_rank, b_rank, c_rank).transposing(
        k_axis.inputs[0][0] == a_rank - 2,
        k_axis.inputs[1][0] == b_rank - 1,
        m_axis.result == Some(c_rank - 1),
    );
    let op = MatMul { axes };
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
                .context("wiring initial network")
                .unwrap();
            model.set_output_outlets(&c).unwrap();
            let expect = model
                .clone()
                .into_runnable()
                .unwrap()
                .run(tvec!(self.a.clone().into(), self.b.clone().into()))
                .context("running original network")
                .unwrap();
            let model = model.into_decluttered().unwrap();
            assert!(model.nodes.iter().all(|n| !n.op_is::<EinSum>()));
            let found = model
                .into_runnable()
                .unwrap()
                .run(tvec!(self.a.clone().into(), self.b.clone().into()))
                .unwrap();
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

    #[test]
    fn test_vmm() {
        EinSumProblem::new(t(&[2]), t(&[2, 4]), "i,io->o").check()
    }

    #[test]
    fn test_mvm() {
        EinSumProblem::new(t(&[2, 4]), t(&[2]), "io,i->o").check()
    }

    #[test]
    fn test_complex() {
        EinSumProblem::new(t(&[1, 1, 2, 3]), t(&[2, 3, 4]), "abgi,gih->abgh").check()
    }
}
