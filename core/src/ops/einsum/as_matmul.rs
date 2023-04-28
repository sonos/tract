use super::EinSum;
use crate::internal::*;

pub fn decompose(op: &EinSum, model: &TypedModel, node: &TypedNode) -> TractResult<TypedModel> {
    let mut substitute = TypedModel::default();
    let inputs = node
        .inputs
        .iter()
        .enumerate()
        .map(|(ix, input)| {
            substitute.add_source(
                format!("adhoc_source_{ix}"),
                model.outlet_fact(*input)?.without_value(),
            )
        })
        .collect::<TractResult<Vec<_>>>()?;
    let outputs = substitute.wire_node(&node.name, op.clone(), &inputs)?;
    let op = substitute.nodes[outputs[0].slot].op_as::<EinSum>();
    decompose_einsums_in_place(&mut substitute)?;
    Ok(substitute)
}

pub fn decompose_einsums_in_place(model: &mut TypedModel) -> TractResult<()> {
    loop {
        for n in model.eval_order()? {
            let node = &model.nodes[n];
            if let Some(einsum) = node.op_as::<EinSum>() {
                if let Some(patch) = step(einsum, model, node)? {
                    patch.apply(model)?;
                    continue;
                }
            }
        }
        return Ok(());
    }
}

pub fn step(
    op: &EinSum,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let (m_axis, k_axis, n_axis) = super::codegen::choose_mkn_axes(op, model, node)?;
    let Some(k_axis) = k_axis else { todo!("Einsum decomposition, no k axis") };
    let Some(m_axis) = m_axis else {
        return Ok(Some(super::codegen::inject_m_or_n_axis(op, model, node, false)?));
    };
    let Some(n_axis) = n_axis else {
        return Ok(Some(super::codegen::inject_m_or_n_axis(op, model, node, true)?));
    };
    let a = model.outlet_fact(node.inputs[0])?;
    let b = model.outlet_fact(node.inputs[1])?;
    let c = &node.outputs[0].fact;
    assert_eq!(a.rank(), b.rank());
    assert_eq!(a.rank(), c.rank());
    let rank = a.rank();
    let a_m = m_axis.inputs[0][0];
    let a_k = k_axis.inputs[0][0];
    let b_k = k_axis.inputs[1][0];
    let b_n = n_axis.inputs[1][0];
    let c_m = m_axis.outputs[0][0];
    let c_n = n_axis.outputs[0][0];
    assert!(a_m >= rank - 2);
    assert!(a_k >= rank - 2);
    assert!(b_k >= rank - 2);
    assert!(b_n >= rank - 2);
    assert!(c_m >= rank - 2);
    assert!(c_n >= rank - 2);
    let transpose_a = a_m > a_k;
    let transpose_b = b_k > b_n;
    let transpose_c = c_m > c_n;
    Ok(Some(TypedModelPatch::replace_single_op(
        model,
        node,
        &node.inputs,
        NumPyMatMul { transpose_a, transpose_b, transpose_c, output_fact: c.clone() },
    )?))
}

#[derive(Clone, Debug)]
pub struct NumPyMatMul {
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub transpose_c: bool,
    pub output_fact: TypedFact,
}

impl Op for NumPyMatMul {
    fn name(&self) -> Cow<str> {
        "MatMul".into()
    }

    op_as_typed_op!();
}

impl EvalOp for NumPyMatMul {
    fn is_stateless(&self) -> bool {
        false
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        bail!("This op is not meant be evaled.")
    }
}

impl TypedOp for NumPyMatMul {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.output_fact.clone()))
    }

    as_op!();
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! e {
        ($lit:literal) => {
            EinSum::new($lit.parse().unwrap(), i32::datum_type())
        };
    }

    #[test]
    fn decompose_matmul() {
        let mut model = TypedModel::default();
        let a = model.add_source("a", i32::fact(&[2, 3])).unwrap();
        let b = model.add_source("b", i32::fact(&[3, 4])).unwrap();
        let einsum = model.wire_node("einsum", e!("mk,kn->mn"), &[a, b]).unwrap();
        model.set_output_outlets(&einsum).unwrap();
        decompose_einsums_in_place(&mut model).unwrap();
        assert!(model.nodes.iter().all(|n| !n.op_is::<EinSum>()));
    }
    
}
