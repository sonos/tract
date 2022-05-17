use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;

use tract_onnx_opl::einsum::EinSum;

pub fn einsum(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let expr = node.get_attr::<String>("equation")?.parse()?;
    Ok((inference_wrap(EinSum { expr }, 1, rules), vec![]))
}

fn rules<'r, 'p, 's>(
    op: &'s dyn Op,
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    let op = op.downcast_ref::<EinSum>().context("Wrong op")?;
    check_input_arity(inputs, op.expr.n_inputs())?;
    check_output_arity(outputs, 1)?;
    for i in inputs {
        s.equals(&i.datum_type, &outputs[0].datum_type)?;
    }
    for (input_id, input) in inputs.iter().enumerate() {
        let rank = op
            .expr
            .iter_all_axes()
            .flat_map(|axis| axis.inputs.get(input_id).map(|v| &**v).unwrap_or(&[]).iter())
            .max()
            .unwrap();
        s.equals(1 + *rank as i64, &input.rank)?;
    }
    let output_rank = op.expr.index.len();
    s.equals(output_rank as i64, &outputs[0].rank)?;
    for axis in op.expr.iter_all_axes() {
        let mut axes = vec![];
        if let Some(result) = axis.result {
            axes.push(outputs[0].shape[result].bex())
        }
        for (input_id, input_axis_positions) in axis.inputs.iter().enumerate() {
            for position in input_axis_positions {
                axes.push(inputs[input_id].shape[*position].bex());
            }
        }
        s.equals_all(axes)?;
    }
    Ok(())
}

