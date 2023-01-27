use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_hir::internal::*;

use tract_onnx_opl::lrn::Lrn;

pub fn lrn(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let alpha = node.get_attr_opt("alpha")?.unwrap_or(0.0001);
    let beta = node.get_attr_opt("beta")?.unwrap_or(0.75);
    let bias = node.get_attr_opt("bias")?.unwrap_or(1.);
    let size = node.get_attr("size")?;
    Ok((inference_wrap(Lrn { alpha, beta, bias, size }, 1, lrn_rules), vec![]))
}

fn lrn_rules<'p>(
    _op: &dyn Op,
    s: &mut Solver,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    check_input_arity(inputs, 1)?;
    check_output_arity(outputs, 1)?;
    s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
    s.equals(&inputs[0].shape, &outputs[0].shape)?;
    Ok(())
}
