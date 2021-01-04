use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;
use tract_hir::internal::*;
use tract_onnx_opl::ml::CategoryMapper;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("CategoryMapper", category_mapper);
}

fn category_mapper(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let ints: Vec<i64> = node.get_attr_vec("cats_int64s")?;
    let strings: Vec<String> = node.get_attr_vec("cats_strings")?;
    let default_int: Option<i64> = node.get_attr_opt("default_int64")?;
    let default_string: Option<String> = node.get_attr_opt("default_string")?;
    let op: Box<dyn InferenceOp> = match (default_int, default_string.as_ref()) {
        (None, None) | (Some(_), Some(_)) => bail!(
            "CategoryMapper requires exactly one of default_int64 and default_string (found {:?})",
            (default_int, default_string)
        ),
        (Some(def), None) => inference_wrap(
            CategoryMapper { hash: tract_itertools::zip(strings, ints).collect(), default: def },
            1,
            rules,
        ),
        (None, Some(def)) => inference_wrap(
            CategoryMapper {
                hash: tract_itertools::zip(ints, strings).collect(),
                default: def.to_string(),
            },
            1,
            rules,
        ),
    };
    Ok((op, vec![]))
}

fn rules<'r, 'p, 's>(
    op: &'s dyn Op,
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    let (src, dst) = if op.downcast_ref::<CategoryMapper<i64, String>>().is_some() {
        (i64::datum_type(), String::datum_type())
    } else {
        (String::datum_type(), i64::datum_type())
    };
    check_input_arity(&inputs, 1)?;
    check_output_arity(&outputs, 1)?;
    s.equals(&inputs[0].shape, &outputs[0].shape)?;
    s.equals(&inputs[0].datum_type, src)?;
    s.equals(&outputs[0].datum_type, dst)?;
    Ok(())
}
