use crate::internal::*;
use crate::ser::*;
use tract_core::ops;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<ops::array::Topk>(), ser_topk);
    registry.register_primitive(
        "tract_core_topk",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.tensor().named("k"),
            TypeName::Integer.named("axis"),
            TypeName::Logical.named("largest"),
        ],
        &[("values", TypeName::Scalar.tensor()), ("indices", TypeName::Integer.tensor())],
        de_topk,
    );
}

fn ser_topk(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op().downcast_ref::<ops::array::Topk>().unwrap();
    let input = ast.mapping[&node.inputs[0]].clone();
    let k = ast.mapping[&node.inputs[1]].clone();
    Ok(Some(invocation(
        "tract_core_topk",
        &[input, k],
        &[("largest", logical(op.largest)), ("axis", numeric(op.axis))],
    )))
}

fn de_topk(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let k = invocation.named_arg_as(builder, "k")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let largest = invocation.named_arg_as(builder, "largest")?;
    let fallback_k = builder.model.symbol_table.new_with_prefix("k").into();
    builder.wire(ops::array::Topk { largest, fallback_k, axis }, &[input, k])
}
