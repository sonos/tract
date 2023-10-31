use tract_core::ops::array::DynSlice;

use crate::internal::*;
use crate::ser::tdim;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser);
    registry.register_primitive(
        "tract_core_dyn_slice",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("start"),
            TypeName::Integer.named("end"), //.default(0),
            TypeName::Integer.named("len"),
            TypeName::Integer.named("axis"),
            //            TypeName::Integer.named("to_the_end").default(0),
        ],
        &[("output", TypeName::Scalar.tensor())],
        deser,
    );
}

pub fn ser(ast: &mut IntoAst, node: &TypedNode, op: &DynSlice) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let start = ast.mapping[&node.inputs[1]].clone();
    let end = ast.mapping[&node.inputs[2]].clone();
    Ok(Some(invocation(
        concat!("tract_core_dyn_slice"),
        &[input, start, end],
        &[("axis", numeric(op.axis)), ("len", tdim(&op.len))],
    )))
}

pub fn deser(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let start = invocation.named_arg_as(builder, "start")?;
    let end = invocation.named_arg_as(builder, "end")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let len = invocation.named_arg_as(builder, "len")?;
    builder.wire(DynSlice { axis, len }, &[wire, start, end])
}
