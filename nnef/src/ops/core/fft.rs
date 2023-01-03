use crate::internal::*;
use tract_core::ops::fft::Fft;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<Fft>(), dump);
    registry.register_primitive(
        "tract_onnx_fft",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Logical.named("inverse"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        load,
    );
}

fn dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let op = node.op_as::<Fft>().context("wrong op")?;
    Ok(Some(invocation(
        "tract_onnx_fft",
        &[input],
        &[("axis", numeric(op.axis)), ("inverse", logical(op.inverse))],
    )))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let inverse: bool = invocation.named_arg_as(builder, "inverse")?;
    let op = Fft { axis, inverse };
    builder.wire(op, &[input])
}

