use crate::internal::*;
use crate::ser::*;
use std::any::TypeId;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::nn::gelu_approximate::GeluApproximate;

fn parameters() -> Vec<Parameter> {
    vec![TypeName::Scalar.tensor().named("input"), TypeName::Logical.named("fast_impl")]
}

fn dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op_as::<ElementWiseOp>().unwrap().0.downcast_ref::<GeluApproximate>().unwrap();
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_gelu_approx",
        &[input],
        &[("fast_impl", logical(op.fast_impl))],
    )))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let fast_impl = invocation.named_arg_as(builder, "fast_impl")?;
    builder.wire(ElementWiseOp(Box::new(GeluApproximate { fast_impl }), None), &[input])
}

pub fn register(registry: &mut Registry) {
    registry.register_element_wise(
        "tract_core_gelu_approx",
        TypeId::of::<GeluApproximate>(),
        Box::new(dump),
        parameters(),
        load,
    );
    // Backward compatibility alias
    registry.register_element_wise(
        "tract_transformers_gelu_approx",
        TypeId::of::<GeluApproximate>(),
        Box::new(dump),
        parameters(),
        load,
    );
}
