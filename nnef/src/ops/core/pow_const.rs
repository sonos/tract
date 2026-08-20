use crate::internal::*;
use crate::ser::*;
use std::any::TypeId;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::math::PowConst;

fn parameters() -> Vec<Parameter> {
    vec![TypeName::Scalar.tensor().named("input"), TypeName::Scalar.named("exponent")]
}

fn dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op_as::<ElementWiseOp>().unwrap().0.downcast_ref::<PowConst>().unwrap();
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation("tract_core_pow_const", &[input], &[("exponent", numeric(op.exponent))])))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let exponent = invocation.named_arg_as(builder, "exponent")?;
    builder.wire(ElementWiseOp(Box::new(PowConst { exponent }), None), &[input])
}

pub fn register(registry: &mut Registry) {
    registry.register_element_wise(
        "tract_core_pow_const",
        TypeId::of::<PowConst>(),
        Box::new(dump),
        parameters(),
        load,
    );
}
