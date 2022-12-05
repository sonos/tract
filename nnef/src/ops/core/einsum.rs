use crate::internal::*;
use crate::ser::*;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::einsum::Expr;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<EinSum>(), dump);
    registry.register_primitive(
        "tract_core_einsum",
        &parameters(),
        &[("output", TypeName::Scalar.tensor())],
        load,
    );
}

pub fn parameters() -> Vec<Parameter> {
    vec![TypeName::Scalar.tensor().array().named("inputs"), TypeName::String.named("expr")]
}

pub fn dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let einsum = node.op_as::<EinSum>().unwrap();
    let inputs = node.inputs.iter().map(|i| (*ast.mapping[i]).clone()).collect();
    Ok(Some(invocation(
        "tract_core_einsum",
        &[Arc::new(RValue::Array(inputs))],
        &[("expr", string(einsum.expr.to_string()))],
    )))
}

pub fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let expr = invocation.named_arg_as::<String>(builder, "expr")?.parse::<Expr>()?;
    let inputs: TVec<OutletId> = invocation.named_arg_as(builder, "inputs")?;
    let operating_dt = builder.model.outlet_fact(inputs[0])?.datum_type;
    let einsum = EinSum::new(expr, operating_dt);
    builder.wire(einsum, &inputs)
}
