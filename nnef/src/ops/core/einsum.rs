use crate::internal::*;
use crate::ser::*;
use tract_core::ops::einsum::EinSum;

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
    vec![
        TypeName::Scalar.tensor().array().named("inputs"),
        TypeName::String.named("expr"),
        TypeName::String.named("acc"),
        TypeName::String.named("output").default(""),
    ]
}

pub fn dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let einsum = node.op_as::<EinSum>().unwrap();
    let inputs = node.inputs.iter().map(|i| (*ast.mapping[i]).clone()).collect();
    Ok(Some(invocation(
        "tract_core_einsum",
        &[Arc::new(RValue::Array(inputs))],
        &[
            ("expr", string(einsum.expr.to_string())),
            ("acc", datum_type(einsum.operating_dt)),
            ("output", einsum.q_params.map(datum_type).unwrap_or_else(|| string(""))),
        ],
    )))
}

pub fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let expr = invocation.named_arg_as::<String>(builder, "expr")?.parse::<AxesMapping>()?;
    let inputs: TVec<OutletId> = invocation.named_arg_as(builder, "inputs")?;
    let operating_dt = invocation.named_arg_as::<String>(builder, "acc")?;
    let operating_dt = operating_dt.parse()?;
    let output_dt = if let Some(odt) =
        invocation.get_named_arg_as::<String>(builder, "output")?.filter(|odt| odt.len() > 0)
    {
        Some(odt.parse()?)
    } else {
        None
    };
    let einsum = EinSum::new(expr, operating_dt, output_dt);
    builder.wire(einsum, &inputs)
}
