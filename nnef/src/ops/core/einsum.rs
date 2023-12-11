use crate::internal::*;
use crate::ser::*;
use tract_core::ops::einsum::EinSum;
use tract_core::tract_data::itertools::Itertools;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser);
    registry.register_primitive(
        "tract_core_einsum",
        &parameters(),
        &[("output", TypeName::Scalar.tensor())],
        de_einsum,
    );
    registry.register_primitive(
        "tract_core_einsum_q",
        &parameters_q(),
        &[("output", TypeName::Scalar.tensor())],
        de_einsum_q,
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

pub fn parameters_q() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().array().named("inputs"),
        TypeName::String.named("expr"),
        TypeName::String.named("acc"),
        TypeName::String.named("output").default(""),
        TypeName::Scalar.tensor().named("bias").default(0),
        TypeName::Integer.tensor().named("a0"),
        TypeName::Scalar.tensor().named("a_scale"),
        TypeName::Integer.tensor().named("b0"),
        TypeName::Scalar.tensor().named("b_scale"),
        TypeName::Integer.tensor().named("c0"),
        TypeName::Scalar.tensor().named("c_scale"),
    ]
}

pub fn ser(ast: &mut IntoAst, node: &TypedNode, op: &EinSum) -> TractResult<Option<Arc<RValue>>> {
    if op.q_params.is_some() {
        ser_einsum_q(ast, node)
    } else {
        ser_einsum(ast, node)
    }
}

pub fn ser_einsum(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let einsum = node.op_as::<EinSum>().unwrap();
    let inputs: Vec<_> = node.inputs.iter().map(|i| (*ast.mapping[i]).clone()).collect();
    Ok(Some(invocation(
        "tract_core_einsum",
        &[Arc::new(RValue::Array(inputs))],
        &[
            ("expr", string(einsum.axes.to_string())),
            ("acc", datum_type(einsum.operating_dt)),
            ("output", einsum.q_params.map(datum_type).unwrap_or_else(|| string(""))),
        ],
    )))
}

pub fn ser_einsum_q(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let einsum = node.op_as::<EinSum>().unwrap();
    let inputs = node.inputs.iter().map(|i| (*ast.mapping[i]).clone()).collect_vec();
    Ok(Some(invocation(
        "tract_core_einsum_q",
        &[Arc::new(RValue::Array(vec![inputs[0].clone(), inputs[1].clone()]))],
        &[
            ("expr", string(einsum.axes.to_string())),
            ("acc", datum_type(einsum.operating_dt)),
            ("output", einsum.q_params.map(datum_type).unwrap_or_else(|| string(""))),
            ("bias", inputs[2].clone()),
            ("a0", inputs[3].clone()),
            ("a_scale", inputs[4].clone()),
            ("b0", inputs[5].clone()),
            ("b_scale", inputs[6].clone()),
            ("c0", inputs[7].clone()),
            ("c_scale", inputs[8].clone()),
        ],
    )))
}

pub fn de_einsum(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let expr = invocation.named_arg_as::<String>(builder, "expr")?.parse::<AxesMapping>()?;
    let inputs: TVec<OutletId> = invocation.named_arg_as(builder, "inputs")?;
    let operating_dt = invocation.named_arg_as::<String>(builder, "acc")?;
    let operating_dt = operating_dt.parse()?;
    let einsum = EinSum::new(expr, operating_dt);
    builder.wire(einsum, &inputs)
}

pub fn de_einsum_q(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let expr = invocation.named_arg_as::<String>(builder, "expr")?.parse::<AxesMapping>()?;
    let mut inputs: TVec<OutletId> = invocation.named_arg_as(builder, "inputs")?;
    for qp in parameters_q().iter().skip(4) {
        inputs.push(invocation.named_arg_as(builder, &qp.id.0)?);
    }
    let operating_dt = invocation.named_arg_as::<String>(builder, "acc")?;
    let operating_dt = operating_dt.parse()?;
    let output_dt = if let Some(odt) =
        invocation.get_named_arg_as::<String>(builder, "output")?.filter(|odt| odt.len() > 0)
    {
        odt.parse()?
    } else {
        bail!("Expected an output type for tract_core_einsum_q")
    };
    let einsum = EinSum::newq(expr, operating_dt, output_dt);
    builder.wire(einsum, &inputs)
}
