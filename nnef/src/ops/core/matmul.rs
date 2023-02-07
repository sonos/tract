use crate::internal::*;
use crate::ser::*;
use tract_core::ops::matmul::mir::MatMul;
use tract_core::ops::matmul::MatMulAxes;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<tract_core::ops::matmul::mir::MatMul>(), matmul_dump);
    registry.register_primitive(
        "tract_core_matmul", 
        &matmul_parameters(), 
        &[("output", TypeName::Scalar.tensor())],
        matmul_load
    );
}

fn matmul_parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("A"),
        TypeName::Scalar.tensor().named("B"),
        TypeName::Integer.array().named("axes"),
    ]
}

pub fn matmul_dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op_as::<MatMul>().unwrap();
    let a = ast.force_variable(format!("{}_a", node.name), &ast.mapping[&node.inputs[0]].clone());
    let b = ast.force_variable(format!("{}_b", node.name), &ast.mapping[&node.inputs[1]].clone());
    let c = invocation("tract_core_matmul", &[a, b], &[("axes", ints(&op.axes.to_array()))]);
    Ok(Some(ast.force_variable(&node.name, &c)))
}

fn matmul_load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let a: OutletId = invocation.named_arg_as(builder, "A")?;
    let b: OutletId = invocation.named_arg_as(builder, "B")?;
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let axes = MatMulAxes::from_array(&axes)?;
    builder.wire(MatMul { axes }, &[a, b])
}
