use crate::internal::*;
use crate::ser::*;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<tract_core::pulse::delay::Delay>(), ser_delay);
    registry.register_primitive(
        "tract_core_delay",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Integer.named("delay"),
            TypeName::Integer.named("overlap"),
        ],
        de_delay,
    );
}

fn ser_delay(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op().downcast_ref::<tract_core::pulse::delay::Delay>().unwrap();
    let wire = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_delay",
        &[wire],
        &[
            ("axis", numeric(op.axis)),
            ("delay", numeric(op.delay)),
            ("overlap", numeric(op.overlap)),
        ],
    )))
}

fn de_delay(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as::<i64>(builder, "axis")? as usize;
    let delay = invocation.named_arg_as::<i64>(builder, "delay")? as usize;
    let overlap = invocation.named_arg_as::<i64>(builder, "overlap")? as usize;
    let input_fact = builder.model.outlet_fact(wire)?;
    let op = tract_core::pulse::delay::Delay::new_typed(input_fact, axis, delay, overlap)?;
    builder.wire(op, &[wire])
}
