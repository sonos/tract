use crate::internal::*;
use crate::ser::*;
use tract_core::ops::nn::{Reduce, Reducer};

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_reduce);
    for red in &[
        "tract_core_argmax_reduce_last",
        "tract_core_argmin_reduce_last",
        "tract_core_product_reduce",
    ] {
        registry.register_primitive(
            red,
            &[TypeName::Scalar.tensor().named("input"), TypeName::Integer.array().named("axes")],
            &[("output", TypeName::Scalar.tensor())],
            de_reduce,
        );
    }
}

fn ser_reduce(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &Reduce,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let oper = match op.reducer {
        Reducer::ArgMax(last) if last => "tract_core_argmax_reduce_last",
        Reducer::ArgMin(last) if last => "tract_core_argmin_reduce_last",
        Reducer::Prod => "tract_core_product_reduce",
        _ => return Ok(None),
    };
    Ok(Some(invocation(oper, &[wire], &[("axes", ints(&op.axes))])))
}

fn de_reduce(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let reducer = match &*invocation.invocation.id.0 {
        "tract_core_argmin_reduce_last" => Reducer::ArgMin(true),
        "tract_core_argmax_reduce_last" => Reducer::ArgMax(true),
        "tract_core_product_reduce" => Reducer::Prod,
        _ => panic!(),
    };
    let axes = invocation.named_arg_as(builder, "axes")?;
    let reduce = Reduce { axes, reducer };
    builder.wire(reduce, &[wire])
}
