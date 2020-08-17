use crate::internal::*;
use crate::ser::*;
use tract_core::ops;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<ops::nn::Reduce>(), ser_reduce);
    for red in &[
        "tract_core_argmax_reduce_last",
        "tract_core_argmin_reduce_last",
        "tract_core_product_reduce",
    ] {
        registry.register_primitive(
            red,
            &[TypeName::Scalar.tensor().named("input"), TypeName::Integer.array().named("axes")],
            de_reduce,
        );
    }
}

fn ser_reduce(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op().downcast_ref::<ops::nn::Reduce>().unwrap();
    let wire = ast.mapping[&node.inputs[0]].clone();
    let oper = match op.reducer {
        ops::nn::Reducer::ArgMax(last) if last => "tract_core_argmax_reduce_last",
        ops::nn::Reducer::ArgMin(last) if last => "tract_core_argmin_reduce_last",
        ops::nn::Reducer::Prod => "tract_core_product_reduce",
        _ => return Ok(None),
    };
    Ok(Some(invocation(oper, &[wire], &[("axes", ints(&*op.axes))])))
}

fn de_reduce(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let reducer = match &*invocation.invocation.id {
        "tract_core_argmin_reduce_last" => ops::nn::Reducer::ArgMin(true),
        "tract_core_argmax_reduce_last" => ops::nn::Reducer::ArgMax(true),
        "tract_core_product_reduce" => ops::nn::Reducer::Prod,
        _ => panic!(),
    };
    let axes = invocation.named_arg_as(builder, "axes")?;
    let reduce = ops::nn::Reduce { axes, reducer };
    builder.wire(reduce, &[wire])
}
