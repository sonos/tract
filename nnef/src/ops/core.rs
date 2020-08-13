use crate::internal::*;
use crate::ser::*;
use tract_core::ops;

pub fn register(registry: &mut Registry) {
    registry.register_unit_element_wise("tract_core_tan", &ops::math::Tan {});
    registry.register_unit_element_wise("tract_core_acos", &ops::math::Acos {});
    registry.register_unit_element_wise("tract_core_asin", &ops::math::Asin {});
    registry.register_unit_element_wise("tract_core_atan", &ops::math::Atan {});
    registry.register_unit_element_wise("tract_core_cosh", &ops::math::Cosh {});
    registry.register_unit_element_wise("tract_core_sinh", &ops::math::Sinh {});
    registry.register_unit_element_wise("tract_core_acosh", &ops::math::Acosh {});
    registry.register_unit_element_wise("tract_core_asinh", &ops::math::Asinh {});
    registry.register_unit_element_wise("tract_core_atanh", &ops::math::Atanh {});

    registry.register_unit_element_wise("tract_core_round_even", &ops::math::RoundHalfToEven {});

    registry.register_binary("tract_core_xor", &ops::logic::Xor {});

    registry.register_dumper(TypeId::of::<ops::array::MultiBroadcastTo>(), ser_broadcast);
    registry.register_primitive(
        "tract_core_broadcast",
        &[TypeName::Scalar.tensor().named("input"), TypeName::Integer.array().named("shape")],
        de_broadcast,
    );

    registry.register_dumper(TypeId::of::<ops::Downsample>(), ser_downsample);
    registry.register_primitive(
        "tract_core_downsample",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Integer.named("stride"),
            TypeName::Integer.named("modulo").default(0),
        ],
        de_downsample,
    );

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

fn ser_downsample(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op().downcast_ref::<ops::Downsample>().unwrap();
    let wire = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_downsample",
        &[wire],
        &[
            ("axis", numeric(op.axis)),
            ("stride", numeric(op.stride)),
            ("modulo", numeric(op.modulo)),
        ],
    )))
}

fn de_downsample(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let stride = invocation.named_arg_as::<i64>(builder, "stride")? as isize;
    let modulo = invocation.named_arg_as(builder, "modulo")?;
    builder.wire(ops::Downsample { axis, stride, modulo }, &[wire])
}

fn ser_broadcast(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op().downcast_ref::<ops::array::MultiBroadcastTo>().unwrap();
    let wire = ast.mapping[&node.inputs[0]].clone();
    let shape = op
        .shape
        .iter()
        .map(|d| d.to_integer().map(|x| x as usize))
        .collect::<TractResult<TVec<usize>>>()?;
    Ok(Some(invocation("tract_core_broadcast", &[wire], &[("shape", ints(&shape))])))
}

fn de_broadcast(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let shape = invocation.named_arg_as(builder, "shape")?;
    builder.wire(ops::array::MultiBroadcastTo { shape }, &[wire])
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
