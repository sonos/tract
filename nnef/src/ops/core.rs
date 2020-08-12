use crate::ast::parse::parse_parameters;
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
        &parse_parameters("input: tensor<scalar>, dims:integer[]").unwrap(),
        de_broadcast,
    );

    registry.register_dumper(TypeId::of::<ops::Downsample>(), ser_downsample);
    registry.register_primitive(
        "tract_core_downsample",
        &parse_parameters(
            "input: tensor<scalar>, axis: integer, stride:integer, modulo:integer = 0",
        )
        .unwrap(),
        de_downsample,
    );
}

pub fn ser_downsample(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Arc<RValue>> {
    let op = node.op().downcast_ref::<ops::Downsample>().unwrap();
    let wire = ast.mapping[&node.inputs[0]].clone();
    Ok(invocation(
        "tract_core_downsample",
        &[wire],
        &[
            ("axis", numeric(op.axis)),
            ("stride", numeric(op.stride)),
            ("modulo", numeric(op.modulo)),
        ],
    ))
}

pub fn de_downsample(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let stride = invocation.named_arg_as::<i64>(builder, "stride")? as isize;
    let modulo = invocation.named_arg_as(builder, "modulo")?;
    builder.wire(ops::Downsample { axis, stride, modulo }, &[wire])
}

pub fn ser_broadcast(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Arc<RValue>> {
    let op = node.op().downcast_ref::<ops::array::MultiBroadcastTo>().unwrap();
    let wire = ast.mapping[&node.inputs[0]].clone();
    let shape = op
        .shape
        .iter()
        .map(|d| d.to_integer().map(|x| x as usize))
        .collect::<TractResult<TVec<usize>>>()?;
    Ok(invocation("tract_core_broadcast", &[wire], &[("shape", ints(&shape))]))
}

pub fn de_broadcast(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let shape = invocation.named_arg_as(builder, "shape")?;
    builder.wire(ops::array::MultiBroadcastTo { shape }, &[wire])
}

