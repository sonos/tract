use tract_core::ops::array::Gather;

use crate::internal::*;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_gather);
    registry.register_primitive(
        "tract_core_gather",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Scalar.tensor().named("indices"),
            TypeName::Integer.named("axis"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_gather,
    );

    macro_rules! gather_op_nnef {
        ($GatherOp:ty, $name:ident, $field_name:ident) => {
            mod $name {
                use crate::internal::*;

                pub fn ser_gather(
                    ast: &mut IntoAst,
                    node: &TypedNode,
                    op: &$GatherOp,
                ) -> TractResult<Option<Arc<RValue>>> {
                    let wire = ast.mapping[&node.inputs[0]].clone();
                    let indices = ast.mapping[&node.inputs[1]].clone();
                    Ok(Some(invocation(
                        concat!("tract_core_", stringify!($name)),
                        &[wire, indices],
                        &[(stringify!($field_name), numeric(op.$field_name))],
                    )))
                }

                pub fn de_gather(
                    builder: &mut ModelBuilder,
                    invocation: &ResolvedInvocation,
                ) -> TractResult<Value> {
                    let wire = invocation.named_arg_as(builder, "input")?;
                    let indices = invocation.named_arg_as(builder, "indices")?;
                    let cast_indices = builder.wire_as_outlets(
                        tract_core::ops::cast::cast(i64::datum_type()),
                        &[indices],
                    )?[0];
                    let value = invocation.named_arg_as(builder, stringify!($field_name))?;
                    builder.wire((<$GatherOp>::new(value)), &[wire, cast_indices])
                }
            }
            registry.register_dumper($name::ser_gather);
            registry.register_primitive(
                concat!("tract_core_", stringify!($name)),
                &[
                    TypeName::Scalar.tensor().named("input"),
                    TypeName::Scalar.tensor().named("indices"),
                    TypeName::Integer.named(stringify!($field_name)),
                ],
                &[("output", TypeName::Scalar.tensor())],
                $name::de_gather,
            );
        };
    }
    gather_op_nnef!(tract_core::ops::array::GatherElements, gather_elements, axis);
    gather_op_nnef!(tract_core::ops::array::GatherNd, gather_nd, batch_dims);
}

pub fn ser_gather(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &Gather,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let indices = ast.mapping[&node.inputs[1]].clone();
    let mut named_args = tvec!(("axis", numeric(op.axis)));
    if let Some(dt) = op.output_type {
        named_args.push(("datum_type", string(format!("{dt:?}"))));
    }
    Ok(Some(invocation("tract_core_gather", &[wire, indices], &named_args)))
}

pub fn de_gather(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let indices = invocation.named_arg_as(builder, "indices")?;
    let output_type: Option<DatumType> = invocation
        .optional_named_arg_as::<String>(builder, "datum_type")?
        .map(|s| s.parse())
        .transpose()?;
    let cast_indices =
        builder.wire_as_outlets(tract_core::ops::cast::cast(i64::datum_type()), &[indices])?[0];
    let axis = invocation.named_arg_as(builder, "axis")?;
    builder.wire(Gather { axis, output_type }, &[wire, cast_indices])
}
