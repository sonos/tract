use crate::internal::*;

pub fn register(registry: &mut Registry) {
    macro_rules! gather_op_nnef {
        ($GatherOp:ty, $name:ident, $field_name:ident) => {
            mod $name {
                use crate::internal::*;

                pub fn ser_gather(
                    ast: &mut IntoAst,
                    node: &TypedNode,
                ) -> TractResult<Option<Arc<RValue>>> {
                    let op = node.op().downcast_ref::<$GatherOp>().unwrap();
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
                ) -> TractResult<TVec<OutletId>> {
                    let wire = invocation.named_arg_as(builder, "input")?;
                    let indices = invocation.named_arg_as(builder, "indices")?;
                    let value = invocation.named_arg_as(builder, stringify!($field_name))?;
                    builder.wire((<$GatherOp>::new(value)), &[wire, indices])
                }
            }
            registry.register_dumper(TypeId::of::<$GatherOp>(), $name::ser_gather);
            registry.register_primitive(
                concat!("tract_core_", stringify!($name)),
                &[
                    TypeName::Scalar.tensor().named("input"),
                    TypeName::Scalar.tensor().named("indices"),
                    TypeName::Integer.named(stringify!($field_name)),
                ],
                $name::de_gather,
            );
        };
    }

    gather_op_nnef!(tract_core::ops::array::Gather, gather, axis);
    gather_op_nnef!(tract_core::ops::array::GatherElements, gather_elements, axis);
    gather_op_nnef!(tract_core::ops::array::GatherNd, gather_nd, batch_dims);
}
