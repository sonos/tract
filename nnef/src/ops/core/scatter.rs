use crate::internal::*;
use tract_core::ops::array::ScatterElements;

pub fn register(registry: &mut Registry) {
    use crate::internal::*;

    pub fn ser_scatter_elements(
        ast: &mut IntoAst,
        node: &TypedNode,
    ) -> TractResult<Option<Arc<RValue>>> {
        let op = node.op().downcast_ref::<ScatterElements>().unwrap();
        let wire = ast.mapping[&node.inputs[0]].clone();
        let indices = ast.mapping[&node.inputs[1]].clone();
        let updates = ast.mapping[&node.inputs[2]].clone();
        Ok(Some(invocation(
            "tract_core_scatter_elements",
            &[wire, indices, updates],
            &[("axis", numeric(op.axis))],
        )))
    }

    pub fn de_scatter_elements(
        builder: &mut ModelBuilder,
        invocation: &ResolvedInvocation,
    ) -> TractResult<TVec<OutletId>> {
        let wire = invocation.named_arg_as(builder, "input")?;
        let indices = invocation.named_arg_as(builder, "indices")?;
        let updates = invocation.named_arg_as(builder, "updates")?;
        let axis = invocation.named_arg_as(builder, "axis")?;
        builder.wire(ScatterElements::new(axis), &[wire, indices, updates])
    }

    registry.register_dumper(TypeId::of::<ScatterElements>(), ser_scatter_elements);
    registry.register_primitive(
        "tract_core_scatter_elements",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Scalar.tensor().named("indices"),
            TypeName::Scalar.tensor().named("updates"),
            TypeName::Integer.named("axis"),
        ],
        de_scatter_elements,
    );
}
