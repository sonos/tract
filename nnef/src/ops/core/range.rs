use crate::internal::*;
use crate::ser::*;
use tract_core::ops::array::Range;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(range_dump);
    registry.register_primitive(
        "tract_core_range",
        &range_parameters(),
        &[("output", TypeName::Scalar.tensor())],
        range_load,
    );
}

fn range_parameters() -> Vec<Parameter> {
    vec![
        TypeName::Integer.named("start"),
        TypeName::Integer.named("end"),
        TypeName::Integer.named("step"),
    ]
}

fn range_dump(ast: &mut IntoAst, node: &TypedNode, _: &Range) -> TractResult<Option<Arc<RValue>>> {
    let start = ast.mapping[&node.inputs[0]].clone();
    let end = ast.mapping[&node.inputs[1]].clone();
    let step = ast.mapping[&node.inputs[2]].clone();

    Ok(Some(invocation("tract_core_range", &[start, end, step], &[])))
}

fn range_load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let start: OutletId = invocation.named_arg_as(builder, "start")?;
    let end: OutletId = invocation.named_arg_as(builder, "end")?;
    let step: OutletId = invocation.named_arg_as(builder, "step")?;

    // If start/end/step are all constant TDim scalars, compute the length symbolically
    // so that self.len carries the correct expression (e.g. T_tokens) rather than a
    // fresh free symbol.  This matters for pulsification: NonPulsingWrappingOp drops
    // konst when converting PulsedFact → TypedFact, so output_facts falls back to
    // self.len for the output shape — a free symbol would contaminate pulsed shapes.
    let len: TDim = {
        let sf = builder.model.outlet_fact(start)?;
        let ef = builder.model.outlet_fact(end)?;
        let kf = builder.model.outlet_fact(step)?;
        if let (Some(s), Some(e), Some(k)) = (&sf.konst, &ef.konst, &kf.konst) {
            if s.datum_type() == TDim::datum_type() {
                if let (Ok(s_tdim), Ok(e_tdim), Ok(step_val)) = (
                    s.try_as_plain().and_then(|t| t.to_scalar::<TDim>()).map(|d| d.clone()),
                    e.try_as_plain().and_then(|t| t.to_scalar::<TDim>()).map(|d| d.clone()),
                    k.cast_to_scalar::<i64>(),
                ) {
                    if step_val > 0 {
                        (e_tdim - s_tdim).divceil(step_val as usize)
                    } else {
                        builder.model.symbols.new_with_prefix("range").into()
                    }
                } else {
                    builder.model.symbols.new_with_prefix("range").into()
                }
            } else {
                builder.model.symbols.new_with_prefix("range").into()
            }
        } else {
            builder.model.symbols.new_with_prefix("range").into()
        }
    };
    builder.wire(Range::new(len), &[start, end, step])
}
