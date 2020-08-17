use crate::ast;
use crate::internal::*;
use crate::ser::*;
use tract_core::ops;
use tract_core::ops::scan::*;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<ops::scan::Scan>(), ser_scan);
    registry.register_primitive(
        "tract_core_scan",
        &[
            TypeName::String.named("body"),
            ast::TypeSpec::Tuple(vec![
                TypeName::String.spec(),   // body param name
                TypeName::Scalar.tensor(), // input
                TypeName::Integer.spec(),  // axis
                TypeName::Integer.spec(),  // step
            ])
            .array()
            .named("scan"),
            ast::TypeSpec::Tuple(vec![
                TypeName::String.spec(),   // body param name
                TypeName::Scalar.tensor(), // input
            ])
            .array()
            .named("full"),
            ast::TypeSpec::Tuple(vec![
                TypeName::String.spec(), // body param name
                TypeName::String.spec(), // "all" or "last"
            ])
            .array()
            .named("output"),
            TypeName::Integer.spec()    // if present, assumes B is first axis in all inputs
            .named("seq_length")
            .default(-1),
            TypeName::Integer.spec()    // if present, assumes B is first axis in all inputs
            .named("skip")
            .default(0),                // needed for pulse
        ],
        de_scan,
    );
}

fn ser_scan(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op().downcast_ref::<Scan>().unwrap();
    let body = crate::ser::to_fragment_def(ast, &op.body)?;
    crate::ast::dump::Dumper::new(&mut std::io::stderr()).fragments(&[body]);
    /*
    let mut inputs = vec!();
    let mut scan = vec!();
    let mut full = vec!();
    for (ix, input) in op.input_mapping.iter().() {
        match input {
            InputMapping::Scan { slot, axis, chunk } => {
                scans.push(RValue::Tuple(vec!()))
            }
        }
    }
    */
    todo!();
    /*
    let wire = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_scan",
        &[wire],
        &[
            ("axis", numeric(op.axis)),
            ("stride", numeric(op.stride)),
            ("modulo", numeric(op.modulo)),
        ],
    )))
    */
}

fn de_scan(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let stride = invocation.named_arg_as::<i64>(builder, "stride")? as isize;
    let modulo = invocation.named_arg_as(builder, "modulo")?;
    builder.wire(ops::Downsample { axis, stride, modulo }, &[wire])
}
