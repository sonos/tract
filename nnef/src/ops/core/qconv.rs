use crate::deser::Value;
use crate::internal::*;
use crate::ops::nnef::deser::read_conv_parameters;
use crate::ops::nnef::ser::make_conv_named_args;
use crate::ser::*;
use tract_core::ops::cnn::Conv;
use tract_core::ops::cnn::KernelFormat;

use super::qmatmul::qparams_as_outlets;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(qconv_unary_dump);
    registry.register_primitive(
        "tract_core_qconv",
        &qconv_parameters(),
        &[("output", TypeName::Scalar.tensor())],
        qconv_load,
    );
}

fn qconv_parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Scalar.tensor().named("filter"),
        TypeName::Scalar.tensor().named("bias").default(0),
        TypeName::Integer.spec().named("group"),
        TypeName::Integer.array().named("dilation"),
        TypeName::Integer.array().named("stride"),
        TypeName::Integer.array().array().named("padding"),
        TypeName::String.spec().named("border"),
        TypeName::Integer.spec().named("a0"),
        TypeName::Scalar.spec().named("a_scale"),
        TypeName::Integer.spec().named("b0"),
        TypeName::Scalar.spec().named("b_scale"),
        TypeName::Integer.spec().named("c0"),
        TypeName::Scalar.spec().named("c_scale"),
    ]
}

fn qconv_unary_dump(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &Conv,
) -> TractResult<Option<Arc<RValue>>> {
    if op.q_params.is_none() || node.outputs[0].fact.datum_type.is_quantized() {
        return Ok(None);
    }
    let mut named_args = make_conv_named_args(node, &op.pool_spec, op.group, false, None)?;

    for (ix, name) in ["b0", "b_scale", "a0", "a_scale", "c0", "c_scale"].iter().enumerate() {
        named_args.push((name, (*ast.mapping[&node.inputs[3 + ix]]).clone()));
    }

    let wire = ast.mapping[&node.inputs[0]].clone();
    ensure!(op.kernel_fmt == KernelFormat::OIHW);
    let weights = ast.mapping[&node.inputs[1]].clone();
    let bias = ast.mapping[&node.inputs[2]].clone();
    let inputs = tvec![wire, weights, bias];

    Ok(Some(invocation("tract_core_qconv", &inputs, &named_args)))
}

fn qconv_load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let mut inputs: TVec<OutletId> = tvec!(invocation.named_arg_as(builder, "input")?);
    inputs.push(invocation.named_arg_as(builder, "filter")?);
    inputs.push(invocation.named_arg_as(builder, "bias")?);

    let input_fact = builder.model.outlet_fact(inputs[0])?.clone();
    let kernel_fact = builder.model.outlet_fact(inputs[1])?.clone();

    if input_fact.rank() != kernel_fact.rank() {
        bail!(
            "Convolution input expected as NCHW, filter as OIHW. Got {:?} and {:?}.",
            input_fact,
            kernel_fact
        );
    }

    let (group, pool_spec) = read_conv_parameters(
        builder,
        invocation,
        kernel_fact.shape.as_concrete().context("Expect fixed size kernel")?,
        &input_fact,
    )?;

    let mut qparams = qparams_as_outlets(builder, invocation).context("Loading qparams")?;
    qparams.swap(0, 2);
    qparams.swap(1, 3);
    inputs.extend(qparams.iter().cloned());

    let Some(c0) = &builder.model.outlet_fact(qparams[4])?.konst else {
        bail!("For quantized convolution, output quantization must be static");
    };
    let Some(c_scale) = &builder.model.outlet_fact(qparams[5])?.konst else {
        bail!("For quantized convolution, output quantization must be static");
    };
    let output_dt = input_fact.datum_type.with_qparams(QParams::ZpScale {
        zero_point: c0.cast_to_scalar()?,
        scale: c_scale.cast_to_scalar()?,
    });

    let op: Box<dyn TypedOp> =
        Box::new(Conv::new(pool_spec, KernelFormat::OIHW, group, Some(output_dt)));

    builder.wire(op, &inputs)
}
