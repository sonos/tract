use crate::deser::Value;
use crate::internal::*;
use crate::ops::nnef::deser::read_conv_parameters;
use crate::ops::nnef::ser::make_conv_named_args;
use crate::ops::nnef::ser::ser_axis_op;
use crate::ser::*;
use tract_core::ops::cnn::ConvUnary;
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
    op: &ConvUnary,
) -> TractResult<Option<Arc<RValue>>> {
    if op.q_params.is_none() || node.outputs[0].fact.datum_type.is_quantized() {
        return Ok(None);
    }
    let name = &node.name;
    let mut named_args = make_conv_named_args(node, &op.pool_spec, op.group, false, None)?;

    for (ix, name) in ["a0", "a_scale", "b0", "b_scale", "c0", "c_scale"].iter().enumerate() {
        named_args.push((name, (*ast.mapping[&node.inputs[1 + ix]]).clone()));
    }

    let mut wire = ast.mapping[&node.inputs[0]].clone();
    let mut weights = ast.konst_variable(format!("{name}_weights"), &op.kernel)?;
    let mut rank = op.kernel.rank();
    for fix in op.kernel_fmt.kernel_as_group_o_i_h_w_ops(op.kernel.shape(), op.group) {
        weights = ser_axis_op(&fix, weights, rank);
        match fix {
            AxisOp::Add(_) => rank += 1,
            AxisOp::Rm(_) => rank -= 1,
            AxisOp::Move(_, _) => (),
            AxisOp::Reshape(_, before, after) => rank = rank + after.len() - before.len(),
        }
    }
    weights = ser_axis_op(
        &AxisOp::Reshape(
            0,
            tvec!(op.group.to_dim(), op.pool_spec.output_channels.to_dim() / op.group),
            tvec!(op.pool_spec.output_channels.to_dim()),
        ),
        weights,
        rank,
    );
    wire = ast.force_variable(format!("{name}_input"), &wire);

    let mut inputs = tvec![wire, weights];
    if let Some(bias) = op.bias.as_ref() {
        let bias = ast.konst(format!("{name}_bias"), bias)?;
        inputs.push(bias)
    }

    Ok(Some(invocation("tract_core_qconv", &inputs, &named_args)))
}

fn qconv_load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let mut inputs: TVec<OutletId> = tvec!(invocation.named_arg_as(builder, "input")?);
    let kernel: Arc<Tensor> = invocation.named_arg_as(builder, "filter")?;

    let input_fact = builder.model.outlet_fact(inputs[0])?.clone();
    if input_fact.rank() != kernel.rank() {
        bail!(
            "Convolution input expected as NCHW, filter as OIHW. Got {:?} and {:?}.",
            input_fact,
            kernel
        );
    }

    let (group, pool_spec) =
        read_conv_parameters(builder, invocation, kernel.shape(), &input_fact)?;

    let qparams = qparams_as_outlets(builder, invocation).context("Loading qparams")?;
    inputs.extend(qparams.iter().cloned());
    let bias: Arc<Tensor> = invocation.named_arg_as(builder, "bias")?;

    let bias: Option<Arc<Tensor>> =
        if bias.is_uniform() && bias.cast_to_scalar::<f32>()? == 0.0 { None } else { Some(bias) };

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

    let op: Box<dyn TypedOp> = Box::new(ConvUnary::new(
        pool_spec,
        KernelFormat::OIHW,
        kernel.clone(),
        group,
        bias,
        Some(output_dt),
    ));

    builder.wire(op, &inputs)
}
