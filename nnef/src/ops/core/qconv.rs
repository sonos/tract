use crate::deser::Value;
use crate::internal::*;
use crate::ops::nnef::deser::read_conv_parameters;
use crate::ops::nnef::ser::make_conv_named_args;
use crate::ser::*;
use tract_core::ops::cnn::ConvUnary;
use tract_core::ops::cnn::KernelFormat;
use tract_core::ops::matmul::MatMulQParams;

use super::qmatmul::qparams_as_outlets;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<tract_core::ops::cnn::ConvUnary>(), qconv_unary_dump);
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

fn qconv_unary_dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op_as::<ConvUnary>().unwrap();
    if op.q_params.is_none() || node.outputs[0].fact.datum_type.is_quantized() {
        return Ok(None);
    }
    let name = &node.name;
    let mut named_args = make_conv_named_args(node, &op.pool_spec, op.group, false, None)?;

    for (ix, (name, val)) in op.q_params.as_ref().unwrap().1.iter().enumerate() {
        let v = match val {
            AttrOrInput::Attr(t) if ix % 2 == 0 => numeric(t.cast_to_scalar::<i32>()?),
            AttrOrInput::Attr(t) => numeric(t.cast_to_scalar::<f32>()?),
            AttrOrInput::Input(i) => (*ast.mapping[&node.inputs[*i]]).clone(),
        };
        named_args.push((name, v));
    }

    let ci = op
        .pool_spec
        .data_format
        .shape(&ast.model.outlet_fact(node.inputs[0])?.shape.to_tvec())?
        .c()
        .to_usize()?;
    let output_shape = op.pool_spec.data_format.shape(node.outputs[0].fact.shape.to_tvec())?;
    let co = output_shape.c().to_usize()?;
    let mut wire = ast.mapping[&node.inputs[0]].clone();
    let mut kernel_shape = tvec!(co, ci / op.group);
    kernel_shape.extend(op.pool_spec.kernel_shape.iter().copied());
    let mut weights = op.kernel_as_group_o_ihw()?.into_tensor();
    weights.set_shape(&kernel_shape)?;
    let weigths = ast.konst_variable(format!("{name}_weigths"), &weights.into_arc_tensor())?;
    wire = ast.force_variable(format!("{name}_input"), &wire);

    let mut inputs = tvec![wire, weigths];
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
        Some((output_dt, MatMulQParams::all_dynamic(1))),
    ));

    builder.wire(op, &inputs)
}
