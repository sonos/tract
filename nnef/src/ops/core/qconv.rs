use super::qmatmul::qparams_to_rvalues;
use crate::deser::Value;
use crate::internal::*;
use crate::ops::core::qmatmul::values_to_qparams;
use crate::ops::nnef::deser::read_conv_parameters;
use crate::ops::nnef::ser::make_conv_named_args;
use crate::ser::*;
use tract_core::ops::cnn::ConvUnary;
use tract_core::ops::cnn::KernelFormat;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<tract_core::ops::cnn::ConvUnary>(), qconv_unary_dump);
    registry.register_primitive("tract_core_qconv", &qconv_parameters(), qconv_load);
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
    let mut named_args = make_conv_named_args(node, &op.pool_spec, op.group, false, None)?;

    let [a0, a_scale, b0, b_scale, c0, c_scale] =
        qparams_to_rvalues(&op.q_params.as_ref().unwrap().1, &node.inputs, &ast.mapping)?;
    macro_rules! push {
        ($a: ident) => {
            if let Some($a) = $a {
                named_args.push((stringify!($a), $a));
            }
        };
    }
    push!(a0);
    push!(a_scale);
    push!(b0);
    push!(b_scale);
    push!(c0);
    push!(c_scale);

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
    weights.set_shape(&*kernel_shape)?;
    let weigths =
        ast.konst_variable(format!("{}_weigths", node.name), &weights.into_arc_tensor())?;
    wire = ast.force_assign(format!("{}_input", node.name), &wire);

    let mut inputs = tvec![wire, weigths];
    if let Some(bias) = op.bias.as_ref() {
        let bias = ast.konst(format!("{}_bias", node.name), bias)?;
        inputs.push(bias)
    }

    Ok(Some(invocation("tract_core_qconv", &inputs, &named_args)))
}

fn qconv_load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input: OutletId = invocation.named_arg_as(builder, "input")?;
    let kernel: Arc<Tensor> = invocation.named_arg_as(builder, "filter")?;

    let input_fact = builder.model.outlet_fact(input)?.clone();
    if input_fact.rank() != kernel.rank() {
        bail!(
            "Convolution input expected as NCHW, filter as OIHW. Got {:?} and {:?}.",
            input_fact,
            kernel
        );
    }

    let (group, pool_spec) =
        read_conv_parameters(builder, invocation, kernel.shape(), &input_fact)?;

    let a0: Option<Value> = invocation.named_arg_as(builder, "a0").ok();
    let a_scale: Option<Value> = invocation.named_arg_as(builder, "a_scale").ok();
    let b0: Option<Value> = invocation.named_arg_as(builder, "b0").ok();
    let b_scale: Option<Value> = invocation.named_arg_as(builder, "b_scale").ok();
    let c0: Option<Value> = invocation.named_arg_as(builder, "c0").ok();
    let c_scale: Option<Value> = invocation.named_arg_as(builder, "c_scale").ok();
    let mut inputs = vec![input];
    let qparams = values_to_qparams(a0, a_scale, b0, b_scale, c0, c_scale, &mut inputs, builder)?;
    let bias: Arc<Tensor> = invocation.named_arg_as(builder, "bias")?;

    let bias: Option<Arc<Tensor>> =
        if bias.is_uniform() && bias.cast_to_scalar::<f32>()? == 0.0 { None } else { Some(bias) };
    let output_dt = input_fact.datum_type.with_qparams(QParams::ZpScale {
        zero_point: *qparams
            .c0
            .as_static()
            .ok_or(format_err!("The output quantization need to be static in convolution"))?
            .to_scalar()?,
        scale: *qparams
            .c_scale
            .as_static()
            .ok_or(format_err!("The output quantization need to be static in convolution"))?
            .to_scalar()?,
    });
    let op: Box<dyn TypedOp> = Box::new(ConvUnary::new(
        pool_spec,
        KernelFormat::OIHW,
        kernel.clone(),
        group,
        bias,
        Some((output_dt, qparams)),
    ));

    builder.wire(op, &inputs)
}
