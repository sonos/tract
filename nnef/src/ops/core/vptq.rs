use crate::internal::*;
use crate::ser::*;
use tract_core::ops::cast::cast;
use tract_core::ops::vptq::VPTQGemm;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_vptq_gemm);
    registry.register_primitive(
        "tract_core_vptq_gemm",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Scalar.tensor().named("indices"),
            TypeName::Scalar.tensor().named("centroids"),
            TypeName::Scalar.tensor().named("outlier_indices"),
            TypeName::Scalar.tensor().named("outlier_centroids"),
            TypeName::Scalar.tensor().named("perm"),
            TypeName::Scalar.tensor().named("weight_scale"),
            TypeName::Scalar.tensor().named("weight_bias"),
            TypeName::Scalar.tensor().named("bias"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_vptq_gemm,
    );
}

fn ser_vptq_gemm(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &VPTQGemm,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let indices = ast.mapping[&node.inputs[1]].clone();
    let centroids = ast.mapping[&node.inputs[2]].clone();
    let outlier_indices = ast.mapping[&node.inputs[3]].clone();
    let outlier_centroids = ast.mapping[&node.inputs[4]].clone();
    let perm = ast.mapping[&node.inputs[5]].clone();
    let weight_scale = ast.mapping[&node.inputs[6]].clone();
    let weight_bias = ast.mapping[&node.inputs[7]].clone();
    let bias = ast.mapping[&node.inputs[8]].clone();
    Ok(Some(invocation(
        "tract_core_vptq_gemm",
        &[
            input,
            indices,
            centroids,
            outlier_indices,
            outlier_centroids,
            perm,
            weight_scale,
            weight_bias,
            bias,
        ],
        &[],
    )))
}

fn de_vptq_gemm(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let indices = invocation.named_arg_as(builder, "indices")?;
    let centroids = invocation.named_arg_as(builder, "centroids")?;
    let outlier_indices = invocation.named_arg_as(builder, "outlier_indices")?;
    let outlier_centroids = invocation.named_arg_as(builder, "outlier_centroids")?;
    let perm = invocation.named_arg_as(builder, "perm")?;
    let weight_scale = invocation.named_arg_as(builder, "weight_scale")?;
    let weight_bias = invocation.named_arg_as(builder, "weight_bias")?;
    let bias = invocation.named_arg_as(builder, "bias")?;

    builder.wire(
        VPTQGemm {},
        &[
            input,
            indices,
            centroids,
            outlier_indices,
            outlier_centroids,
            perm,
            weight_scale,
            weight_bias,
            bias,
        ],
    )
}
