use crate::internal::*;
use crate::ser::*;
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
            TypeName::Integer.named("vector_len"),
            TypeName::Integer.tensor().named("in_features"),
            TypeName::Integer.tensor().named("out_features"),
            TypeName::Integer.tensor().named("group_size"),
            TypeName::Integer.tensor().named("outlier_size"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_vptq_gemm,
    );
}

fn ser_vptq_gemm(
    ast: &mut IntoAst,
    node: &TypedNode,
    _op: &VPTQGemm,
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

    let vector_len = ast.mapping[&node.inputs[9]].clone();
    let in_features = ast.mapping[&node.inputs[10]].clone();
    let out_features = ast.mapping[&node.inputs[11]].clone();
    let group_size = ast.mapping[&node.inputs[12]].clone();
    let outlier_size = ast.mapping[&node.inputs[13]].clone();
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
        &[
            ("vector_len", numeric(vector_len)),
            ("in_features", numeric(in_features)),
            ("out_features", numeric(out_features)),
            ("group_size", numeric(group_size)),
            ("outlier_size", numeric(outlier_size)),
        ],
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

    let vector_len = invocation.named_arg_as(builder, "vector_len")?;
    let in_features = invocation.named_arg_as(builder, "in_features")?;
    let out_features = invocation.named_arg_as(builder, "out_features")?;
    let is_indice_packed = invocation.named_arg_as(builder, "is_indice_packed")?;

    let group_size = invocation.named_arg_as(builder, "group_size")?;
    let outlier_size = invocation.named_arg_as(builder, "outlier_size")?;

    builder.wire(
        VPTQGemm { vector_len, in_features, out_features, is_indice_packed, group_size, outlier_size},
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
