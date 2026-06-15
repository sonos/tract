use crate::internal::*;
use tract_core::ops::nn::resize::{CoordTransformer, Interpolator, Nearest, Resize};

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_core_resize",
        &parameters(),
        &[("output", TypeName::Scalar.tensor())],
        load,
    );
    registry.register_dumper(dump);
    registry.register_primitive(
        "nearest_upsample",
        &upsample_parameters(),
        &[("output", TypeName::Scalar.tensor())],
        load_nearest_upsample,
    );
    registry.register_primitive(
        "multilinear_upsample",
        &multilinear_parameters(),
        &[("output", TypeName::Scalar.tensor())],
        load_multilinear_upsample,
    );
}

fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Scalar.tensor().named("scales").default(false),
        TypeName::Scalar.tensor().named("sizes").default(false),
        TypeName::String.named("coord_transformer").default("half_pixel"),
        TypeName::String.named("interpolator").default("nearest"),
        TypeName::String.named("nearest_mode").default("floor"),
    ]
}

fn dump(ast: &mut IntoAst, node: &TypedNode, op: &Resize) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let (arg_name, aux) = if let Some(scales_ix) = op.optional_scales_input {
        ("scales", ast.mapping[&node.inputs[scales_ix]].clone())
    } else if let Some(sizes_ix) = op.optional_sizes_input {
        ("sizes", ast.mapping[&node.inputs[sizes_ix]].clone())
    } else {
        bail!("Resize op has neither scales nor sizes input")
    };
    Ok(Some(invocation(
        "tract_core_resize",
        &[input],
        &[
            (arg_name, (*aux).clone()),
            ("coord_transformer", string(op.coord_transformer.as_str())),
            ("interpolator", string(op.interpolator.as_str())),
            ("nearest_mode", string(op.nearest.as_str())),
        ],
    )))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let scales = invocation.optional_named_arg_as::<OutletId>(builder, "scales")?;
    let sizes = invocation.optional_named_arg_as::<OutletId>(builder, "sizes")?;
    let coord_transformer: String = invocation.named_arg_as(builder, "coord_transformer")?;
    let interpolator: String = invocation.named_arg_as(builder, "interpolator")?;
    let nearest_mode: String = invocation.named_arg_as(builder, "nearest_mode")?;

    let (aux, optional_scales_input, optional_sizes_input) = match (scales, sizes) {
        (Some(scales), None) => (scales, Some(1), None),
        (None, Some(sizes)) => (sizes, None, Some(1)),
        (Some(_), Some(_)) => {
            bail!("tract_core_resize: provide either scales or sizes, not both")
        }
        (None, None) => bail!("tract_core_resize: needs either scales or sizes"),
    };

    let op = Resize {
        axes: None,
        coord_transformer: CoordTransformer::parse(&coord_transformer)?,
        interpolator: Interpolator::parse(&interpolator)?,
        nearest: Nearest::parse(&nearest_mode)?,
        optional_scales_input,
        optional_sizes_input,
    };
    builder.wire(op, &[input, aux])
}

fn upsample_parameters() -> Vec<Parameter> {
    vec![TypeName::Scalar.tensor().named("input"), TypeName::Integer.array().named("factor")]
}

fn multilinear_parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Integer.array().named("factor"),
        TypeName::String.named("method").default("symmetric"),
        TypeName::String.named("border").default("replicate"),
    ]
}

/// Builds the per-axis scales constant for the NNEF `*_upsample` fragments: the
/// `factor` array describes the trailing spatial axes, leading axes stay at 1.
fn upsample_scales(
    builder: &mut ModelBuilder,
    input: &OutletId,
    factor: &[i64],
) -> TractResult<OutletId> {
    let rank = builder.model.outlet_fact(*input)?.rank();
    ensure!(
        factor.len() <= rank,
        "upsample factor has {} entries but input rank is {rank}",
        factor.len()
    );
    let mut scales = vec![1.0f32; rank - factor.len()];
    scales.extend(factor.iter().map(|&f| f as f32));
    builder.add_const(tract_ndarray::Array1::from(scales))
}

fn load_nearest_upsample(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input: OutletId = invocation.named_arg_as(builder, "input")?;
    let factor: TVec<i64> = invocation.named_arg_as(builder, "factor")?;
    let scales = upsample_scales(builder, &input, &factor)?;
    let op = Resize {
        axes: None,
        coord_transformer: CoordTransformer::Asymmetric,
        interpolator: Interpolator::Nearest,
        nearest: Nearest::Floor,
        optional_scales_input: Some(1),
        optional_sizes_input: None,
    };
    builder.wire(op, &[input, scales])
}

fn load_multilinear_upsample(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input: OutletId = invocation.named_arg_as(builder, "input")?;
    let factor: TVec<i64> = invocation.named_arg_as(builder, "factor")?;
    let method: String = invocation.named_arg_as(builder, "method")?;
    let border: String = invocation.named_arg_as(builder, "border")?;
    let coord_transformer = match method.as_str() {
        "symmetric" => CoordTransformer::HalfPixel,
        "asymmetric" => CoordTransformer::Asymmetric,
        "aligned" => CoordTransformer::AlignCorners,
        s => bail!("multilinear_upsample method: {s}"),
    };
    if border != "replicate" {
        bail!("multilinear_upsample only supports border = 'replicate', got '{border}'");
    }
    let scales = upsample_scales(builder, &input, &factor)?;
    let op = Resize {
        axes: None,
        coord_transformer,
        interpolator: Interpolator::Linear,
        nearest: Nearest::Floor,
        optional_scales_input: Some(1),
        optional_sizes_input: None,
    };
    builder.wire(op, &[input, scales])
}
