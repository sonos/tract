use crate::ast::*;
use std::collections::HashMap;
use tract_core::internal::*;
use tract_core::itertools::Itertools;

use tract_core::ops;

use crate::model::{AugmentedInvocation, ModelBuilder};

pub type Primitives = HashMap<
    String,
    Arc<dyn Fn(&mut ModelBuilder, &AugmentedInvocation) -> TractResult<TVec<OutletId>>>,
>;

pub fn primitives() -> Primitives {
    let mut primitives: Primitives = Default::default();
    primitives.insert("external".to_string(), Arc::new(external));
    primitives.insert("variable".to_string(), Arc::new(variable));
    primitives.insert("conv".to_string(), Arc::new(conv));

    macro_rules! mew {
        ($nnef: ident, $tract: expr) => {
            primitives.insert(
                stringify!($nnef).to_string(),
                Arc::new(|b, i| multiary_elementwise(b, i, Box::new($tract))),
            );
        };
    };

    mew!(add, ops::math::add::bin_typed());
    mew!(sub, ops::math::sub::bin_typed());
    mew!(mul, ops::math::mul::bin_typed());
    mew!(div, ops::math::div::bin_typed());
    mew!(pow, ops::math::pow::bin_typed());

    mew!(lt, ops::logic::lesser::bin_typed());
    mew!(gt, ops::logic::greater::bin_typed());
    mew!(le, ops::logic::lesser_equal::bin_typed());
    mew!(ge, ops::logic::greater_equal::bin_typed());
    mew!(eq, ops::logic::equals::bin_typed());
    mew!(ne, ops::logic::not_equals::bin_typed());

    mew!(select, ops::logic::Iff);

    primitives.insert("sum_reduce".to_string(), Arc::new(reduce));
    primitives.insert("max_reduce".to_string(), Arc::new(reduce));
    primitives.insert("min_reduce".to_string(), Arc::new(reduce));

    primitives.insert("matmul".to_string(), Arc::new(matmul));

    primitives.insert("transpose".to_string(), Arc::new(transpose));
    primitives.insert("unsqueeze".to_string(), Arc::new(unsqueeze));
    primitives.insert("squeeze".to_string(), Arc::new(squeeze));

    primitives
}

// fragment external<? = scalar>( shape: integer[] ) -> ( output: tensor<?> );
fn external(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let type_name = invocation.invocation.generic_type_name.unwrap_or(TypeName::Scalar);
    let dt = if type_name == TypeName::Scalar { f32::datum_type() } else { todo!() };
    let shape = invocation.named_arg("shape")?.to_shape_fact(builder)?;
    Ok(tvec!(builder.model.add_source("", TypedFact::dt_shape(dt, shape)?)?))
}

// fragment variable<? = scalar>( shape: integer[], label: string ) -> ( output: tensor<?> );
fn variable(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let type_name = invocation.invocation.generic_type_name.unwrap_or(TypeName::Scalar);
    let dt = if type_name == TypeName::Scalar { f32::datum_type() } else { todo!() };
    let shape = invocation.named_arg("shape")?.to_shape_fact(builder)?;
    let shape = shape.as_finite().unwrap();
    Ok(tvec!(builder.model.add_const("", Tensor::zero_dt(dt, &shape)?.into_arc_tensor())?))
}

// fragment transpose<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
fn transpose(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let axes = invocation.named_arg("axes")?.to_tensor(builder)?;
    let axes = axes.cast_to::<i64>()?;
    let axes = axes.as_slice::<i64>()?.iter().map(|a| *a as usize).collect::<TVec<_>>();
    let wire = tvec!(invocation.named_arg("input")?.to_wire(builder)?);
    ops::change_axes::perm_to_ops(&axes)
        .into_iter()
        .try_fold(wire, |wire, mov| Ok(builder.wire(mov, &wire)?))
}

// fragment squeeze<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
fn squeeze(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let axes = invocation.named_arg("axes")?.to_tensor(builder)?;
    let axes = axes.cast_to::<i64>()?;
    let wire = tvec!(invocation.named_arg("input")?.to_wire(builder)?);
    axes.as_slice::<i64>()?.iter().sorted().rev().try_fold(wire, |wire, &axis| {
        Ok(builder.wire(ops::change_axes::AxisOp::Rm(axis as usize), &wire)?)
    })
}

// fragment unsqueeze<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
fn unsqueeze(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let axes = invocation.named_arg("axes")?.to_tensor(builder)?;
    let axes = axes.cast_to::<i64>()?;
    let wire = tvec!(invocation.named_arg("input")?.to_wire(builder)?);
    axes.as_slice::<i64>()?.iter().sorted().try_fold(wire, |wire, &axis| {
        Ok(builder.wire(ops::change_axes::AxisOp::Add(axis as usize), &wire)?)
    })
}

/*
fragment conv( input: tensor<scalar>, filter: tensor<scalar>,
bias: tensor<scalar> = 0.0, border: string = 'constant',
padding: (integer,integer)[] = [], stride: integer[] = [],
dilation: integer[] = [], groups: integer = 1 )
-> ( output: tensor<scalar> );
*/

fn conv(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    use ops::cnn::{ConvUnary, KernelFormat, PaddingSpec, PoolSpec};
    use ops::nn::DataFormat;
    let input = invocation.named_arg("input")?.to_wire(builder)?;
    let kernel = invocation.named_arg("filter")?.to_tensor(builder)?;
    let input_fact = builder.model.outlet_fact(input)?;
    if input_fact.rank() != kernel.rank() {
        bail!(
            "Convolution input expected as NCHW, filter as OIHW. Got {:?} and {:?}.",
            input_fact,
            kernel
        );
    }
    if input_fact.shape.dim(1) != kernel.shape()[1].to_dim() {
        bail!("Convolution input and kernel channels (second axis in both) must match. Got {:?} and {:?}.", input_fact, kernel);
    }
    let bias = invocation.named_arg("bias")?.to_tensor(builder)?;
    let dilation = invocation.named_arg("dilation")?.to_tensor(builder)?;
    let dilation = dilation.cast_to::<i64>()?;
    let dilation: TVec<usize> = dilation.as_slice::<i64>()?.iter().map(|d| *d as usize).collect();
    let stride = invocation.named_arg("stride")?.to_tensor(builder)?;
    let stride = stride.cast_to::<i64>()?;
    let stride: TVec<usize> = stride.as_slice::<i64>()?.iter().map(|d| *d as usize).collect();
    let padding = invocation.named_arg("padding")?.to_tensor(builder)?;
    let border = invocation.named_arg("border")?.to_tensor(builder)?;
    assert_eq!(border, rctensor0("constant".to_string()));
    let group = invocation.named_arg("groups")?.to_tensor(builder)?.cast_to_scalar::<i64>()?;
    let padding = if padding.len() == 0 {
        PaddingSpec::Valid
    } else {
        let padding: tract_ndarray::ArrayView2<i64> =
            padding.to_array_view::<i64>()?.into_dimensionality()?;
        PaddingSpec::Explicit(
            padding.row(0).iter().map(|x| *x as usize).collect(),
            padding.row(1).iter().map(|x| *x as usize).collect(),
            false,
        )
    };
    let pool_spec = PoolSpec::new(
        DataFormat::NCHW,
        kernel.shape()[2..].into(),
        padding,
        if dilation.len() > 0 { Some(dilation) } else { None },
        if stride.len() > 0 { Some(stride) } else { None },
        Some(kernel.shape()[0]),
    );
    let op = ConvUnary::new(
        pool_spec,
        KernelFormat::OIHW,
        kernel.clone(),
        group as usize,
        Some(bias.clone()),
        None,
    );
    builder.wire(op, &[input])
}

/*
 *   fragment sum_reduce( input: tensor<scalar>, axes: integer[], normalize: logical = false ) -> ( output: tensor<scalar> );
 *   fragment max_reduce( input: tensor<scalar>, axes: integer[] ) -> ( output: tensor<scalar> );
 *   and also min, argmax, armmin, any, all
 */
fn reduce(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input = invocation.named_arg("input")?.to_wire(builder)?;
    let axes = invocation.named_arg("axes")?.to_tensor(builder)?;
    let axes = axes.cast_to::<i64>()?;
    let axes = axes.as_slice::<i64>()?.iter().map(|&i| i as usize).collect::<TVec<_>>();
    let reducer = match invocation.invocation.id.split("_").next().unwrap() {
        "sum" => ops::nn::Reducer::Sum,
        "min" => ops::nn::Reducer::Min,
        "max" => ops::nn::Reducer::Max,
        _ => bail!("unsupported reducer: {}", invocation.invocation.id),
    };
    let mut wire = builder.wire(ops::nn::Reduce::new(axes.clone(), reducer), &[input])?;
    let normalize = invocation.named_arg("normalize")?;
    let tensor = normalize.to_tensor(builder)?;
    if tensor.cast_to_scalar::<bool>()? {
        let input_shape = &builder.model.outlet_fact(input)?.shape;
        let cardinality = axes.iter().map(|ax| input_shape.dim(*ax)).maybe_product()?;
        let cardinality = tensor0(cardinality).broadcast_into_rank(input_shape.rank())?;
        wire = builder.wire(ops::math::div::unary(cardinality.into_arc_tensor()), &[input])?;
    }
    Ok(wire)
}

/*
 * fragment matmul( A: tensor<scalar>, B: tensor<scalar>, transposeA: logical = false, transposeB: logical = false ) -> ( C: tensor<scalar> );
 */
fn matmul(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let a = invocation.named_arg("A")?.to_wire(builder)?;
    let b = invocation.named_arg("B")?.to_wire(builder)?;
    let a_trans =
        invocation.named_arg("transposeA")?.to_tensor(builder)?.cast_to_scalar::<bool>()?;
    let b_trans =
        invocation.named_arg("transposeB")?.to_tensor(builder)?.cast_to_scalar::<bool>()?;
    builder.wire(ops::matmul::MatMul { a_trans, b_trans, c_trans: false, q_params: None }, &[a, b])
}

fn multiary_elementwise(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
    op: Box<dyn TypedOp>,
) -> TractResult<TVec<OutletId>> {
    let inputs = invocation
        .invocation
        .arguments
        .iter()
        .map(|arg| Ok(arg.rvalue.to_wire(builder)?))
        .collect::<TractResult<TVec<_>>>()?;
    let inputs = multicast(builder, &inputs)?;
    builder.wire(op, &inputs)
}

fn multicast(builder: &mut ModelBuilder, inputs: &[OutletId]) -> TractResult<TVec<OutletId>> {
    let ranks = inputs
        .iter()
        .map(|&i| Ok(builder.model.outlet_fact(i)?.rank()))
        .collect::<TractResult<Vec<usize>>>()?;
    let max_rank = ranks.iter().copied().max().unwrap();
    (inputs.iter())
        .zip(ranks.iter())
        .map(|(&i, &r)| {
            (r..max_rank).try_fold(i, |w, n| Ok(builder.wire(AxisOp::Add(n), &[w])?[0]))
        })
        .collect()
}
