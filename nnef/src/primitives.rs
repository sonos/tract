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

    primitives.insert("reshape".to_string(), Arc::new(reshape));
    primitives.insert("transpose".to_string(), Arc::new(transpose));
    primitives.insert("concat".to_string(), Arc::new(concat));
    primitives.insert("unsqueeze".to_string(), Arc::new(unsqueeze));
    primitives.insert("squeeze".to_string(), Arc::new(squeeze));

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

    mew!(exp, ops::math::exp());
    mew!(log, ops::math::ln());
    mew!(sin, ops::math::sin());
    mew!(cos, ops::math::cos());
    mew!(abs, ops::math::abs());
    mew!(sign, ops::math::sign());
    mew!(rcp, ops::math::recip());
    mew!(neg, ops::math::neg());
    mew!(copy, ops::identity::Identity);

    mew!(lt, ops::logic::lesser::bin_typed());
    mew!(gt, ops::logic::greater::bin_typed());
    mew!(le, ops::logic::lesser_equal::bin_typed());
    mew!(ge, ops::logic::greater_equal::bin_typed());
    mew!(eq, ops::logic::equals::bin_typed());
    mew!(ne, ops::logic::not_equals::bin_typed());

    mew!(and, ops::logic::and::bin_typed());
    mew!(or, ops::logic::or::bin_typed());
    mew!(not, ops::logic::not());

    mew!(floor, ops::math::floor());
    mew!(ceil, ops::math::ceil());
    mew!(round, ops::math::round());

    mew!(select, ops::logic::Iff);

    mew!(sqr, ops::math::square());
    mew!(sqrt, ops::math::sqrt());
    mew!(rsqrt, ops::math::rsqrt());

    mew!(min, ops::math::min::bin_typed());
    mew!(max, ops::math::max::bin_typed());

    primitives.insert("matmul".to_string(), Arc::new(matmul));

    primitives.insert("conv".to_string(), Arc::new(conv));

    primitives.insert("sum_reduce".to_string(), Arc::new(reduce));
    primitives.insert("max_reduce".to_string(), Arc::new(reduce));
    primitives.insert("min_reduce".to_string(), Arc::new(reduce));

    primitives.insert("max_pool_with_index".to_string(), Arc::new(max_pool_with_index));
    primitives.insert("box".to_string(), Arc::new(sum_pool));

    mew!(tanh, ops::math::tanh());
    mew!(sigmoid, ops::nn::sigmoid());

    primitives
}

// fragment external<? = scalar>( shape: integer[] ) -> ( output: tensor<?> );
fn external(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let type_name = invocation.invocation.generic_type_name.unwrap_or(TypeName::Scalar);
    let dt = if type_name == TypeName::Scalar { f32::datum_type() } else { todo!() };
    let shape: TVec<usize> = invocation.named_arg_as(builder, "shape")?;
    Ok(tvec!(builder.model.add_source("", TypedFact::dt_shape(dt, &*shape)?)?))
}

// fragment variable<? = scalar>( shape: integer[], label: string ) -> ( output: tensor<?> );
fn variable(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let type_name = invocation.invocation.generic_type_name.unwrap_or(TypeName::Scalar);
    let dt = if type_name == TypeName::Scalar { f32::datum_type() } else { todo!() };
    let shape: TVec<usize> = invocation.named_arg_as(builder, "shape")?;
    let label: String = invocation.named_arg_as(builder, "label")?;
    let tensor = builder
        .proto_model
        .tensors
        .get(&label)
        .ok_or_else(|| format!("No data for tensor {:?}", label))?;
    if tensor.datum_type() != dt {
        bail!("Wrong datum type for tensor: {:?}, tensor file says {:?}, graph files says {:?}", label, tensor.datum_type(), dt);
    }
    if tensor.shape() != &*shape {
        bail!("Wrong shape for tensor: {:?}, tensor file says {:?}, graph files says {:?}", label, tensor.shape(), shape);
    }
    builder.wire(tract_core::ops::konst::Const::new(tensor.clone()), &[])
}

// fragment reshape<?>( input: tensor<?>, shape: integer[], axis_start: integer = 0, axis_count: integer = -1 )
//      -> ( output: tensor<?> );
fn reshape(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input = invocation.named_arg_as(builder, "input")?;
    let input_shape = builder.model.outlet_fact(input)?.shape.to_tvec();
    let start: usize = invocation.named_arg_as(builder, "axis_start")?;
    let count: i64 = invocation.named_arg_as(builder, "axis_count")?;
    let count = if count == -1 { input_shape.len() - start } else { count as usize };
    let shape: TVec<TDim> = invocation.named_arg_as(builder, "shape")?;

    let mut replacement = shape.clone();
    for i in 0..replacement.len() {
        if replacement[i] == 0.to_dim() {
            replacement[i] = input_shape[i + start].clone();
        }
    }
    if let Some(pos) = replacement.iter().position(|d| *d == (-1).to_dim()) {
        let product: TDim = replacement.iter().filter(|d| **d != (-1).to_dim()).maybe_product()?;
        let product_input: TDim = input_shape[start..][..count].iter().maybe_product()?;
        replacement[pos] = product_input.maybe_div(&product)?.0;
    }

    let op = AxisOp::Reshape(start, input_shape[start..][..count].into(), replacement);
    builder.wire(op, &[input])
}

// fragment transpose<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
fn transpose(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    ops::change_axes::perm_to_ops(&axes)
        .into_iter()
        .try_fold(wire, |wire, mov| Ok(builder.wire(mov, &wire)?))
}

// fragment concat<?>( values: tensor<?>[], axis: integer ) -> ( value: tensor<?> );
fn concat(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let values: TVec<OutletId> = invocation.named_arg_as(builder, "values")?;
    builder.wire(ops::array::TypedConcat::concat_vars(axis, values.len()), &values)
}

// fragment squeeze<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
fn squeeze(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    axes.iter().sorted().rev().try_fold(wire, |wire, &axis| {
        Ok(builder.wire(ops::change_axes::AxisOp::Rm(axis as usize), &wire)?)
    })
}

// fragment unsqueeze<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
fn unsqueeze(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    axes.iter().sorted().try_fold(wire, |wire, &axis| {
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
    use ops::cnn::{ConvUnary, KernelFormat};
    use ops::cnn::{PaddingSpec, PoolSpec};
    use ops::nn::DataFormat;
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
    if input_fact.shape.dim(1) != kernel.shape()[1].to_dim() {
        bail!("Convolution input and kernel channels (second axis in both) must match. Got {:?} and {:?}.", input_fact, kernel);
    }
    let dilation: TVec<usize> = invocation.named_arg_as(builder, "dilation")?;
    if dilation.len() != 0 && dilation.len() != input_fact.rank() - 2 {
        bail!("Convolution dilation only apply to spatial dimensions, so it should be of rank {}. Got {:?}", input_fact.rank() -2, dilation)
    }
    let stride: TVec<usize> = invocation.named_arg_as(builder, "stride")?;
    if stride.len() != 0 && stride.len() != input_fact.rank() - 2 {
        bail!("Convolution stride only apply to spatial dimensions, so it should be of rank {}. Got {:?}", input_fact.rank() -2, stride)
    }
    let padding: TVec<TVec<usize>> = invocation.named_arg_as(builder, "padding")?;
    let padding = if padding.len() == 0 {
        PaddingSpec::SameUpper
    } else {
        let mut before = tvec!();
        let mut after = tvec!();
        for p in padding {
            before.push(p[0]);
            after.push(p[1]);
        }
        PaddingSpec::Explicit(before, after, false)
    };
    let pool_spec = PoolSpec::new(
        DataFormat::NCHW,
        kernel.shape()[2..].into(),
        padding,
        if dilation.len() > 0 { Some(dilation) } else { None },
        if stride.len() > 0 { Some(stride) } else { None },
        Some(kernel.shape()[0]),
    );
    let bias: Arc<Tensor> = invocation.named_arg_as(builder, "bias")?;
    let border: String = invocation.named_arg_as(builder, "border")?;
    assert_eq!(border, "constant");
    let group = invocation.named_arg_as(builder, "groups")?;
    let op = ConvUnary::new(
        pool_spec,
        KernelFormat::OIHW,
        kernel.clone(),
        group,
        Some(bias.clone()),
        None,
    );
    builder.wire(op, &[input])
}

fn pool_spec_for_pools(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
    shape: &[usize],
) -> TractResult<ops::cnn::PoolSpec> {
    use ops::cnn::{PaddingSpec, PoolSpec};
    use ops::nn::DataFormat;
    let dilation: TVec<usize> = invocation.named_arg_as(builder, "dilation")?;
    if dilation.len() > 0 && (dilation.len() != shape.len() || dilation[0] != 1 || dilation[1] != 1)
    {
        bail!("dilation should be like [1, 1, ... ]. Got dilation {:?}.", dilation);
    }
    let stride: TVec<usize> = invocation.named_arg_as(builder, "stride")?;
    if stride.len() > 0 && (stride.len() != shape.len() || stride[0] != 1 || stride[1] != 1) {
        bail!("stride should be like [1, 1, ... ]. Got stride {:?}.", stride);
    }
    let padding: TVec<TVec<usize>> = invocation.named_arg_as(builder, "padding")?;
    let padding = if padding.len() == 0 {
        PaddingSpec::SameUpper
    } else {
        let mut before = tvec!();
        let mut after = tvec!();
        for p in padding {
            before.push(p[0]);
            after.push(p[1]);
        }
        PaddingSpec::Explicit(before, after, false)
    };
    Ok(PoolSpec::new(
        DataFormat::NCHW,
        shape[2..].into(),
        padding,
        if dilation.len() > 2 { Some(dilation[2..].into()) } else { None },
        if stride.len() > 2 { Some(stride[2..].into()) } else { None },
        None,
    ))
}

/*
 * fragment max_pool_with_index( input: tensor<scalar>, size: integer[], border: string = 'constant',
 *  padding: (integer,integer)[] = [], stride: integer[] = [], dilation: integer[] = [] )
 *   -> ( output: tensor<scalar>, index: tensor<integer> )
 */

fn max_pool_with_index(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input = invocation.named_arg_as(builder, "input")?;
    let size: TVec<usize> = invocation.named_arg_as(builder, "size")?;
    let input_fact = builder.model.outlet_fact(input)?;
    if input_fact.rank() != size.len() {
        bail!(
            "Max pool input expected as NCHW, and \"size\" paramater must be [ 1, 1, x, y ]. Got {:?}, and {:?}",
            input_fact,
            size
            );
    }
    let border: String = invocation.named_arg_as(builder, "border")?;
    assert_eq!(border, "ignore");
    let pool_spec = pool_spec_for_pools(builder, invocation, &size)?;
    let op = ops::cnn::MaxPool { pool_spec, with_index_outputs: Some(i64::datum_type()) };
    builder.wire(op, &[input])
}

/*
 * fragment box( input: tensor<scalar>, size: integer[], border: string = 'constant', padding: (integer,integer)[] = [],
 *   stride: integer[] = [], dilation: integer[] = [], normalize: logical = false )
 * -> ( output: tensor<scalar> );
 */

fn sum_pool(
    builder: &mut ModelBuilder,
    invocation: &AugmentedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input = invocation.named_arg_as(builder, "input")?;
    let size: TVec<usize> = invocation.named_arg_as(builder, "size")?;
    let input_fact = builder.model.outlet_fact(input)?;
    if input_fact.rank() != size.len() {
        bail!(
            "Max pool input expected as NCHW, and \"size\" paramater must be [ 1, 1, x, y ]. Got {:?}, and {:?}",
            input_fact,
            size
            );
    }
    let border: String = invocation.named_arg_as(builder, "border")?;
    assert_eq!(border, "ignore");
    let pool_spec = pool_spec_for_pools(builder, invocation, &size)?;
    let op = ops::cnn::SumPool {
        pool_spec,
        count_include_pad: false,
        normalize: invocation.named_arg_as(builder, "normalize")?,
    };
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
    let input = invocation.named_arg_as(builder, "input")?;
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let reducer_name = invocation.invocation.id.split("_").next().unwrap();
    let reducer = match reducer_name {
        "sum" => ops::nn::Reducer::Sum,
        "min" => ops::nn::Reducer::Min,
        "max" => ops::nn::Reducer::Max,
        _ => bail!("unsupported reducer: {}", invocation.invocation.id),
    };
    let mut wire = builder.wire(ops::nn::Reduce::new(axes.clone(), reducer), &[input])?;
    if reducer_name == "sum" && invocation.named_arg_as(builder, "normalize")? {
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
    let a = invocation.named_arg_as(builder, "A")?;
    let b = invocation.named_arg_as(builder, "B")?;
    let a_trans = invocation.named_arg_as(builder, "transposeA")?;
    let b_trans = invocation.named_arg_as(builder, "transposeB")?;
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
        .map(|arg| arg.rvalue.resolve(builder)?.to(builder))
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
