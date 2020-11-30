use crate::ast::*;
use tract_core::internal::*;
use tract_core::itertools::izip;
use tract_core::itertools::Itertools;

use tract_core::ops;

use crate::deser::{ModelBuilder, ResolvedInvocation};

// fragment external<? = scalar>( shape: integer[] ) -> ( output: tensor<?> );
pub fn external(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let type_name = invocation.invocation.generic_type_name.unwrap_or(TypeName::Scalar);
    let dt = if type_name == TypeName::Scalar {
        f32::datum_type()
    } else if type_name == TypeName::Logical {
        bool::datum_type()
    } else {
        todo!()
    };
    let shape: TVec<usize> = invocation.named_arg_as(builder, "shape")?;
    Ok(tvec!(builder.model.add_source("", TypedFact::dt_shape(dt, &shape))?))
}

// fragment variable<? = scalar>( shape: integer[], label: string ) -> ( output: tensor<?> );
pub fn variable(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let shape: TVec<usize> = invocation.named_arg_as(builder, "shape")?;
    let label: String = invocation.named_arg_as(builder, "label")?;
    let tensor = builder
        .proto_model
        .tensors
        .iter()
        .find(|pair| pair.0 == label)
        .ok_or_else(|| format_err!("No data for tensor {:?}", label))?
        .1
        .clone();
    if tensor.shape() != &*shape {
        bail!(
            "Wrong shape for tensor: {:?}, tensor file says {:?}, graph files says {:?}",
            label,
            tensor.shape(),
            shape
        );
    }
    builder.wire(tract_core::ops::konst::Const::new(tensor), &[])
}

// fragment reshape<?>( input: tensor<?>, shape: integer[], axis_start: integer = 0, axis_count: integer = -1 )
//      -> ( output: tensor<?> );
pub fn reshape(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
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
pub fn transpose(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    ops::change_axes::perm_to_ops(&axes)
        .into_iter()
        .try_fold(wire, |wire, mov| Ok(builder.wire(mov, &wire)?))
}

// fragment concat<?>( values: tensor<?>[], axis: integer ) -> ( value: tensor<?> );
pub fn concat(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let values: TVec<OutletId> = invocation.named_arg_as(builder, "values")?;
    builder.wire(ops::array::TypedConcat::concat_vars(axis, values.len()), &values)
}

// fragment slice<?>( input: tensor<?>, axes: integer[], begin: integer[], end: integer[] ) -> ( output: tensor<?> );
pub fn slice(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    let input_fact = builder.model.outlet_fact(wire[0])?.clone();
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let begins: TVec<i64> = invocation.named_arg_as(builder, "begin")?;
    let begins = begins.into_iter().enumerate().map(|(ix, b)| -> TDim {
        if b < 0 {
            input_fact.shape[ix].clone() + b
        } else {
            b.into()
        }
    });
    let ends: TVec<i64> = invocation.named_arg_as(builder, "end")?;
    let ends = ends.into_iter().enumerate().map(|(ix, b)| -> TDim {
        if b < 0 {
            input_fact.shape[ix].clone() + b
        } else {
            b.into()
        }
    });
    izip!(axes, begins, ends).try_fold(wire, |wire, (axis, start, end)| {
        builder.wire(tract_core::ops::array::Slice { axis, start, end }, &wire)
    })
}

// fragment squeeze<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
pub fn squeeze(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    axes.iter().sorted().rev().try_fold(wire, |wire, &axis| {
        Ok(builder.wire(ops::change_axes::AxisOp::Rm(axis as usize), &wire)?)
    })
}

// fragment unsqueeze<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
pub fn unsqueeze(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    axes.iter().sorted().try_fold(wire, |wire, &axis| {
        Ok(builder.wire(ops::change_axes::AxisOp::Add(axis as usize), &wire)?)
    })
}

// fragment tile<?>( input: tensor<?>, repeats: integer[] ) -> ( output: tensor<?> );
pub fn tile(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let multipliers: TVec<usize> = invocation.named_arg_as(builder, "repeat")?;
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    Ok(builder.wire(ops::array::Tile { multipliers }, &wire)?)
}

// fragment pad( input: tensor<scalar>, padding: (integer, integer)[], border: string = 'constant', value: scalar = 0.0 ) -> ( output: tensor<scalar> );
pub fn pad(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    use tract_core::ops::array::{Pad, PadMode};
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    let padding: TVec<TVec<usize>> = invocation.named_arg_as(builder, "padding")?;
    let padding: Vec<(usize, usize)> = padding.iter().map(|a| (a[0], a[1])).collect();
    let value: Tensor = tensor0(invocation.named_arg_as::<f32>(builder, "value")?);
    let border: String = invocation.named_arg_as(builder, "border")?;
    let mode = match &*border {
        "constant" => PadMode::Constant(value.into_arc_tensor()),
        "replicated" => PadMode::Edge,
        "reflect" => PadMode::Reflect,
        _ => bail!("unsupported padding mode {}", border),
    };
    builder.wire(Pad { pads: padding, mode }, &wire)
}

/*
fragment conv( input: tensor<scalar>, filter: tensor<scalar>,
bias: tensor<scalar> = 0.0, border: string = 'constant',
padding: (integer,integer)[] = [], stride: integer[] = [],
dilation: integer[] = [], groups: integer = 1 )
-> ( output: tensor<scalar> );
*/

pub fn conv(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
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
    let mut group = invocation.named_arg_as(builder, "groups")?;
    if group == 0 {
        group = kernel.shape()[0]
    }
    if input_fact.shape[1] != kernel.shape()[1].to_dim() * group {
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
    let bias: Option<Arc<Tensor>> =
        if bias.is_uniform() && bias.cast_to_scalar::<f32>()? == 0.0 { None } else { Some(bias) };

    let border: String = invocation.named_arg_as(builder, "border")?;
    assert_eq!(border, "constant");
    let op = ConvUnary::new(pool_spec, KernelFormat::OIHW, kernel.clone(), group, bias);
    builder.wire(op, &[input])
}

fn pool_spec_for_pools(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
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

pub fn max_pool_with_index(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
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

pub fn sum_pool(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
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
pub fn reduce(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input = invocation.named_arg_as(builder, "input")?;
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let reducer_name = invocation.invocation.id.split("_").next().unwrap();
    let reducer = match reducer_name {
        "sum" => ops::nn::Reducer::Sum,
        "min" => ops::nn::Reducer::Min,
        "max" => ops::nn::Reducer::Max,
        "argmin" => ops::nn::Reducer::ArgMin(false),
        "argmax" => ops::nn::Reducer::ArgMax(false),
        _ => bail!("unsupported reducer: {}", invocation.invocation.id),
    };
    let wire = builder.wire(ops::nn::Reduce::new(axes.clone(), reducer), &[input])?;
    if reducer_name != "sum" || !invocation.named_arg_as(builder, "normalize")? {
        return Ok(wire);
    }

    let fact = builder.model.outlet_fact(wire[0])?;
    let input_shape = &builder.model.outlet_fact(input)?.shape;
    let cardinality: TDim = axes.iter().map(|ax| &input_shape[*ax]).maybe_product()?;
    if let Ok(c) = cardinality.to_isize() {
        if fact.datum_type.is_float() {
            let cardinality = tensor0((c as f64).recip())
                .cast_to_dt(fact.datum_type)?
                .into_owned()
                .broadcast_into_rank(input_shape.rank())?;
            return builder.wire(ops::math::mul::unary(cardinality.into_arc_tensor()), &wire);
        }
    }
    bail!("Normalization only works with float items and known dimensions");
}

/*
 * fragment matmul( A: tensor<scalar>, B: tensor<scalar>, transposeA: logical = false, transposeB: logical = false ) -> ( C: tensor<scalar> );
 */
pub fn matmul(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let a = invocation.named_arg_as(builder, "A")?;
    let b = invocation.named_arg_as(builder, "B")?;
    let a_trans = invocation.named_arg_as(builder, "transposeA")?;
    let b_trans = invocation.named_arg_as(builder, "transposeB")?;
    builder.wire(ops::matmul::MatMul { a_trans, b_trans, c_trans: false }, &[a, b])
}

/*
* fragment select<?>(
condition: tensor<logical>,     # the condition for selecting the result
true_value: tensor<?>,          # the result when the condition is true
false_value: tensor<?> )        # the result when the condition is false
-> ( output: tensor<?> )
*/

pub fn select(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let cond = invocation.named_arg_as(builder, "condition")?;
    let true_value = invocation.named_arg_as(builder, "true_value")?;
    let false_value = invocation.named_arg_as(builder, "false_value")?;
    let inputs = crate::registry::multicast(builder, &[cond, true_value, false_value])?;
    builder.wire(ops::logic::Iff {}, &inputs)
}
