use crate::ast::*;
use tract_core::internal::*;
use tract_core::ops::cnn::deconv::adjustments;
use tract_core::ops::cnn::PaddingSpec;
use tract_core::ops::cnn::PoolSpec;
use tract_core::ops::matmul::MatMulQParams;
use tract_core::ops::nn::DataFormat;
use tract_itertools::izip;
use tract_itertools::Itertools;

use tract_core::ops;

use crate::deser::{ModelBuilder, ResolvedInvocation};

// fragment external<? = scalar>( shape: integer[] ) -> ( output: tensor<?> );
pub fn external(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let type_name = invocation.invocation.generic_type_name.unwrap_or(TypeName::Scalar);
    let dt = if let Some(Some(dt)) = invocation.dt_from_quant_file.get(0) {
        *dt
    } else if type_name == TypeName::Scalar {
        f32::datum_type()
    } else if type_name == TypeName::Logical {
        bool::datum_type()
    } else if type_name == TypeName::Integer {
        i64::datum_type()
    } else {
        todo!()
    };
    let shape: TVec<TDim> = invocation.named_arg_as(builder, "shape")?;
    Ok(tvec!(builder.model.add_source("", dt.fact(&shape))?))
}

// fragment variable<? = scalar>( shape: integer[], label: string ) -> ( output: tensor<?> );
pub fn variable(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let shape: TVec<usize> = invocation.named_arg_as(builder, "shape")?;
    let label: String = invocation.named_arg_as(builder, "label")?;
    let mut tensor = builder
        .proto_model
        .tensors
        .iter()
        .find(|pair| pair.0 == label)
        .ok_or_else(|| format_err!("No data for tensor {:?}", label))?
        .1
        .clone();
    if let Some(Some(dt)) = invocation.dt_from_quant_file.get(0) {
        if dt.size_of() != tensor.datum_type().size_of() {
            bail!(
                "Mismatched tensor type for tensor {}: expected {:?}, got {:?}",
                label,
                *dt,
                tensor.datum_type()
            );
        }
        if *dt != tensor.datum_type() {
            trace!(
                "Casting tensor {} from {:?} to {:?} when deserializing",
                label,
                tensor.datum_type(),
                *dt
            );
            //FIXME: avoid cast by late-loading tensors ?
            tensor = tensor.cast_to_dt(*dt)?.into_owned().into_arc_tensor()
        }
    }
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
        let product: TDim = replacement.iter().filter(|d| **d != (-1).to_dim()).product();
        let product_input: TDim = input_shape[start..][..count].iter().product();
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
    let mut values: TVec<OutletId> = invocation.named_arg_as(builder, "values")?;
    if let Some(Some(dt)) = invocation.dt_from_quant_file.get(0) {
        for value in &mut values {
            if builder.model.node(value.node).outputs[value.slot].fact.datum_type != *dt {
                *value = builder.wire(ops::cast::cast(*dt), &[*value])?[0];
            }
        }
    }

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
    let multipliers: TVec<TDim> = invocation.named_arg_as(builder, "repeats")?;
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

/*  fragment deconv(
input: tensor<scalar>,
filter: tensor<scalar>,
bias: tensor<scalar> = 0.0,
border: string = 'constant',
padding: (integer,integer)[] = [],
stride: integer[] = [],
dilation: integer[] = [],
output_shape: integer[] = [],
groups: integer = 1 )
-> ( output: tensor<scalar> );
*/

pub fn conv(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    conv_or_deconv(builder, invocation, false)
}

pub fn deconv(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    conv_or_deconv(builder, invocation, true)
}

pub fn read_conv_parameters(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
    kernel_shape: &[usize],
    input_fact: &TypedFact,
) -> TractResult<(usize, PoolSpec)> {
    let mut group = invocation.named_arg_as(builder, "groups")?;
    if group == 0 {
        group = kernel_shape[0]
    }
    if input_fact.shape[1] != kernel_shape[1].to_dim() * group {
        bail!("Convolution input and kernel channels (second axis in both) must match. Got {:?} and {:?}.", input_fact, kernel_shape);
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
        kernel_shape[2..].into(),
        padding,
        if dilation.len() > 0 { Some(dilation) } else { None },
        if stride.len() > 0 { Some(stride) } else { None },
        Some(kernel_shape[0]),
    );

    let border: String = invocation.named_arg_as(builder, "border")?;
    assert_eq!(border, "constant");

    Ok((group, pool_spec))
}

pub fn conv_or_deconv(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
    deconv: bool,
) -> TractResult<TVec<OutletId>> {
    use ops::cnn::deconv::DeconvUnary;
    use ops::cnn::{ConvUnary, KernelFormat};

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

    let output_dt =
        invocation.dt_from_quant_file.get(0).cloned().flatten().unwrap_or(DatumType::F32);
    let quantized = input_fact.datum_type.is_quantized()
        || kernel.datum_type().is_quantized()
        || output_dt.is_quantized();

    let qparams = if quantized { Some((output_dt, MatMulQParams::all_from_qtype())) } else { None };
    let bias: Arc<Tensor> = invocation.named_arg_as(builder, "bias")?;

    let bias: Option<Arc<Tensor>> =
        if bias.is_uniform() && bias.cast_to_scalar::<f32>()? == 0.0 { None } else { Some(bias) };

    let op: Box<dyn TypedOp> = if deconv {
        let output_shape = invocation.named_arg_as::<TVec<usize>>(builder, "output_shape")?;
        let output_shape = Some(output_shape).filter(|os| os.len() == pool_spec.rank());
        let adjustments = if let Some(output_shape) = output_shape {
            let input_shape = &input_fact
                .shape
                .as_concrete()
                .context("symbolic dimension not supported in deconv")?[2..];
            adjustments(&pool_spec, &input_shape, &output_shape)?
        } else {
            tvec!(0; pool_spec.rank())
        };
        // nnef form is O I/g H W
        // tract expects O/g I H W
        let kernel =
            kernel.into_tensor().split_axis(0, group)?.move_axis(0, 1)?.collapse_axis_with_next(1);
        Box::new(DeconvUnary::new(
            pool_spec,
            KernelFormat::OIHW,
            kernel.into_arc_tensor(),
            bias,
            adjustments,
            group,
        ))
    } else {
        Box::new(ConvUnary::new(
            pool_spec,
            KernelFormat::OIHW,
            kernel.clone(),
            group,
            bias,
            qparams,
        ))
    };
    builder.wire(op, &[input])
}

fn pool_spec_for_pools(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
    shape: &[usize],
) -> TractResult<ops::cnn::PoolSpec> {
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
    assert!(&*border == "ignore" || &*border == "constant");
    //FIXME : constant is not actually supported, but it should be the same in most cases
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
    assert!(&*border == "ignore" || &*border == "constant");
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

    let fact = builder.model.outlet_fact(wire[0])?.clone();
    let input_shape = &builder.model.outlet_fact(input)?.shape;
    let cardinality: TDim = axes.iter().map(|ax| &input_shape[*ax]).product();
    let cardinality = builder.wire(ops::konst::Const::new(tensor0(cardinality).broadcast_into_rank(fact.rank())?.into_arc_tensor()), &[])?;
    let cardinality = builder.wire(ops::cast::Cast::new(fact.datum_type), &cardinality)?;
    dbg!(&cardinality);
    return builder.wire(ops::math::div::bin_typed(), &[wire[0], cardinality[0]])
}

/*
/* override linear to manage quantization in intermediaries
 * fragment linear(
 *    input: tensor<scalar>,
 *    filter: tensor<scalar>,
 *    bias: tensor<scalar> = 0.0 ) -> ( output: tensor<scalar> )
 {
 output = matmul(input, filter, transposeB = true) + bias;
 }
 */
pub fn linear(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
    ) -> TractResult<TVec<OutletId>> {
    let input: OutletId = invocation.named_arg_as(builder, "input")?;
    let filter: OutletId = invocation.named_arg_as(builder, "filter")?;
    let bias: OutletId = invocation.named_arg_as(builder, "bias")?;
    if let Some(Some(dt)) = &invocation.dt_from_quant_file.get(0) {
        if let Some(_) = dt.qparams() {
            return builder.wire(
                ops::matmul::QMatMul {
                    a_trans: false,
                    b_trans: true,
                    c_trans: false,
                    output_type_foo: *dt,
                    params: MatMulQParams::all_from_qtype(),
                },
                &[input, filter, bias],
                );
        }
    }
    let mul = builder.wire(
        ops::matmul::MatMul { a_trans: false, b_trans: true, c_trans: false },
        &[input, filter],
        )?[0];
    builder.wire(ops::math::add::bin_typed(), &[bias, mul])
}
*/

/*
 * fragment matmul( A: tensor<scalar>, B: tensor<scalar>, transposeA: logical = false, transposeB: logical = false ) -> ( C: tensor<scalar> );
 */
pub fn matmul(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let a: OutletId = invocation.named_arg_as(builder, "A")?;
    let b: OutletId = invocation.named_arg_as(builder, "B")?;
    let a_trans = invocation.named_arg_as(builder, "transposeA")?;
    let b_trans = invocation.named_arg_as(builder, "transposeB")?;
    let a_dt = builder.model.outlet_fact(a)?.datum_type;
    let b_dt = builder.model.outlet_fact(b)?.datum_type;
    if a_dt.is_quantized() || b_dt.is_quantized() {
        let accum_dt = DatumType::QI32(QParams::ZpScale {
            scale: a_dt.zp_scale().1 * b_dt.zp_scale().1,
            zero_point: 0,
        });
        let dt = invocation.dt_from_quant_file.get(0).cloned().flatten().unwrap_or(accum_dt);
        let bias = builder.model.add_const(
            format!("{}.bias", invocation.invocation.id),
            Tensor::zero_dt(accum_dt, &[1])?,
        )?;
        builder.model.node(a.node);

        return builder.wire(
            ops::matmul::QMatMul {
                a_trans,
                b_trans,
                c_trans: false,
                output_type: dt,
                params: MatMulQParams::all_from_qtype(),
            },
            &[a, b, bias],
        );
    } else {
        builder.wire(ops::matmul::MatMul { a_trans, b_trans, c_trans: false }, &[a, b])
    }
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

/*
 * fragment leaky_relu( x: tensor<scalar>, alpha: scalar )-> ( y: tensor<scalar> )
 */

pub fn leaky_relu(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let x = invocation.named_arg_as(builder, "x")?;
    let alpha = invocation.named_arg_as(builder, "alpha")?;
    builder.wire(ops::nn::leaky_relu(alpha), &[x])
}

/*
 * fragment stack<?>( values: tensor<?>[], axis: integer ) -> ( value: tensor<?> )
 *
 * Same as concat but on dedicated axis
 */

pub fn stack(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let mut values: TVec<OutletId> = invocation.named_arg_as(builder, "values")?;
    if let Some(Some(dt)) = invocation.dt_from_quant_file.get(0) {
        for value in &mut values {
            if builder.model.node(value.node).outputs[value.slot].fact.datum_type != *dt {
                *value = builder.wire(ops::cast::cast(*dt), &[*value])?[0];
            }
        }
    }

    for value in &mut values {
        // add unsqueeze
        *value = builder.wire(ops::change_axes::AxisOp::Add(axis as usize), &[*value])?[0];
    }

    builder.wire(ops::array::TypedConcat::concat_vars(axis, values.len()), &values)
}

/*
 * fragment unstack<?>( value: tensor<?>, axis: integer ) -> ( values: tensor<?>[] )
 *
 * Inverse of stack operator
 */
pub fn unstack(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let wire = tvec!(invocation.named_arg_as(builder, "value")?);
    let axis: usize = invocation.named_arg_as(builder, "axis")?;

    let input_fact = builder.model.outlet_fact(wire[0])?.clone();

    (0..input_fact.shape[axis].clone().to_i32()?)
        .into_iter()
        .map(|start_int| {
            let start = start_int.to_dim();
            let end = (start_int + 1).to_dim();
            let sliced_wire =
                builder.wire(tract_core::ops::array::Slice { axis, start, end }, &wire)?;
            let squeezed_wire =
                builder.wire(ops::change_axes::AxisOp::Rm(axis as usize), &sliced_wire)?;
            Ok(squeezed_wire[0])
        })
        .collect()
}

/*
* fragment softmax( x: tensor<scalar>, axes: integer[] = [1] ) -> ( y: tensor<scalar> )
 * {
 *    m = max_reduce(x, axes = axes);
 *    e = exp(x - m);
 *    y = e / sum_reduce(e, axes = axes);
 * }
 */

pub fn softmax(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let x = invocation.named_arg_as(builder, "x")?;
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;

    let input_fact = builder.model.outlet_fact(x)?.clone();
    let output_dt =
        invocation.dt_from_quant_file.get(0).cloned().flatten().unwrap_or(input_fact.datum_type);

    builder.wire(ops::nn::Softmax { axes, output_dt }, &[x])
}
