use crate::ast::*;
use crate::deser::Value;
use tract_core::internal::*;
use tract_core::ops::array::PadMode;
use tract_core::ops::cnn::deconv::adjustments;
use tract_core::ops::cnn::PaddingSpec;
use tract_core::ops::cnn::PoolSpec;
use tract_core::ops::nn::DataFormat;
use tract_itertools::izip;
use tract_itertools::Itertools;

use tract_core::ops;

use crate::deser::{ModelBuilder, ResolvedInvocation};

// fragment external<? = scalar>( shape: integer[] ) -> ( output: tensor<?> );
pub fn external(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
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
    let shape: TVec<TDim> =
        builder.allowing_new_symbols(|builder| invocation.named_arg_as(builder, "shape"))?;
    Ok(Value::Wire(builder.model.add_source("", dt.fact(&shape))?))
}

// fragment variable<? = scalar>( shape: integer[], label: string ) -> ( output: tensor<?> );
pub fn variable(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let shape: TVec<usize> = invocation.named_arg_as(builder, "shape")?;
    let label = Identifier(invocation.named_arg_as(builder, "label")?);
    let tensors = &builder.proto_model.tensors;
    let mut tensor = Arc::clone(
        tensors
            .get(&label)
            .or_else(|| tensors.get(&Identifier(label.0.trim_start_matches('/').to_owned())))
            .ok_or_else(|| format_err!("No data for tensor {:?}", label))?,
    );
    if let Some(Some(dt)) = invocation.dt_from_quant_file.get(0) {
        if dt.size_of() != tensor.datum_type().size_of() {
            bail!(
                "Mismatched tensor type for tensor {}: expected {:?}, got {:?}",
                label.0,
                *dt,
                tensor.datum_type()
            );
        }
        if *dt != tensor.datum_type() {
            trace!(
                "Casting tensor {} from {:?} to {:?} when deserializing",
                label.0,
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
pub fn reshape(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let input_shape = builder.model.outlet_fact(input)?.shape.to_tvec();
    let start: usize = invocation.named_arg_as(builder, "axis_start")?;
    let count: i64 = invocation.named_arg_as(builder, "axis_count")?;
    let count = if count == -1 { input_shape.len() - start } else { count as usize };
    let shape: TVec<TDim> =
        builder.allowing_new_symbols(|builder| invocation.named_arg_as(builder, "shape"))?;

    let mut replacement = shape;
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
) -> TractResult<Value> {
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    ops::change_axes::perm_to_ops(&axes)
        .into_iter()
        .try_fold(wire, |wire, mov| builder.wire_as_outlets(mov, &wire))
        .map(Value::from)
}

// fragment concat<?>( values: tensor<?>[], axis: integer ) -> ( value: tensor<?> );
pub fn concat(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let mut values: TVec<OutletId> = invocation.named_arg_as(builder, "values")?;
    if let Some(Some(dt)) = invocation.dt_from_quant_file.get(0) {
        for value in &mut values {
            if builder.model.node(value.node).outputs[value.slot].fact.datum_type != *dt {
                *value = builder.wire_as_outlets(ops::cast::cast(*dt), &[*value])?[0];
            }
        }
    }

    builder.wire(ops::array::TypedConcat::new(axis), &values)
}

// fragment slice<?>( input: tensor<?>, axes: integer[], begin: integer[], end: integer[] ) -> ( output: tensor<?> );
pub fn slice(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    let input_fact = builder.model.outlet_fact(wire[0])?.clone();
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let (mut begins, mut ends): (TVec<TDim>, TVec<TDim>) =
        builder.allowing_new_symbols(|builder| -> TractResult<_> {
            Ok((
                invocation.named_arg_as(builder, "begin")?,
                invocation.named_arg_as(builder, "end")?,
            ))
        })?;
    for (ix, d) in begins.iter_mut().enumerate() {
        if let Ok(i) = d.to_i64() {
            if i < 0 {
                *d += input_fact.shape[axes[ix]].to_dim();
            }
        }
    }

    // use "<=", no "<" end[axis] = 0 means "up to the end"
    // CAUTION: this notation is 1/ deprecated 2/ invalid with non trivial slicing
    for (ix, d) in ends.iter_mut().enumerate() {
        if let Ok(i) = d.to_i64() {
            if i <= 0 {
                *d += input_fact.shape[axes[ix]].to_dim();
            }
        }
    }
    let strides: TVec<isize> = invocation
        .named_arg_as(builder, "stride")
        .unwrap_or_else(|_| ends.iter().map(|_| 1).collect());
    izip!(axes, begins, ends, strides)
        .try_fold(wire, |wire, (axis, start, end, stride)| {
            let mut w =
                builder.wire_as_outlets(tract_core::ops::array::Slice { axis, start, end }, &wire);
            if stride != 1 {
                w = builder.wire_as_outlets(
                    tract_core::ops::downsample::Downsample::new(axis, stride, 0),
                    &w?,
                );
            }
            w
        })
        .map(Value::from)
}

// fragment squeeze<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
pub fn squeeze(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    axes.iter()
        .sorted()
        .rev()
        .try_fold(wire, |wire, &axis| {
            builder.wire_as_outlets(ops::change_axes::AxisOp::Rm(axis), &wire)
        })
        .map(Value::from)
}

// fragment unsqueeze<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
pub fn unsqueeze(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    axes.iter()
        .sorted()
        .try_fold(wire, |wire, &axis| {
            builder.wire_as_outlets(ops::change_axes::AxisOp::Add(axis), &wire)
        })
        .map(Value::from)
}

// fragment tile<?>( input: tensor<?>, repeats: integer[] ) -> ( output: tensor<?> );
pub fn tile(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let multipliers: TVec<TDim> = invocation.named_arg_as(builder, "repeats")?;
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    builder.wire(ops::array::Tile { multipliers }, &wire)
}

pub fn pad_mode(border: &str, value: Tensor) -> TractResult<tract_core::ops::array::PadMode> {
    Ok(match border {
        "constant" => PadMode::Constant(value.into_arc_tensor()),
        "replicated" => PadMode::Edge,
        "reflect" => PadMode::Reflect,
        _ => bail!("unsupported padding mode {}", border),
    })
}

// fragment pad( input: tensor<scalar>, padding: (integer, integer)[], border: string = 'constant', value: scalar = 0.0 ) -> ( output: tensor<scalar> );
pub fn pad(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    use tract_core::ops::array::Pad;
    let wire = tvec!(invocation.named_arg_as(builder, "input")?);
    let padding: TVec<TVec<usize>> = invocation.named_arg_as(builder, "padding")?;
    let padding: Vec<(usize, usize)> = padding.iter().map(|a| (a[0], a[1])).collect();
    let value: Tensor = tensor0(invocation.named_arg_as::<f32>(builder, "value")?);
    let border: String = invocation.named_arg_as(builder, "border")?;
    let mode = pad_mode(&border, value)?;
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

pub fn conv(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    conv_or_deconv(builder, invocation, false)
}

pub fn deconv(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
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
        PaddingSpec::Explicit(before, after)
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
) -> TractResult<Value> {
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
    let mut inputs = tvec!(input);

    let (group, pool_spec) =
        read_conv_parameters(builder, invocation, kernel.shape(), &input_fact)?;

    let output_dt =
        invocation.dt_from_quant_file.get(0).cloned().flatten().unwrap_or(DatumType::F32);
    if let (Some(a), Some(b), Some(c)) =
        (kernel.datum_type().qparams(), input_fact.datum_type.qparams(), output_dt.qparams())
    {
        inputs.push(builder.add_const(tensor0(a.zp_scale().0))?);
        inputs.push(builder.add_const(tensor0(a.zp_scale().1))?);
        inputs.push(builder.add_const(tensor0(b.zp_scale().0))?);
        inputs.push(builder.add_const(tensor0(b.zp_scale().1))?);
        inputs.push(builder.add_const(tensor0(c.zp_scale().0))?);
        inputs.push(builder.add_const(tensor0(c.zp_scale().1))?);
    };
    let bias: Arc<Tensor> = invocation.named_arg_as(builder, "bias")?;

    // Remove or reshape bias for efficient processing
    let bias: Option<Tensor> = if bias.is_uniform() && bias.cast_to_scalar::<f32>()? == 0.0 {
        None
    } else if bias.rank() > 1 {
        let output_channels = kernel.shape()[0];
        ensure!(
            output_channels == bias.len(),
            "Bias tensor should be scalar or have one value per output channel"
        );
        let mut reshaped_bias = bias.into_tensor();
        reshaped_bias.set_shape(&[output_channels])?;
        Some(reshaped_bias)
    } else {
        Some(bias.into_tensor())
    };

    let op: Box<dyn TypedOp> = if deconv {
        let output_shape = invocation.named_arg_as::<TVec<usize>>(builder, "output_shape")?;
        let output_shape = Some(output_shape).filter(|os| os.len() == pool_spec.rank());
        let adjustments = if let Some(output_shape) = output_shape {
            let input_shape = &input_fact
                .shape
                .as_concrete()
                .context("symbolic dimension not supported in deconv")?[2..];
            adjustments(&pool_spec, input_shape, &output_shape)?
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
            bias.map(Tensor::into_arc_tensor),
            adjustments,
            group,
        ))
    } else {
        Box::new(ConvUnary::new(
            pool_spec,
            KernelFormat::OIHW,
            kernel.clone(),
            DataFormat::NCHW.shape(input_fact.shape)?.c().to_usize()?,
            group,
            bias.map(Tensor::into_arc_tensor),
            Some(output_dt).filter(|dt| dt.is_quantized()),
        ))
    };
    builder.wire(op, &inputs)
}

fn pool_spec_for_pools(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
    shape: &[usize],
) -> TractResult<ops::cnn::PoolSpec> {
    let spatial_shape = DataFormat::NCHW.shape(shape)?.hw_dims().into();
    let dilation: TVec<usize> = invocation.named_arg_as(builder, "dilation")?;
    if dilation.len() > 0 && (dilation.len() != shape.len() || dilation[0] != 1 || dilation[1] != 1)
    {
        bail!("dilation should be like [1, 1, ... ]. Got dilation {:?}.", dilation);
    }
    let spatial_dilation = if dilation.iter().all(|it| *it == 1) || dilation.len() == 0 {
        None
    } else {
        Some(DataFormat::NCHW.shape(&dilation)?.hw_dims().into())
    };
    let stride: TVec<usize> = invocation.named_arg_as(builder, "stride")?;
    if stride.len() > 0 && (stride.len() != shape.len() || stride[0] != 1 || stride[1] != 1) {
        bail!("stride should be like [1, 1, ... ]. Got stride {:?}.", stride);
    }
    let spatial_stride = if stride.len() == 0 || stride.iter().all(|it| *it == 1) {
        None
    } else {
        Some(DataFormat::NCHW.shape(&stride)?.hw_dims().into())
    };
    let padding: TVec<TVec<usize>> = invocation.named_arg_as(builder, "padding")?;
    if padding.len() > 0 && (padding.len() != padding.len()) {
        bail!("padding should have the same rank as the input. Got padding {:?}.", padding);
    }
    let padding = if padding.len() == 0 {
        PaddingSpec::SameUpper
    } else {
        let mut before = tvec!();
        let mut after = tvec!();
        for p in padding {
            before.push(p[0]);
            after.push(p[1]);
        }
        let spatial_pool_bef = DataFormat::NCHW.shape(&before)?.hw_dims().into();
        let spatial_pool_aft = DataFormat::NCHW.shape(&after)?.hw_dims().into();
        PaddingSpec::ExplicitOnnxPool(spatial_pool_bef, spatial_pool_aft, false)
    };
    Ok(PoolSpec::new(
        DataFormat::NCHW,
        spatial_shape,
        padding,
        spatial_dilation,
        spatial_stride,
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
) -> TractResult<Value> {
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

pub fn sum_pool(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
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
pub fn reduce(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let reducer_name = invocation.invocation.id.0.split('_').next().unwrap();
    let reducer = match reducer_name {
        "sum" => ops::nn::Reducer::Sum,
        "min" => ops::nn::Reducer::Min,
        "max" => ops::nn::Reducer::Max,
        "argmin" => ops::nn::Reducer::ArgMin(false),
        "argmax" => ops::nn::Reducer::ArgMax(false),
        _ => bail!("unsupported reducer: {}", invocation.invocation.id.0),
    };
    let wire = builder.wire_as_outlets(ops::nn::Reduce::new(axes.clone(), reducer), &[input])?;
    if reducer_name != "sum" || !invocation.named_arg_as(builder, "normalize")? {
        return Ok(wire.into());
    }

    let fact = builder.model.outlet_fact(wire[0])?.clone();
    let input_shape = &builder.model.outlet_fact(input)?.shape;
    let cardinality: TDim = axes.iter().map(|ax| &input_shape[*ax]).product();
    let cardinality = builder.wire_as_outlets(
        ops::konst::Const::new(
            tensor0(cardinality).broadcast_into_rank(fact.rank())?.into_arc_tensor(),
        ),
        &[],
    )?;
    let cardinality =
        builder.wire_as_outlets(ops::cast::Cast::new(fact.datum_type), &cardinality)?;
    builder.wire(ops::math::div(), &[wire[0], cardinality[0]])
}

/*
 * fragment matmul( A: tensor<scalar>, B: tensor<scalar>, transposeA: logical = false, transposeB: logical = false ) -> ( C: tensor<scalar> );
 */
pub fn matmul(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let a: OutletId = invocation.named_arg_as(builder, "A")?;
    let b: OutletId = invocation.named_arg_as(builder, "B")?;
    let a_trans = invocation.named_arg_as(builder, "transposeA")?;
    let b_trans = invocation.named_arg_as(builder, "transposeB")?;
    let a_dt = builder.model.outlet_fact(a)?.datum_type;
    let b_dt = builder.model.outlet_fact(b)?.datum_type;
    let a_rank = builder.model.outlet_fact(a)?.rank();
    let b_rank = builder.model.outlet_fact(b)?.rank();
    let c_rank = a_rank.max(b_rank);
    let mut axes = AxesMapping::for_numpy_matmul(c_rank, a_trans, b_trans, false)?;
    let name = &*invocation.invocation.id.0;
    if a_dt.is_quantized() || b_dt.is_quantized() {
        for input in 0..7 {
            axes = axes.with_extra_input(2 + input)?;
        }
        let accum_dt = DatumType::QI32(QParams::ZpScale {
            scale: a_dt.zp_scale().1 * b_dt.zp_scale().1,
            zero_point: 0,
        });
        let c_dt = invocation.dt_from_quant_file.get(0).cloned().flatten().unwrap_or(accum_dt);

        let a_qp = a_dt.qparams().unwrap_or_default().zp_scale();
        let b_qp = b_dt.qparams().unwrap_or_default().zp_scale();
        let c_qp = c_dt.qparams().unwrap_or_default().zp_scale();
        let bias =
            builder.model.add_const(format!("{name}.bias"), Tensor::zero_scalar_dt(accum_dt)?)?;
        let a0 = builder.model.add_const(format!("{name}.a0"), rctensor0(a_qp.0))?;
        let a_scale = builder.model.add_const(format!("{name}.a_scale"), rctensor0(a_qp.1))?;
        let b0 = builder.model.add_const(format!("{name}.b0"), rctensor0(b_qp.0))?;
        let b_scale = builder.model.add_const(format!("{name}.b_scale"), rctensor0(b_qp.1))?;
        let c0 = builder.model.add_const(format!("{name}.c0"), rctensor0(c_qp.0))?;
        let c_scale = builder.model.add_const(format!("{name}.c_scale"), rctensor0(c_qp.1))?;

        builder.wire(
            ops::einsum::EinSum { axes, operating_dt: i32::datum_type(), q_params: Some(c_dt) },
            &[a, b, bias, a0, a_scale, b0, b_scale, c0, c_scale],
        )
    } else {
        builder.wire(ops::einsum::EinSum { axes, operating_dt: a_dt, q_params: None }, &[a, b])
    }
}

/*
* fragment select<?>(
condition: tensor<logical>,     # the condition for selecting the result
true_value: tensor<?>,          # the result when the condition is true
false_value: tensor<?> )        # the result when the condition is false
-> ( output: tensor<?> )
*/

pub fn select(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
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
) -> TractResult<Value> {
    let x = invocation.named_arg_as(builder, "x")?;
    let alpha = invocation.named_arg_as(builder, "alpha")?;
    builder.wire(ops::nn::leaky_relu(alpha), &[x])
}

/*
 * fragment stack<?>( values: tensor<?>[], axis: integer ) -> ( value: tensor<?> )
 *
 * Same as concat but on dedicated axis
 */

pub fn stack(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let mut values: TVec<OutletId> = invocation.named_arg_as(builder, "values")?;
    if let Some(Some(dt)) = invocation.dt_from_quant_file.get(0) {
        for value in &mut values {
            if builder.model.node(value.node).outputs[value.slot].fact.datum_type != *dt {
                *value = builder.wire_as_outlets(ops::cast::cast(*dt), &[*value])?[0];
            }
        }
    }

    for value in &mut values {
        // add unsqueeze
        *value = builder.wire_as_outlets(ops::change_axes::AxisOp::Add(axis), &[*value])?[0];
    }

    builder.wire(ops::array::TypedConcat::new(axis), &values)
}

/*
 * fragment unstack<?>( value: tensor<?>, axis: integer ) -> ( values: tensor<?>[] )
 *
 * Inverse of stack operator
 */
pub fn unstack(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = tvec!(invocation.named_arg_as(builder, "value")?);
    let axis: usize = invocation.named_arg_as(builder, "axis")?;

    let input_fact = builder.model.outlet_fact(wire[0])?.clone();

    (0..input_fact.shape[axis].clone().to_i32()?)
        .map(|start_int| {
            let start = start_int.to_dim();
            let end = (start_int + 1).to_dim();
            let sliced_wire = builder
                .wire_as_outlets(tract_core::ops::array::Slice { axis, start, end }, &wire)?;
            let squeezed_wire =
                builder.wire_as_outlets(ops::change_axes::AxisOp::Rm(axis), &sliced_wire)?;
            Ok(squeezed_wire[0])
        })
        .collect::<TractResult<TVec<_>>>()
        .map(Value::from)
}

/*
 * fragment softmax( x: tensor<scalar>, axes: integer[] = [1] ) -> ( y: tensor<scalar> )
 * {
 *    m = max_reduce(x, axes = axes);
 *    e = exp(x - m);
 *    y = e / sum_reduce(e, axes = axes);
 * }
 */

pub fn softmax(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let x = invocation.named_arg_as(builder, "x")?;
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;

    let input_fact = builder.model.outlet_fact(x)?.clone();
    let output_dt =
        invocation.dt_from_quant_file.get(0).cloned().flatten().unwrap_or(input_fact.datum_type);

    builder.wire(ops::nn::Softmax { axes, output_dt }, &[x])
}
