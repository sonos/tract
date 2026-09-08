use tract_core::internal::*;
use tract_core::ops as core;
use tract_core::ops::cast::wire_cast;
use tract_core::ops::cast::Cast;
use tract_core::ops::change_axes::wire_with_rank_broadcast;
use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::math::add;
use tract_core::ops::nn::Softmax;
use tract_core::ops::nn::{CoordTransformer, Interpolator, Nearest, Reduce, Reducer, Resize};
use tract_core::prelude::tract_itertools::Itertools;

use crate::registry::{DeserOp, Registry};
use crate::ser::BuiltinOp;
use crate::ser::SubgraphBuilder;
use crate::tflite::ArgMaxOptions;
use crate::tflite::ArgMaxOptionsArgs;
use crate::tflite::BatchMatMulOptions;
use crate::tflite::BatchMatMulOptionsArgs;
use crate::tflite::BuiltinOptions;
use crate::tflite::ExpandDimsOptions;
use crate::tflite::ExpandDimsOptionsArgs;
use crate::tflite::ReducerOptions;
use crate::tflite::ReducerOptionsArgs;
use crate::tflite::SoftmaxOptions;
use crate::tflite::SoftmaxOptionsArgs;
use crate::tflite::TensorType;
use crate::tflite::{BuiltinOperator, FullyConnectedOptionsWeightsFormat};

pub fn register_all(reg: &mut Registry) {
    reg.reg_to_tflite(ser_matmul);
    reg.reg_to_tract(BuiltinOperator::BATCH_MATMUL, de_batch_matmul);

    reg.reg_to_tract(BuiltinOperator::FULLY_CONNECTED, de_fully_connected);
    reg.reg_to_tract(BuiltinOperator::MEAN, de_reduce_mean);
    reg.reg_to_tflite(ser_softmax);
    reg.reg_to_tract(BuiltinOperator::SOFTMAX, de_softmax);

    reg.reg_to_tract(BuiltinOperator::RESIZE_BILINEAR, de_resize_bilinear);
    reg.reg_to_tract(BuiltinOperator::RESIZE_NEAREST_NEIGHBOR, de_resize_nearest);

    reg.reg_to_tract(BuiltinOperator::RELU, de_relu);
    reg.reg_to_tract(BuiltinOperator::RELU6, de_relu6);

    reg.reg_to_tflite(ser_reduce);
    reg.reg_to_tract(BuiltinOperator::REDUCE_MAX, |op| de_reduce(op, Reducer::Max));
    reg.reg_to_tract(BuiltinOperator::REDUCE_MIN, |op| de_reduce(op, Reducer::Min));
    reg.reg_to_tract(BuiltinOperator::SUM, |op| de_reduce(op, Reducer::Sum));
    reg.reg_to_tract(BuiltinOperator::REDUCE_PROD, |op| de_reduce(op, Reducer::Prod));
}

fn de_batch_matmul(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (a, b) = args_2!(op.facts()?);
    let options = builtin!(op, builtin_options_as_batch_mat_mul_options);
    ensure!(a.datum_type.is_float());
    ensure!(!options.asymmetric_quantize_inputs());
    ensure!(a.rank() == b.rank());
    let rank = a.rank();
    let mut axes = tvec!(
        Axis::new('M', 2, 1).input(0, rank - 2 + options.adj_x() as usize).output(0, rank - 2),
        Axis::new('N', 2, 1).input(1, rank - 1 - options.adj_y() as usize).output(0, rank - 1),
        Axis::new('K', 2, 1)
            .input(0, rank - 1 - options.adj_x() as usize)
            .input(1, rank - 2 + options.adj_y() as usize)
    );
    for (ix, repr) in ('a'..).take(rank - 2).enumerate() {
        axes.push(Axis::new(repr, 2, 1).input(0, ix).input(1, ix).output(0, ix));
    }
    let axes: AxesMapping = AxesMapping::new(2, 1, axes)?;
    let einsum = EinSum { axes, q_params: None, operating_dt: a.datum_type };
    op.ctx.target.wire_node(op.prefix, einsum, op.inputs)
}

fn de_fully_connected(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (input, weights, bias) = args_3!(op.facts()?);
    let options = builtin!(op, builtin_options_as_fully_connected_options);
    ensure!(options.weights_format() == FullyConnectedOptionsWeightsFormat::DEFAULT);
    ensure!(!options.asymmetric_quantize_inputs());
    ensure!(input.rank() == 2);
    ensure!(weights.rank() == 2);
    ensure!(bias.rank() == 1);
    let mut inputs: TVec<OutletId> = op.inputs.into();
    let wires = if input.datum_type.is_float() {
        let axes = "BI,OI->BO".parse()?;
        let einsum = EinSum { axes, q_params: None, operating_dt: input.datum_type };
        let mut wires = op.ctx.target.wire_node(op.prefix, einsum, &inputs[0..2])?;
        if inputs.len() == 3 {
            let bias = op.ctx.target.wire_node(
                format!("{}.bias_rank", op.prefix),
                AxisOp::Add(0),
                &inputs[2..3],
            )?;
            wires = op.ctx.target.wire_node(
                format!("{}.bias", op.prefix),
                add(),
                &[wires[0], bias[0]],
            )?;
        }
        wires
    } else {
        let qp = super::linearops_quantization_suport(op, &input, &mut inputs)?;
        let axes = "BI,OI,O,,,,,,->BO".parse()?;
        let einsum = EinSum { axes, q_params: qp, operating_dt: i32::datum_type() };
        op.ctx.target.wire_node(op.prefix, einsum, &inputs)?
    };
    super::wire_fused_activation(op, &wires, &options.fused_activation_function())
}

fn de_reduce(op: &mut DeserOp, reducer: Reducer) -> TractResult<TVec<OutletId>> {
    let (_, axes) = args_2!(op.facts()?);
    let options = builtin!(op, builtin_options_as_reducer_options);
    let axes: TVec<usize> = axes
        .konst
        .as_ref()
        .unwrap()
        .try_as_plain()?
        .as_slice::<i32>()?
        .iter()
        .map(|d| *d as usize)
        .sorted()
        .collect();
    let p = &op.prefix;
    let mut wire = op.ctx.target.wire_node(
        format!("{p}.reduce"),
        core::nn::Reduce::new(axes.clone(), reducer),
        &[op.inputs[0]],
    )?;
    if !options.keep_dims() {
        for axis in axes.iter().rev() {
            wire =
                op.ctx.target.wire_node(format!("{p}.rm_axis_{axis}"), AxisOp::Rm(*axis), &wire)?;
        }
    }
    Ok(wire)
}

fn de_reduce_mean(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (input, axes) = args_2!(op.facts()?);
    let axes: TVec<usize> = axes
        .konst
        .as_ref()
        .unwrap()
        .try_as_plain()?
        .as_slice::<i32>()?
        .iter()
        .map(|d| *d as usize)
        .sorted()
        .collect();
    let norm: TDim = axes.iter().map(|d| &input.shape[*d]).product();
    let wire = de_reduce(op, Reducer::Sum)?;
    let p = &op.prefix;
    let norm = op.ctx.target.add_const(format!("{p}.card"), tensor0(norm))?;
    let norm = op.ctx.target.wire_node(
        format!("{p}.as_float"),
        Cast { to: f32::datum_type() },
        &[norm],
    )?;
    let norm = op.ctx.target.wire_node(format!("{p}.recip"), core::math::recip(), &norm)?;
    wire_with_rank_broadcast(op.prefix, op.ctx.target, core::quant::scale(), &[norm[0], wire[0]])
}

fn de_softmax(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input = args_1!(op.facts()?);
    let options = builtin!(op, builtin_options_as_softmax_options);
    ensure!(options.beta() == 1.0);
    let quant_output_dt = Some(input.datum_type).filter(|dt| !dt.is_float());
    let softmax = Softmax { axes: tvec!(input.rank() - 1), quant_output_dt, ..Softmax::default() };
    op.ctx.target.wire_node(op.prefix, softmax, op.inputs)
}

/// The size input names the two spatial axes only, and the coordinate
/// transformation is spelled as two flags rather than a name: neither set is
/// TensorFlow's original mapping, where the output grid simply spans the input
/// one.
fn de_resize_bilinear(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let options = builtin!(op, builtin_options_as_resize_bilinear_options);
    let coord_transformer = if options.align_corners() {
        CoordTransformer::AlignCorners
    } else if options.half_pixel_centers() {
        CoordTransformer::HalfPixel
    } else {
        CoordTransformer::Asymmetric
    };
    de_resize(op, coord_transformer, Interpolator::Linear, Nearest::Floor)
}

/// Nearest neighbour reads its half-pixel mode through a transformation of its
/// own, and rounds up on a tie where bilinear has nothing to round.
fn de_resize_nearest(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let options = builtin!(op, builtin_options_as_resize_nearest_neighbor_options);
    let coord_transformer = if options.align_corners() {
        CoordTransformer::AlignCorners
    } else if options.half_pixel_centers() {
        CoordTransformer::TfHalfPixelForNn
    } else {
        CoordTransformer::Asymmetric
    };
    let nearest =
        if options.half_pixel_centers() { Nearest::Floor } else { Nearest::RoundPreferCeil };
    de_resize(op, coord_transformer, Interpolator::Nearest, nearest)
}

fn de_resize(
    op: &mut DeserOp,
    coord_transformer: CoordTransformer,
    interpolator: Interpolator,
    nearest: Nearest,
) -> TractResult<TVec<OutletId>> {
    let (input, sizes) = args_2!(op.facts()?);
    ensure!(input.rank() == 4, "Resize expects NHWC, got rank {}", input.rank());
    let sizes = sizes.konst.clone().context("Dynamic resize size is not supported")?;
    let sizes = sizes.cast_to::<i64>()?;
    let sizes = sizes.try_as_plain()?.as_slice::<i64>()?;
    ensure!(sizes.len() == 2, "Resize expects a size per spatial axis, got {sizes:?}");
    let shape = &input.shape;
    let full = tensor1(&[shape[0].clone(), sizes[0].to_dim(), sizes[1].to_dim(), shape[3].clone()]);
    let sizes = op.ctx.target.add_const(format!("{}.sizes", op.prefix), full)?;
    op.ctx.target.wire_node(
        op.prefix,
        Resize {
            coord_transformer,
            interpolator,
            nearest,
            optional_scales_input: None,
            optional_sizes_input: Some(1),
        },
        &[op.inputs[0], sizes],
    )
}

pub fn de_relu(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input = op.inputs[0];
    let zero = op.ctx.target.add_const(format!("{}.zero", op.prefix), tensor0(0f32))?;
    let wires = wire_cast(
        op.prefix,
        op.ctx.target,
        &[input, zero],
        op.ctx.target.outlet_fact(input)?.datum_type,
    )?;
    wire_with_rank_broadcast(
        format!("{}.relu", op.prefix),
        op.ctx.target,
        core::math::max(),
        &wires,
    )
}

pub fn de_relu6(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input = de_relu(op)?[0];
    let six = op.ctx.target.add_const(format!("{}.six", op.prefix), tensor0(6f32))?;
    let wires = wire_cast(
        op.prefix,
        op.ctx.target,
        &[input, six],
        op.ctx.target.outlet_fact(input)?.datum_type,
    )?;
    wire_with_rank_broadcast(
        format!("{}.relu6", op.prefix),
        op.ctx.target,
        core::math::min(),
        &wires,
    )
}

fn ser_matmul(
    builder: &mut SubgraphBuilder,
    model: &TypedModel,
    node: &TypedNode,
    op: &PrefixMatMul,
) -> TractResult<()> {
    let mut inputs =
        [builder.map_outlet(model, node.inputs[0])?, builder.map_outlet(model, node.inputs[1])?];
    let (adj_x, adj_y) = if op.transpose_c {
        inputs.swap(0, 1);
        (!op.transpose_b, !op.transpose_a)
    } else {
        (op.transpose_a, op.transpose_b)
    };
    let output = builder.map_outlets(model, [OutletId::from(node.id)])?;
    let options = BatchMatMulOptions::create(
        builder.fb(),
        &BatchMatMulOptionsArgs { adj_x, adj_y, asymmetric_quantize_inputs: false },
    );
    builder.write_op_with_options(
        &inputs,
        &output,
        BuiltinOp::new(126, 1, BuiltinOperator::BATCH_MATMUL, BuiltinOptions::BatchMatMulOptions),
        options.as_union_value(),
    )?;
    Ok(())
}

fn ser_reduce(
    builder: &mut SubgraphBuilder,
    model: &TypedModel,
    node: &TypedNode,
    op: &Reduce,
) -> TractResult<()> {
    let axes = builder.write_fact(
        format!("{}.axes", node.name),
        TypedFact::try_from(tensor1(&op.axes.iter().map(|axis| *axis as i32).collect_vec()))?,
    )?;
    let inputs = [builder.map_outlet(model, node.inputs[0])?, axes];
    let output = builder.map_outlets(model, [OutletId::from(node.id)])?;
    if matches!(op.reducer, Reducer::ArgMin(_) | Reducer::ArgMax(_)) {
        let mut intermediate_shape = model.outlet_fact(node.inputs[0])?.shape.to_vec();
        for axis in op.axes.iter().sorted().rev() {
            intermediate_shape.remove(*axis);
        }
        let intermediate_fact = i32::fact(intermediate_shape);
        let intermediate_tensor =
            builder.write_fact(format!("{}.removed_axes", node.name), intermediate_fact)?;
        let options = ArgMaxOptions::create(
            builder.fb(),
            &ArgMaxOptionsArgs { output_type: TensorType::INT64 },
        );
        builder.write_op_with_options(
            &inputs,
            &[intermediate_tensor],
            BuiltinOp::new(56, 1, BuiltinOperator::ARG_MAX, BuiltinOptions::ArgMaxOptions),
            options.as_union_value(),
        )?;
        let expand_dim_options = ExpandDimsOptions::create(builder.fb(), &ExpandDimsOptionsArgs {});
        builder.write_op_with_options(
            &[intermediate_tensor, axes],
            &output,
            BuiltinOp::new(70, 1, BuiltinOperator::EXPAND_DIMS, BuiltinOptions::ExpandDimsOptions),
            expand_dim_options.as_union_value(),
        )?;
        Ok(())
    } else {
        let options = ReducerOptions::create(builder.fb(), &ReducerOptionsArgs { keep_dims: true });
        ensure!(model.outlet_fact(node.inputs[0])?.datum_type != f64::datum_type());
        match op.reducer {
            Reducer::Max => builder.write_op_with_options(
                &inputs,
                &output,
                BuiltinOp::new(82, 1, BuiltinOperator::REDUCE_MAX, BuiltinOptions::ReducerOptions),
                options.as_union_value(),
            ),
            Reducer::Min => builder.write_op_with_options(
                &inputs,
                &output,
                BuiltinOp::new(89, 1, BuiltinOperator::REDUCE_MIN, BuiltinOptions::ReducerOptions),
                options.as_union_value(),
            ),
            Reducer::Prod => builder.write_op_with_options(
                &inputs,
                &output,
                BuiltinOp::new(81, 1, BuiltinOperator::REDUCE_PROD, BuiltinOptions::ReducerOptions),
                options.as_union_value(),
            ),
            Reducer::Sum => builder.write_op_with_options(
                &inputs,
                &output,
                BuiltinOp::new(74, 1, BuiltinOperator::SUM, BuiltinOptions::ReducerOptions),
                options.as_union_value(),
            ),
            Reducer::All | Reducer::Any => todo!(),
            Reducer::ArgMin(_) | Reducer::ArgMax(_) | Reducer::MeanOfSquares => unreachable!(),
        }
    }
}

fn ser_softmax(
    builder: &mut SubgraphBuilder,
    model: &TypedModel,
    node: &TypedNode,
    op: &Softmax,
) -> TractResult<()> {
    let rank = model.outlet_fact(node.inputs[0])?.rank();
    let input = builder.map_outlet(model, node.inputs[0])?;
    let output = builder.map_outlet(model, node.id.into())?;
    ensure!(&*op.axes == &[rank - 1]);
    let options = SoftmaxOptions::create(builder.fb(), &SoftmaxOptionsArgs { beta: 1f32 });
    builder.write_op_with_options(
        &[input],
        &[output],
        BuiltinOp::new(25, 1, BuiltinOperator::SOFTMAX, BuiltinOptions::SoftmaxOptions),
        options.as_union_value(),
    )
}
