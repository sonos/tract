use crate::ast::Identifier;
use crate::ast::QuantFormat;
use crate::internal::*;
use crate::ser::*;
use tract_core::num_traits::Zero;
use tract_core::ops;
use tract_core::ops::cast::cast;
use tract_core::ops::cnn::Conv;
use tract_core::ops::cnn::Deconv;
use tract_core::ops::cnn::KernelFormat;
use tract_core::ops::cnn::PoolSpec;
use tract_core::ops::einsum::block_quant_aware_input_shape;
use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
use tract_core::ops::identity::PinConst;
use tract_core::ops::konst::Const;
use tract_core::ops::nn::DataFormat;
use tract_core::ops::nn::SoftmaxExp;
use tract_core::tract_data::itertools::Itertools;
use tract_linalg::block_quant::BlockQuantFact;

pub fn source(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::source::TypedSource,
) -> TractResult<Option<Arc<RValue>>> {
    if let Some(shape) = op.fact.shape.as_concrete() {
        if op.fact.datum_type == DatumType::F32 {
            return Ok(Some(invocation("external", &[], &[("shape", ints(shape))])));
        } else if op.fact.datum_type.is_quantized() {
            if let Some(qp) = QuantFormat::from_dt(node.outputs[0].fact.datum_type) {
                ast.quantization.insert(Identifier(node.name.to_string()), qp);
            }
            return Ok(Some(invocation("external", &[], &[("shape", ints(shape))])));
        }
    };
    Ok(None)
}

pub fn basic_matmul(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &PrefixMatMul,
) -> TractResult<Option<Arc<RValue>>> {
    let inputs = node.inputs.iter().map(|i| (*ast.mapping[i]).clone()).collect_vec();
    if op.transpose_c {
        Ok(Some(invocation(
            "matmul",
            &[Arc::new(inputs[1].clone()), Arc::new(inputs[0].clone())],
            &[("transposeA", logical(!op.transpose_b)), ("transposeB", logical(!op.transpose_a))],
        )))
    } else {
        Ok(Some(invocation(
            "matmul",
            &[Arc::new(inputs[0].clone()), Arc::new(inputs[1].clone())],
            &[("transposeA", logical(op.transpose_a)), ("transposeB", logical(op.transpose_b))],
        )))
    }
}

pub fn konst(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::konst::Const,
) -> TractResult<Option<Arc<RValue>>> {
    Ok(Some(ast.konst(&node.name, op.val())?))
}

pub fn comp(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::logic::Comp,
) -> TractResult<Option<Arc<RValue>>> {
    use ops::logic::Comp::*;
    let inputs = node.inputs.iter().map(|i| Arc::clone(&ast.mapping[i])).collect_vec();
    let name = match *op {
        Eq => "eq",
        NE => "ne",
        LT => "lt",
        GT => "gt",
        LTE => "le",
        GTE => "ge",
    };
    Ok(Some(invocation(name, &inputs, &[])))
}

pub fn concat(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::array::TypedConcat,
) -> TractResult<Option<Arc<RValue>>> {
    let wires = node
        .inputs
        .iter()
        .map(|i| Ok(ast.mapping[i].as_ref().clone()))
        .collect::<TractResult<TVec<RValue>>>()?;
    Ok(Some(invocation("concat", &[array(&wires).into()], &[("axis", numeric(op.axis))])))
}

pub fn slice(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::array::Slice,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    // end = 0 means "to the end" in early nnef specs.
    // the case begin = 0, end = 0: tract says "empty tensor", but nnef says "noop"
    // so serialize as begin = 0, end = -dim
    let end = if op.end.is_zero() && op.start == op.end {
        -ast.model.node_input_facts(node.id)?[0].shape[op.axis].clone()
    } else {
        op.end.clone()
    };
    Ok(Some(invocation(
        "slice",
        &[wire],
        &[
            ("axes", ints(&[op.axis])),
            ("begin", tdims(&[op.start.clone()])),
            ("end", tdims(&[end])),
        ],
    )))
}

pub fn tile(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::array::Tile,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation("tile", &[wire], &[("repeats", tdims(&op.multipliers))])))
}

pub fn dyn_tile(
    ast: &mut IntoAst,
    node: &TypedNode,
    _: &ops::array::DynTile,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let multiplier = ast.mapping[&node.inputs[1]].clone();
    Ok(Some(invocation("tile", &[wire], &[("repeats", (*multiplier).clone())])))
}

pub fn pad_mode(mode: &ops::array::PadMode, dt: DatumType) -> TractResult<(&str, Option<RValue>)> {
    use ops::array::PadMode;
    Ok(match &mode {
        PadMode::Constant(c) => (
            "constant",
            Some(if dt.is_float() {
                numeric(c.cast_to_scalar::<f32>()?)
            } else {
                numeric(c.cast_to_scalar::<i64>()?)
            }),
        ),
        PadMode::Reflect => ("reflect", None),
        PadMode::Edge => ("replicated", None),
    })
}

pub fn pad(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::array::Pad,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let dt = ast.model.outlet_fact(node.inputs[0])?.datum_type;
    let padding = array(op.pads.iter().map(|pair| ints(&[pair.0, pair.1])).collect::<TVec<_>>());
    let mut params = tvec!(("padding", padding));
    let (border, value) = pad_mode(&op.mode, dt)?;
    params.push(("border", string(border)));
    if let Some(value) = value {
        params.push(("value", value));
    }
    Ok(Some(invocation("pad", &[wire], &params)))
}

fn data_into_ncwh(data_format: DataFormat, geo_rank: usize, mut wire: Arc<RValue>) -> Arc<RValue> {
    use tract_core::ops::nn::DataFormat::*;
    if !data_format.has_n() {
        wire = invocation("unsqueeze", &[wire], &[("axes", ints(&[0]))]);
    }
    if data_format == NHWC || data_format == HWC {
        let mut perm: TVec<usize> = (0..geo_rank + 2).collect();
        perm[1..].rotate_right(1);
        wire = invocation("transpose", &[wire], &[("axes", ints(&perm))])
    }
    wire
}

fn data_from_ncwh(data_format: DataFormat, geo_rank: usize, mut wire: Arc<RValue>) -> Arc<RValue> {
    use tract_core::ops::nn::DataFormat::*;
    if data_format == NHWC || data_format == HWC {
        let mut perm: TVec<usize> = (0..geo_rank + 2).collect();
        perm[1..].rotate_left(1);
        wire = invocation("transpose", &[wire], &[("axes", ints(&perm))])
    }
    if !data_format.has_n() {
        wire = invocation("squeeze", &[wire], &[("axes", ints(&[0]))]);
    }
    wire
}

pub fn make_conv_named_args<'a>(
    node: &'a TypedNode,
    pool_spec: &'a PoolSpec,
    group: usize,
    deconv: bool,
    adjustments: Option<&[usize]>,
) -> TractResult<TVec<(&'a str, RValue)>> {
    use tract_core::ops::cnn::PaddingSpec;
    let output_shape = pool_spec.data_format.shape(node.outputs[0].fact.shape.to_tvec())?;
    let padding = match &pool_spec.padding {
        PaddingSpec::ExplicitOnnxPool(bef, after, _) | PaddingSpec::Explicit(bef, after) => array(
            bef.iter()
                .zip(after.iter())
                .map(|(a, b)| tuple_2(numeric(a), numeric(b)))
                .collect::<Vec<_>>(),
        ),
        PaddingSpec::SameUpper => array(&[]),
        PaddingSpec::SameLower => bail!("Unsupported padding scheme"),
        PaddingSpec::Valid => array(
            (0..pool_spec.rank()).map(|_| tuple_2(numeric(0), numeric(0))).collect::<Vec<_>>(),
        ),
    };
    let mut named_args = tvec![
        ("dilation", ints(&pool_spec.dilations())),
        ("stride", ints(&pool_spec.strides())),
        ("border", string("constant")),
        ("groups", numeric(group)),
        ("padding", padding),
    ];
    if deconv && adjustments.unwrap().iter().any(|a| *a != 0) {
        let output_shape = output_shape
            .hw_dims()
            .iter()
            .map(|d| d.to_usize())
            .collect::<TractResult<TVec<_>>>()?;
        named_args.push(("output_shape", ints(&output_shape)));
    };
    Ok(named_args)
}

#[allow(clippy::too_many_arguments)]
pub fn conv_or_deconv(
    ast: &mut IntoAst,
    node: &TypedNode,
    pool_spec: &PoolSpec,
    group: usize,
    deconv: bool,
    adjustments: Option<&[usize]>,
) -> TractResult<Option<Arc<RValue>>> {
    let mut wire = ast.mapping[&node.inputs[0]].clone();
    let kernel = ast.mapping[&node.inputs[1]].clone();
    let bias = ast.mapping[&node.inputs[2]].clone();
    let data_format = pool_spec.data_format;
    ensure!(data_format.has_n());
    if data_format.c_is_last() {
        let mut perm: TVec<usize> = (0..pool_spec.rank() + 1).collect();
        perm.insert(1, pool_spec.rank() + 1);
        wire = invocation("transpose", &[wire], &[("axes", ints(&perm))]);
    }
    wire = ast.force_variable(format!("{}_input", node.name), &wire);

    let inputs = tvec![wire, kernel, bias];
    let named_args = make_conv_named_args(node, pool_spec, group, deconv, adjustments)?;

    let name = if deconv { "deconv" } else { "conv" };
    wire = invocation(name, &inputs, &named_args);
    // need to force quantization storage as output code may miss it
    let var_name = Identifier(format!("{}_{}", node.name, name));
    if let Some(qp) = QuantFormat::from_dt(node.outputs[0].fact.datum_type) {
        ast.quantization.insert(var_name.clone(), qp);
    }
    wire = ast.force_variable(var_name, &wire);

    if data_format.c_is_last() {
        let mut perm: TVec<usize> = (0..pool_spec.rank() + 2).collect();
        perm.remove(1);
        perm.push(1);
        wire = invocation("transpose", &[wire], &[("axes", ints(&perm))]);
    }
    Ok(Some(wire))
}

pub fn conv(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::cnn::conv::Conv,
) -> TractResult<Option<Arc<RValue>>> {
    conv_or_deconv(ast, node, &op.pool_spec, op.group, false, None)
}

pub fn deconv(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::cnn::deconv::Deconv,
) -> TractResult<Option<Arc<RValue>>> {
    conv_or_deconv(ast, node, &op.pool_spec, op.group, true, Some(&op.adjustments))
}

fn cnn_pool_fragment(
    ast: &mut IntoAst,
    data_format: DataFormat,
    geo_rank: usize,
    op_name: &str,
) -> Identifier {
    if data_format == DataFormat::NCHW {
        return op_name.into();
    }
    let fragment_name =
        Identifier(format!("tract_{op_name}_{data_format:?}_{geo_rank}D").to_lowercase());
    if ast.fragments.contains_key(&fragment_name) {
        return fragment_name;
    }

    let mut body = vec![];
    let mut fragment =
        ast.framework.stdlib.iter().find(|f| f.decl.id.0 == op_name).unwrap().clone();
    fragment.decl.id = fragment_name.clone();

    let mut wire = ident("input").into();
    wire = data_into_ncwh(data_format, geo_rank, wire);

    body.push(assignment("nchw", wire));
    wire = invocation(
        op_name,
        &[ident("nchw").into()],
        &fragment
            .decl
            .parameters
            .iter()
            .skip(1)
            .map(|f| (&*f.id.0, ident(&f.id)))
            .collect::<Vec<_>>(),
    );
    body.push(assignment(op_name, wire));

    wire = data_from_ncwh(data_format, geo_rank, ident(op_name).into());

    body.push(assignment("output", wire));
    fragment.body = Some(body);
    ast.fragments.insert(fragment_name.clone(), fragment);
    fragment_name
}

fn cnn_pool(
    ast: &mut IntoAst,
    node: &TypedNode,
    op_name: &str,
    pool_spec: &tract_core::ops::cnn::PoolSpec,
    normalize_arg: Option<(&'static str, RValue)>,
) -> TractResult<Option<Arc<RValue>>> {
    use tract_core::ops::cnn::PaddingSpec;
    let mut wire = ast.mapping[&node.inputs[0]].clone();
    wire = ast.force_variable(format!("{}_input", node.name), &wire);
    let conv_fragment = cnn_pool_fragment(ast, pool_spec.data_format, pool_spec.rank(), op_name);
    let padding = match &pool_spec.padding {
        PaddingSpec::ExplicitOnnxPool(bef, after, _) | PaddingSpec::Explicit(bef, after) => Some(
            bef.iter()
                .zip(after.iter())
                .map(|(a, b)| tuple_2(numeric(a), numeric(b)))
                .collect::<Vec<_>>(),
        ),
        PaddingSpec::SameUpper => None,
        PaddingSpec::SameLower => bail!("Unsupported padding scheme"),
        PaddingSpec::Valid => {
            Some((0..pool_spec.rank()).map(|_| tuple_2(numeric(0), numeric(0))).collect::<Vec<_>>())
        }
    };
    let mut size = tvec!(1, 1);
    size.extend(pool_spec.kernel_shape.iter().cloned());
    let mut strides = tvec!(1, 1);
    strides.extend(pool_spec.strides().iter().cloned());
    let mut dilations = tvec!(1, 1);
    dilations.extend(pool_spec.dilations().iter().cloned());
    let padding = if let Some(pad) = padding {
        let mut full_padding =
            vec![tuple_2(numeric(0), numeric(0)), tuple_2(numeric(0), numeric(0))];
        full_padding.extend(pad.iter().cloned());
        array(full_padding)
    } else {
        array(&[])
    };
    let mut params = tvec!(
        ("size", ints(&size)),
        ("dilation", ints(&dilations)),
        ("stride", ints(&strides)),
        ("border", string("ignore")),
        ("padding", padding),
    );
    if let Some(normalize_arg) = normalize_arg {
        params.push(normalize_arg);
    };
    wire = invocation(conv_fragment, &[wire], &params);
    wire = ast.force_variable(&node.name, &wire);
    Ok(Some(wire))
}

pub fn max_pool(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::cnn::MaxPool,
) -> TractResult<Option<Arc<RValue>>> {
    cnn_pool(ast, node, "max_pool", &op.pool_spec, None)
}

pub fn sum_pool(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::cnn::SumPool,
) -> TractResult<Option<Arc<RValue>>> {
    cnn_pool(ast, node, "box", &op.pool_spec, Some(("normalize", logical(op.normalize))))
}

pub fn ser_axis_op(op: &ops::change_axes::AxisOp, wire: Arc<RValue>, rank: usize) -> Arc<RValue> {
    match op {
        AxisOp::Rm(axis) => invocation("squeeze", &[wire], &[("axes", ints(&[*axis]))]),
        AxisOp::Add(axis) => invocation("unsqueeze", &[wire], &[("axes", ints(&[*axis]))]),
        AxisOp::Move(from, to) => {
            let mut perm: TVec<usize> = (0..rank).collect();
            if from < to {
                perm[*from..(to + 1)].rotate_left(1);
            } else {
                perm[*to..(from + 1)].rotate_right(1);
            }
            invocation("transpose", &[wire], &[("axes", ints(&perm))])
        }
        AxisOp::Reshape(start, from, to) => invocation(
            "reshape",
            &[wire],
            &[
                ("shape", tdims(to)),
                ("axis_start", numeric(start)),
                ("axis_count", numeric(from.len())),
            ],
        ),
    }
}

pub fn axis_op(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::change_axes::AxisOp,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let rank = node.outputs[0].fact.rank();
    Ok(Some(ser_axis_op(op, wire, rank)))
}

pub fn reduce(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::nn::Reduce,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let oper = match op.reducer {
        ops::nn::Reducer::ArgMax(last) if !last => "argmax_reduce",
        ops::nn::Reducer::ArgMin(last) if !last => "argmin_reduce",
        ops::nn::Reducer::Sum => "sum_reduce",
        ops::nn::Reducer::Max => "max_reduce",
        ops::nn::Reducer::Min => "min_reduce",
        _ => return Ok(None),
    };
    Ok(Some(invocation(oper, &[wire], &[("axes", ints(&op.axes))])))
}

pub fn select(
    ast: &mut IntoAst,
    node: &TypedNode,
    _op: &ops::logic::Iff,
) -> TractResult<Option<Arc<RValue>>> {
    Ok(Some(invocation(
        "select",
        &node.inputs.iter().map(|o| ast.mapping[o].clone()).collect::<TVec<_>>(),
        &[],
    )))
}

pub fn leaky_relu(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op_as::<ops::element_wise::ElementWiseOp>().context("Wrong op")?;
    let op = op.0.downcast_ref::<ops::nn::LeakyRelu>().context("Wrong op")?;
    Ok(Some(invocation(
        "leaky_relu",
        &node.inputs.iter().map(|o| ast.mapping[o].clone()).collect::<TVec<_>>(),
        &[("alpha", RValue::Literal(op.alpha.into()))],
    )))
}

pub fn softmax(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::nn::Softmax,
) -> TractResult<Option<Arc<RValue>>> {
    if op.exp != SoftmaxExp::default() {
        return Ok(None);
    }
    let litteral_axes: Vec<_> = op.axes.iter().map(|&it| (it as i64).into()).collect();
    Ok(Some(invocation(
        "softmax",
        &[ast.mapping[&node.inputs[0]].clone()],
        &[("axes", RValue::Literal(crate::ast::Literal::Array(litteral_axes)))],
    )))
}

pub fn rewrite_block_quant_const_to_scalar(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    prefix: &str,
    op: &Const,
) -> TractResult<Option<TypedModelPatch>> {
    let fact = &node.outputs[0].fact;
    if fact.shape.len() == 0
        || fact.datum_type.is_integer()
        || !fact.opaque_fact.as_ref().is_some_and(|of| of.is::<BlockQuantFact>())
    {
        return Ok(None);
    };
    ensure!(fact.shape.volume().is_one());
    let mut patch = TypedModelPatch::default();
    let mut new_fact = fact.clone();
    new_fact.shape = ShapeFact::scalar();
    let new_tensor = op.val().to_scalar_tensor()?.into_arc_tensor();
    new_fact.konst = Some(new_tensor.clone());
    let mut output = patch.wire_node(
        prefix,
        Const::new_with_opt_opaque_fact(new_tensor, fact.opaque_fact.clone())?,
        &[],
    )?;
    output = patch.wire_node(format!("{prefix}.lock"), PinConst, &output)?;
    for i in 0..fact.rank() {
        output = patch.wire_node(format!("{prefix}.{i}"), AxisOp::Add(0), &output)?;
    }

    patch.shunt_outside(model, node.id.into(), output[0])?;
    Ok(Some(patch))
}

pub fn rewrite_matmul_to_same_rank(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    prefix: &str,
    op: &PrefixMatMul,
) -> TractResult<Option<TypedModelPatch>> {
    let a_rank = block_quant_aware_input_shape(model.outlet_fact(node.inputs[0])?)?.len();
    let b_rank = block_quant_aware_input_shape(model.outlet_fact(node.inputs[1])?)?.len();
    if a_rank == b_rank {
        return Ok(None);
    }
    let mut patch = TypedModelPatch::default();
    let mut inputs = patch.taps(model, &node.inputs)?;
    for i in a_rank..a_rank.max(b_rank) {
        inputs[0] =
            patch.wire_node(format!("{prefix}.extra_a_axis.{i}"), AxisOp::Add(0), &[inputs[0]])?[0];
    }
    for i in b_rank..a_rank.max(b_rank) {
        inputs[1] =
            patch.wire_node(format!("{prefix}.extra_b_axis.{i}"), AxisOp::Add(0), &[inputs[1]])?[1];
    }
    let result = patch.wire_node(prefix, *op, &inputs)?[0];
    patch.shunt_outside(model, node.id.into(), result)?;
    Ok(Some(patch))
}

pub fn rewrite_consistent_quantized_conv(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    op: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    let facts = model.node_input_facts(node.id)?;
    if facts.len() > 3 {
        ensure!(facts[3..9].iter().all(|fact| fact.konst.is_some()));
        for ix in [0, 1] {
            let fact = model.outlet_fact(node.inputs[ix])?;
            if !fact.datum_type.is_quantized() {
                let mut patch = TypedModelPatch::default();
                let mut wire = patch.taps(model, &node.inputs)?;
                let dt = fact.datum_type.quantize(QParams::ZpScale {
                    zero_point: facts[3 + 2 * ix]
                        .konst
                        .as_ref()
                        .unwrap()
                        .cast_to_scalar::<i32>()?,
                    scale: facts[4 + 2 * ix].konst.as_ref().unwrap().cast_to_scalar::<f32>()?,
                });
                wire[ix] =
                    patch.wire_node(format!("{name}.cast_to_q_{ix}"), cast(dt), &[wire[ix]])?[0];
                let output = patch.wire_node(name, op.clone(), &wire)?[0];
                patch.shunt_outside(model, node.id.into(), output)?;
                return Ok(Some(patch));
            }
        }
    }
    Ok(None)
}

pub fn rewrite_kernel_conv_in_oihw(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    rewrite_kernel_in_oihw(
        model,
        node,
        name,
        conv.kernel_fmt,
        conv.group,
        Box::new(Conv { kernel_fmt: KernelFormat::OIHW, ..conv.clone() }),
    )
}

pub fn rewrite_kernel_deconv_in_oihw(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &Deconv,
) -> TractResult<Option<TypedModelPatch>> {
    rewrite_kernel_in_oihw(
        model,
        node,
        name,
        conv.kernel_format,
        conv.group,
        Box::new(Deconv { kernel_format: KernelFormat::OIHW, ..conv.clone() }),
    )
}

fn rewrite_kernel_in_oihw(
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    fmt: KernelFormat,
    group: usize,
    new: Box<dyn TypedOp>,
) -> TractResult<Option<TypedModelPatch>> {
    if fmt == KernelFormat::OIHW {
        return Ok(None);
    }
    let mut patch = TypedModelPatch::default();
    let mut wire = patch.taps(model, &node.inputs)?;
    let prefix = format!("{name}.kernel_reorg");
    for (ix, op) in fmt
        .kernel_as_group_o_i_h_w_ops(&patch.outlet_fact(wire[1])?.shape, group)
        .into_iter()
        .enumerate()
    {
        wire[1] = patch.wire_node(format!("{prefix}.{ix}"), op, &[wire[1]])?[0];
    }
    wire[1] =
        AxisOp::wire_collapse_axis(&mut patch, format!("{name}.kernel_reorg_go"), wire[1], 0)?[0];
    wire = patch.wire_node(name, new, &wire)?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(Some(patch))
}
