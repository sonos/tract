use crate::internal::*;
use crate::ser::*;
use tract_core::ops;
use tract_core::ops::nn::DataFormat;

pub fn source(
    _ast: &mut IntoAst,
    _node: &TypedNode,
    op: &ops::source::TypedSource,
) -> TractResult<Option<Arc<RValue>>> {
    if op.fact.datum_type == DatumType::F32 {
        Ok(Some(invocation(
            "external",
            &[],
            &[("shape", ints(&*op.fact.shape.as_finite().unwrap()))],
        )))
    } else {
        Ok(None)
    }
}

pub fn konst(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::konst::Const,
) -> TractResult<Option<Arc<RValue>>> {
    Ok(Some(ast.konst(&node.name, &op.0)))
}

pub fn concat(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::array::TypedConcat,
) -> TractResult<Option<Arc<RValue>>> {
    let mut inputs = node.inputs.iter();
    let wires = op
        .slices
        .iter()
        .enumerate()
        .map(|(ix, s)| match s {
            ops::array::ConcatSlice::Var => ast.mapping[inputs.next().unwrap()].as_ref().clone(),
            ops::array::ConcatSlice::Const(t) => {
                ast.konst(format!("{}.const-{}", node.name, ix), t).as_ref().clone()
            }
        })
        .collect::<TVec<RValue>>();
    Ok(Some(invocation("concat", &[array(&wires).into()], &[("axis", numeric(op.axis))])))
}

pub fn slice<D: DimLike>(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::array::Slice<D>,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let start = op.start.to_usize()?;
    let end = op.end.to_usize()?;
    Ok(Some(invocation(
        "slice",
        &[wire],
        &[("axes", ints(&[op.axis])), ("begin", ints(&[start])), ("end", ints(&[end]))],
    )))
}

pub fn tile(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::array::Tile,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation("tile", &[wire], &[("repeat", ints(&op.multipliers))])))
}

pub fn pad(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::array::Pad,
) -> TractResult<Option<Arc<RValue>>> {
    use ops::array::PadMode;
    let wire = ast.mapping[&node.inputs[0]].clone();
    let dt = ast.model.outlet_fact(node.inputs[0])?.datum_type;
    let padding = array(&op.pads.iter().map(|pair| ints(&[pair.0, pair.1])).collect::<TVec<_>>());
    let mut params = tvec!(("padding", padding));
    let border = match &op.mode {
        PadMode::Constant(c) => {
            params.push((
                "value",
                if dt.is_float() {
                    numeric(c.cast_to_scalar::<f32>()?)
                } else {
                    numeric(c.cast_to_scalar::<i64>()?)
                },
            ));
            "constant"
        }
        PadMode::Reflect => "reflect",
        PadMode::Edge => "replicated",
    };
    params.push(("border", string(border)));
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

fn conv_fragment<'a>(ast: &'a mut IntoAst, data_format: DataFormat, geo_rank: usize) -> String {
    if data_format == DataFormat::NCHW {
        return "conv".into();
    }
    let fragment_name = format!("tract_conv_{:?}_{}D", data_format, geo_rank).to_lowercase();
    if ast.fragments.contains_key(&fragment_name) {
        return fragment_name;
    }

    let mut body = vec![];
    let mut fragment = ast.framework.stdlib.iter().find(|f| f.decl.id == "conv").unwrap().clone();
    fragment.decl.id = fragment_name.clone();

    let mut wire = ident("input").into();
    wire = data_into_ncwh(data_format, geo_rank, wire);

    body.push(assignment("nchw", wire));
    wire = invocation(
        "conv",
        &[ident("nchw").into(), ident("filter").into(), ident("bias").into()],
        &*fragment
            .decl
            .parameters
            .iter()
            .skip(3)
            .map(|f| (&*f.id, ident(&f.id)))
            .collect::<Vec<_>>(),
    );
    body.push(assignment("conv", wire));

    wire = data_from_ncwh(data_format, geo_rank, ident("conv").into());

    body.push(assignment("output", wire));
    fragment.body = Some(body);
    ast.fragments.insert(fragment_name.clone(), fragment);
    fragment_name
}

pub fn conv(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::cnn::conv::ConvUnary,
) -> TractResult<Option<Arc<RValue>>> {
    use tract_core::ops::cnn::PaddingSpec;
    let ci = op
        .pool_spec
        .data_format
        .shape(&ast.model.outlet_fact(node.inputs[0])?.shape.to_tvec())?
        .c()
        .to_usize()?;
    let co =
        op.pool_spec.data_format.shape(&node.outputs[0].fact.shape.to_tvec())?.c().to_usize()?;
    let mut wire = ast.mapping[&node.inputs[0]].clone();
    let mut kernel_shape = tvec!(co, ci / op.group);
    kernel_shape.extend(op.pool_spec.kernel_shape.iter().copied());
    let mut weights = op.kernel_as_group_o_ihw()?.into_tensor();
    weights.set_shape(&*kernel_shape)?;
    let weigths = ast.konst_variable(format!("{}_weigths", node.name), &weights.into_arc_tensor());
    wire = ast.force_assign(format!("{}_input", node.name), &wire);
    let conv_fragment = conv_fragment(ast, op.pool_spec.data_format, op.pool_spec.rank());
    let padding = match &op.pool_spec.padding {
        PaddingSpec::Explicit(bef, after, _) => array(
            &bef.iter()
                .zip(after.iter())
                .map(|(a, b)| tuple_2(numeric(a), numeric(b)))
                .collect::<Vec<_>>(),
        ),
        PaddingSpec::SameUpper => array(&[]),
        PaddingSpec::SameLower => bail!("Unsupported padding scheme"),
        PaddingSpec::Valid => array(
            (0..op.pool_spec.rank()).map(|_| tuple_2(numeric(0), numeric(0))).collect::<Vec<_>>(),
        ),
    };
    let mut inputs = tvec![wire, weigths];
    if let Some(bias) = op.bias.as_ref() {
        let bias = ast.konst(format!("{}_bias", node.name), bias);
        inputs.push(bias)
    }
    wire = invocation(
        &conv_fragment,
        &inputs,
        &[
            ("dilation", ints(&op.pool_spec.dilations())),
            ("stride", ints(&op.pool_spec.strides())),
            ("border", string("constant")),
            ("groups", numeric(op.group)),
            ("padding", padding),
        ],
    );
    wire = ast.force_assign(&node.name, &wire);
    Ok(Some(wire))
}

fn cnn_pool_fragment<'a>(
    ast: &'a mut IntoAst,
    data_format: DataFormat,
    geo_rank: usize,
    op_name: &str,
) -> String {
    if data_format == DataFormat::NCHW {
        return op_name.into();
    }
    let fragment_name = format!("tract_sum_pool_{:?}_{}D", data_format, geo_rank).to_lowercase();
    if ast.fragments.contains_key(&fragment_name) {
        return fragment_name;
    }

    let mut body = vec![];
    let mut fragment = ast.framework.stdlib.iter().find(|f| f.decl.id == op_name).unwrap().clone();
    fragment.decl.id = fragment_name.clone();

    let mut wire = ident("input").into();
    wire = data_into_ncwh(data_format, geo_rank, wire);

    body.push(assignment("nchw", wire));
    wire = invocation(
        op_name,
        &[ident("nchw").into()],
        &*fragment
            .decl
            .parameters
            .iter()
            .skip(1)
            .map(|f| (&*f.id, ident(&f.id)))
            .collect::<Vec<_>>(),
    );
    body.push(assignment("sum_pool", wire));

    wire = data_from_ncwh(data_format, geo_rank, ident("sum_pool").into());

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
    wire = ast.force_assign(format!("{}_input", node.name), &wire);
    let conv_fragment = cnn_pool_fragment(ast, pool_spec.data_format, pool_spec.rank(), op_name);
    let padding = match &pool_spec.padding {
        PaddingSpec::Explicit(bef, after, _) => array(
            &bef.iter()
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
    let mut size = tvec!(1, 1);
    size.extend(pool_spec.kernel_shape.iter().cloned());
    let mut strides = tvec!(1, 1);
    strides.extend(pool_spec.strides().iter().cloned());
    let mut dilations = tvec!(1, 1);
    dilations.extend(pool_spec.dilations().iter().cloned());
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
    wire = invocation(&conv_fragment, &[wire], &params);
    wire = ast.force_assign(&node.name, &wire);
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

pub fn axis_op(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::change_axes::AxisOp,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let invoke = match op {
        AxisOp::Rm(axis) => invocation("squeeze", &[wire], &[("axes", ints(&[*axis]))]),
        AxisOp::Add(axis) => invocation("unsqueeze", &[wire], &[("axes", ints(&[*axis]))]),
        AxisOp::Move(from, to) => {
            let rank = node.outputs[0].fact.rank();
            let mut perm: TVec<usize> = (0..rank).collect();
            if from < to {
                perm[*from..(to + 1)].rotate_left(1);
            } else {
                perm[*to..(from + 1)].rotate_right(1);
            }
            invocation("transpose", &[wire], &[("axes", ints(&*perm))])
        }
        AxisOp::Reshape(start, from, to) => invocation(
            "reshape",
            &[wire],
            &[
                (
                    "shape",
                    ints(&*to.iter().map(|d| d.to_usize().unwrap()).collect::<Vec<_>>()),
                ),
                ("axis_start", numeric(start)),
                ("axis_count", numeric(from.len())),
            ],
        ),
    };
    Ok(Some(invoke))
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
    Ok(Some(invocation(oper, &[wire], &[("axes", ints(&*op.axes))])))
}

pub fn matmul(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::matmul::MatMul,
) -> TractResult<Option<Arc<RValue>>> {
    let a = ast.force_assign(format!("{}_a", node.name), &ast.mapping[&node.inputs[0]].clone());
    let b = ast.force_assign(format!("{}_b", node.name), &ast.mapping[&node.inputs[1]].clone());
    let c = if op.c_trans {
        invocation(
            "matmul",
            &[b, a],
            &[("transposeA", logical(!op.b_trans)), ("transposeB", logical(!op.a_trans))],
        )
    } else {
        invocation(
            "matmul",
            &[a, b],
            &[("transposeA", logical(op.a_trans)), ("transposeB", logical(op.b_trans))],
        )
    };
    Ok(Some(ast.force_assign(&node.name, &c)))
}

pub fn matmul_unary(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::matmul::MatMulUnary,
) -> TractResult<Option<Arc<RValue>>> {
    let a = ast.konst(format!("{}_a", node.name), &op.a);
    let b = ast.force_assign(format!("{}_b", node.name), &ast.mapping[&node.inputs[0]].clone());
    let c = if op.c_trans {
        invocation(
            "matmul",
            &[b, a],
            &[("transposeA", logical(!op.b_trans)), ("transposeB", logical(!op.a_trans))],
        )
    } else {
        invocation(
            "matmul",
            &[a, b],
            &[("transposeA", logical(op.a_trans)), ("transposeB", logical(op.b_trans))],
        )
    };
    Ok(Some(ast.force_assign(&node.name, &c)))
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
