use crate::internal::*;
use crate::ser::*;
use std::any::TypeId;
use tract_core::ops;
use tract_core::ops::nn::DataFormat;

pub fn register(registry: &mut Registry) {
    macro_rules! nnef {
        ($op:ty, $path: path) => {
            registry.register_dumper(TypeId::of::<$op>(), |ast, node| {
                $path(ast, node, node.op().downcast_ref::<$op>().unwrap())
            })
        };
    };
    nnef!(ops::array::TypedConcat, concat);
    nnef!(ops::array::Slice<TDim>, slice<TDim>);
    nnef!(ops::array::Slice<usize>, slice<usize>);
    nnef!(ops::binary::UnaryOp, semi_binary);
    nnef!(ops::binary::TypedBinOp, binary);
    nnef!(ops::change_axes::AxisOp, axis_op);
    nnef!(ops::cnn::ConvUnary, conv);
    nnef!(ops::cnn::MaxPool, max_pool);
    nnef!(ops::cnn::SumPool, sum_pool);
    nnef!(ops::nn::Reduce, reduce);
    nnef!(ops::matmul::MatMulUnary, matmul);

    registry.register_element_wise("exp", &ops::math::Exp {});
    registry.register_element_wise("ln", &ops::math::Ln {});
    registry.register_element_wise("sin", &ops::math::Sin {});
    registry.register_element_wise("cos", &ops::math::Cos {});
    registry.register_element_wise("abs", &ops::math::Abs {});
    registry.register_element_wise("neg", &ops::math::Neg {});
    registry.register_element_wise("sign", &ops::math::Sign {});
    registry.register_element_wise("recip", &ops::math::Recip {});

    registry.register_element_wise("floor", &ops::math::Floor {});
    registry.register_element_wise("ceil", &ops::math::Ceil {});
    registry.register_element_wise("round", &ops::math::Round {});

    registry.register_element_wise("square", &ops::math::Square {});
    registry.register_element_wise("sqrt", &ops::math::Sqrt {});
    registry.register_element_wise("rsqrt", &ops::math::Rsqrt {});

    registry.register_element_wise("tanh", &ops::math::Tanh {});
    registry.register_element_wise("sigmoid", &ops::nn::Sigmoid {});

    registry.register_element_wise("not", &ops::logic::Not {});

}

fn concat(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::array::TypedConcat,
) -> TractResult<Arc<RValue>> {
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
    Ok(invocation("concat", &[array(&wires).into()], &[("axis", numeric(op.axis))]))
}

fn slice<D: DimLike>(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::array::Slice<D>,
) -> TractResult<Arc<RValue>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let start = op.start.to_integer()? as usize;
    let end = op.end.to_integer()? as usize;
    Ok(invocation(
        "slice",
        &[wire],
        &[("axes", ints(&[op.axis])), ("begin", ints(&[start])), ("end", ints(&[end]))],
    ))
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

fn conv_fragment<'a>(
    ast: &'a mut IntoAst,
    data_format: DataFormat,
    kernel_fmt: ops::cnn::KernelFormat,
    geo_rank: usize,
) -> String {
    if data_format == DataFormat::NCHW && kernel_fmt == ops::cnn::KernelFormat::OIHW {
        return "conv".into();
    }
    let fragment_name =
        format!("tract_conv_{:?}_{:?}_{}D", data_format, kernel_fmt, geo_rank).to_lowercase();
    if ast.fragments.contains_key(&fragment_name) {
        return fragment_name;
    }

    let mut body = vec![];
    let mut fragment = crate::ops::stdlib().iter().find(|f| f.decl.id == "conv").unwrap().clone();
    fragment.decl.id = fragment_name.clone();

    let filter = if kernel_fmt == ops::cnn::KernelFormat::OIHW {
        let mut perm: TVec<usize> = (0..geo_rank + 2).collect();
        perm[1..].rotate_right(1);
        ident("filter").into()
    } else {
        // ops::cnn::KernelFormat::HWIO
        let mut perm: TVec<usize> = (0..geo_rank + 2).collect();
        perm.rotate_right(1);
        perm[1..].rotate_right(1);
        let oihw = invocation("transpose", &[ident("filter").into()], &[("axes", ints(&perm))]);
        body.push(assignment("oihw", oihw));
        ident("oihw").into()
    };
    let mut wire = ident("input").into();
    wire = data_into_ncwh(data_format, geo_rank, wire);

    body.push(assignment("nchw", wire));
    wire = invocation(
        "conv",
        &[ident("nchw").into(), filter, ident("bias").into()],
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

fn conv(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::cnn::conv::ConvUnary,
) -> TractResult<Arc<RValue>> {
    use tract_core::ops::cnn::PaddingSpec;
    let mut wire = ast.mapping[&node.inputs[0]].clone();
    let weigths = ast.konst(format!("{}_weigths", node.name), &op.kernel);
    wire = ast.force_assign(format!("{}_input", node.name), &wire);
    let conv_fragment =
        conv_fragment(ast, op.pool_spec.data_format, op.kernel_fmt, op.pool_spec.rank());
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
    wire = invocation(
        &conv_fragment,
        &[wire, weigths],
        &[
            ("dilation", ints(&op.pool_spec.dilations())),
            ("stride", ints(&op.pool_spec.strides())),
            ("border", string("constant")),
            ("groups", numeric(op.group)),
            ("padding", padding),
        ],
    );
    wire = ast.force_assign(format!("{}_output", node.name), &wire);
    wire = ast.force_assign(&node.name, &wire);
    Ok(wire)
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
    let mut fragment = crate::ops::stdlib().iter().find(|f| f.decl.id == op_name).unwrap().clone();
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
) -> TractResult<Arc<RValue>> {
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
    wire = ast.force_assign(format!("{}_output", node.name), &wire);
    wire = ast.force_assign(&node.name, &wire);
    Ok(wire)
}

fn max_pool(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::cnn::MaxPool,
) -> TractResult<Arc<RValue>> {
    cnn_pool(ast, node, "max_pool", &op.pool_spec, None)
}

fn sum_pool(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::cnn::SumPool,
) -> TractResult<Arc<RValue>> {
    cnn_pool(ast, node, "box", &op.pool_spec, Some(("normalize", logical(op.normalize))))
}

fn axis_op(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::change_axes::AxisOp,
) -> TractResult<Arc<RValue>> {
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
                    ints(&*to.iter().map(|d| d.to_integer().unwrap() as usize).collect::<Vec<_>>()),
                ),
                ("axis_start", numeric(start)),
                ("axis_count", numeric(from.len())),
            ],
        ),
    };
    Ok(invoke)
}

fn reduce(ast: &mut IntoAst, node: &TypedNode, op: &ops::nn::Reduce) -> TractResult<Arc<RValue>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let oper = match op.reducer {
        ops::nn::Reducer::Sum => "sum_reduce",
        ops::nn::Reducer::Max => "max_reduce",
        ops::nn::Reducer::Min => "min_reduce",
        _ => todo!(),
    };
    Ok(invocation(oper, &[wire], &[("axes", ints(&*op.axes))]))
}

fn matmul(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::matmul::MatMulUnary,
) -> TractResult<Arc<RValue>> {
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
    Ok(ast.force_assign(&node.name, &c))
}

fn binary(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::binary::TypedBinOp,
) -> TractResult<Arc<RValue>> {
    let a = ast.mapping[&node.inputs[0]].clone();
    let b = ast.mapping[&node.inputs[1]].clone();
    let op = bin_miniop(ast, op.0.as_ref())?;
    Ok(invocation(&*op, &[a, b], &[]))
}

fn semi_binary(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::binary::UnaryOp,
) -> TractResult<Arc<RValue>> {
    let a = ast.konst(format!("{}_a", node.name), &op.a);
    let b = ast.mapping[&node.inputs[0]].clone();
    let op = bin_miniop(ast, op.mini_op.as_ref())?;
    Ok(invocation(&*op, &[a, b], &[]))
}

macro_rules! mini {
    ($op: expr, $typ: ty, $reg: ident :: $name:ident) => {
        if let Some(_) = $op.downcast_ref::<$typ>() {
            return Ok((stringify!($reg), stringify!($name)));
        }
    };
}

fn register_reg_mini<'op>(ast: &mut IntoAst, reg: &str, op: &'op str) -> Cow<'op, str> {
    if reg == "tract_nnef" {
        Cow::Borrowed(op)
    } else {
        if !ast.registries.iter().any(|r| r == reg) {
            ast.registries.push(reg.to_string())
        }
        Cow::Owned(format!("{}_{}", reg, op).into())
    }
}

fn bin_miniop(
    ast: &mut IntoAst,
    mini_op: &dyn ops::binary::BinMiniOp,
) -> TractResult<Cow<'static, str>> {
    fn op(op: &dyn ops::binary::BinMiniOp) -> TractResult<(&'static str, &'static str)> {
        mini!(op, ops::math::Add, tract_nnef::add);
        mini!(op, ops::math::Sub, tract_nnef::sub);
        mini!(op, ops::math::Mul, tract_nnef::mul);
        mini!(op, ops::math::Div, tract_nnef::div);
        mini!(op, ops::math::Pow, tract_nnef::pow);

        mini!(op, ops::logic::Lesser, tract_nnef::lt);
        mini!(op, ops::logic::Greater, tract_nnef::gt);
        mini!(op, ops::logic::LesserEqual, tract_nnef::le);
        mini!(op, ops::logic::GreaterEqual, tract_nnef::ge);
        mini!(op, ops::logic::Equals, tract_nnef::eq);
        mini!(op, ops::logic::NotEquals, tract_nnef::ne);

        mini!(op, ops::logic::And, tract_nnef::and);
        mini!(op, ops::logic::Or, tract_nnef::or);

        mini!(op, ops::math::Max, tract_nnef::max);
        mini!(op, ops::math::Min, tract_nnef::min);

        bail!("Untranslated binary mini op: {:?}", op)
    }
    let (reg, op) = op(mini_op)?;
    Ok(register_reg_mini(ast, reg, op))
}
