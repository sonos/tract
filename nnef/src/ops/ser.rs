use crate::ast::*;
use crate::ser::*;
use std::any::TypeId;
use tract_core::internal::*;
use tract_core::ops;
use tract_core::ops::nn::DataFormat;

pub type OpDumper = fn(&mut IntoAst, node: &TypedNode) -> TractResult<Arc<RValue>>;

pub fn registry() -> HashMap<TypeId, OpDumper> {
    let mut registry: HashMap<TypeId, OpDumper> = Default::default();
    macro_rules! reg {
        ($op:ty, $path: path) => {
            registry.insert(TypeId::of::<$op>(), |ast, node| {
                $path(ast, node, node.op().downcast_ref::<$op>().unwrap())
            })
        };
    };
    reg!(ops::cnn::ConvUnary, conv);
    reg!(ops::matmul::MatMulUnary, matmul);
    reg!(ops::binary::UnaryOp, semi_binary);
    registry
}

fn conv_fragment<'a>(
    ast: &'a mut IntoAst,
    data_format: DataFormat,
    kernel_fmt: ops::cnn::KernelFormat,
    geo_rank: usize,
) -> String {
    use tract_core::ops::nn::DataFormat::*;
    if data_format == DataFormat::NHWC && kernel_fmt == ops::cnn::KernelFormat::OIHW {
        return "conv".into();
    }
    let fragment_name = format!("tract_conv_{:?}_{:?}_{}D", data_format, kernel_fmt, geo_rank).to_lowercase();
    if ast.fragments.contains_key(&fragment_name) {
        return fragment_name;
    }

    let mut body = vec![];
    let mut fragment =
        crate::ast::stdlib().iter().find(|f| f.decl.id == "conv").unwrap().as_ref().clone();
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
        let oihw = invoke("transpose", &[ident("filter").into()], &[named_arg("axes", int_array(&perm))]);
        body.push(Assignment { left: LValue::Identifier("oihw".into()), right: oihw.as_ref().clone() });
        ident("oihw").into()
    };

    let mut wire = RValue::Identifier("input".into()).into();
    if !data_format.has_n() {
        wire = invoke("unsqueeze", &[wire], &[named_arg("axes", int_array(&[0]))]);
    }
    if data_format == NHWC || data_format == HWC {
        let mut perm: TVec<usize> = (0..geo_rank + 2).collect();
        perm[1..].rotate_right(1);
        wire = invoke("transpose", &[wire], &[named_arg("axes", int_array(&perm))])
    }

    body.push(Assignment { left: LValue::Identifier("nchw".into()), right: wire.as_ref().clone() });
    wire = invoke(
        "conv",
        &[ident("nchw").into(), filter],
        &fragment
            .decl
            .parameters
            .iter()
            .skip(2)
            .map(|f| named_arg(&f.id, ident(&f.id)))
            .collect::<Vec<_>>(),
    );
    body.push(Assignment { left: lident("conv"), right: wire.as_ref().clone() });

    let mut wire = ident("conv").into();
    if data_format == NHWC || data_format == HWC {
        let mut perm: TVec<usize> = (0..geo_rank + 2).collect();
        perm[1..].rotate_left(1);
        wire = invoke("transpose", &[wire], &[named_arg("axes", int_array(&perm))])
    }
    if !data_format.has_n() {
        wire = invoke("squeeze", &[wire], &[named_arg("axes", int_array(&[0]))]);
    }

    body.push(Assignment { left: lident("output"), right: wire.as_ref().clone() });
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
        PaddingSpec::Explicit(bef, after, _) => RValue::Array(
            bef.iter()
                .zip(after.iter())
                .map(|(b, a)| RValue::Tuple(vec![numeric(b), numeric(a)]))
                .collect(),
        ),
        PaddingSpec::SameUpper => RValue::Array(vec![]),
        PaddingSpec::SameLower => bail!("Unsupported padding scheme"),
        PaddingSpec::Valid => RValue::Array(
            (0..op.pool_spec.rank()).map(|_| RValue::Tuple(vec![numeric(0), numeric(0)])).collect(),
        ),
    };
    wire = invoke(
        &conv_fragment,
        &[wire, weigths],
        &[
            named_arg("dilation", int_array(&op.pool_spec.dilations())),
            named_arg("stride", int_array(&op.pool_spec.strides())),
            named_arg("border", string("constant")),
            named_arg("groups", numeric(op.group)),
            named_arg("padding", padding),
        ],
    );
    wire = ast.force_assign(format!("{}_output", node.name), &wire);
    wire = ast.force_assign(&node.name, &wire);
    Ok(wire)
}

fn matmul(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::matmul::MatMulUnary,
) -> TractResult<Arc<RValue>> {
    let a = ast.konst(format!("{}_a", node.name), &op.a);
    let b = ast.force_assign(format!("{}_b", node.name), &ast.mapping[&node.inputs[0]].clone());
    let c = if op.c_trans {
        invoke(
            "matmul",
            &[b, a],
            &[
                named_arg("transposeA", RValue::Literal(Literal::Logical(!op.b_trans))),
                named_arg("transposeB", RValue::Literal(Literal::Logical(!op.a_trans))),
            ],
        )
    } else {
        invoke(
            "matmul",
            &[a, b],
            &[
                named_arg("transposeA", RValue::Literal(Literal::Logical(op.a_trans))),
                named_arg("transposeB", RValue::Literal(Literal::Logical(op.b_trans))),
            ],
        )
    };
    Ok(ast.force_assign(&node.name, &c))
}

fn semi_binary(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::binary::UnaryOp,
) -> TractResult<Arc<RValue>> {
    let a = ast.konst(format!("{}_a", node.name), &op.a);
    let b = ast.mapping[&node.inputs[0]].clone();
    Ok(invoke(bin_miniop(op.mini_op.as_ref())?, &[a, b], &[]))
}

fn bin_miniop(op: &dyn ops::binary::BinMiniOp) -> TractResult<&'static str> {
    macro_rules! mini {
        ($op: ty, $name: ident) => {
            if let Some(_) = op.downcast_ref::<$op>() {
                return Ok(stringify!($name));
            }
        };
    };
    mini!(ops::math::Add, add);
    mini!(ops::math::Sub, sub);
    mini!(ops::math::Mul, mul);
    mini!(ops::math::Div, div);
    mini!(ops::math::Pow, pow);

    mini!(ops::math::Max, max);
    mini!(ops::math::Min, min);
    bail!("Untranslated binary mini op: {:?}", op)
}
