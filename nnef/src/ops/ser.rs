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
    reg!(ops::binary::MergeOp, binary);
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
    let fragment_name =
        format!("tract_conv_{:?}_{:?}_{}D", data_format, kernel_fmt, geo_rank).to_lowercase();
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
        let oihw = invocation("transpose", &[ident("filter").into()], &[("axes", ints(&perm))]);
        body.push(assignment("oihw", oihw));
        ident("oihw").into()
    };

    let mut wire = ident("input").into();
    if !data_format.has_n() {
        wire = invocation("unsqueeze", &[wire], &[("axes", ints(&[0]))]);
    }
    if data_format == NHWC || data_format == HWC {
        let mut perm: TVec<usize> = (0..geo_rank + 2).collect();
        perm[1..].rotate_right(1);
        wire = invocation("transpose", &[wire], &[("axes", ints(&perm))])
    }

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

    let mut wire = ident("conv").into();
    if data_format == NHWC || data_format == HWC {
        let mut perm: TVec<usize> = (0..geo_rank + 2).collect();
        perm[1..].rotate_left(1);
        wire = invocation("transpose", &[wire], &[("axes", ints(&perm))])
    }
    if !data_format.has_n() {
        wire = invocation("squeeze", &[wire], &[("axes", ints(&[0]))]);
    }

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
    op: &ops::binary::MergeOp,
) -> TractResult<Arc<RValue>> {
    let a = ast.mapping[&node.inputs[0]].clone();
    let b = ast.mapping[&node.inputs[1]].clone();
    Ok(invocation(bin_miniop(op.0.as_ref())?, &[a, b], &[]))
}

fn semi_binary(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::binary::UnaryOp,
) -> TractResult<Arc<RValue>> {
    let a = ast.konst(format!("{}_a", node.name), &op.a);
    let b = ast.mapping[&node.inputs[0]].clone();
    Ok(invocation(bin_miniop(op.mini_op.as_ref())?, &[a, b], &[]))
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
