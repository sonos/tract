use crate::ast::*;
use crate::ser::*;
use std::any::TypeId;
use tract_core::internal::*;
use tract_core::ops;

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

fn conv(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ops::cnn::conv::ConvUnary,
) -> TractResult<Arc<RValue>> {
    use tract_core::ops::cnn::PaddingSpec;
    use tract_core::ops::nn::DataFormat::*;
    let mut wire = ast.mapping[&node.inputs[0]].clone();
    if !op.pool_spec.data_format.has_n() {
        wire = invoke("unsqueeze", &[wire], &[named_arg("axis", numeric(0))]);
    }
    if op.pool_spec.data_format == NHWC || op.pool_spec.data_format == HWC {
        let mut perm: TVec<usize> = (0..op.pool_spec.rank() + 2).collect();
        perm[1..].rotate_right(1);
        wire = invoke("transpose", &[wire], &[named_arg("axes", shape(&perm))])
    }
    let weigths = ast.konst(format!("{}_weigths", node.name), &op.kernel);
    wire = ast.force_assign(format!("{}_input", node.name), &wire);
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
        "conv",
        &[wire, weigths],
        &[
            named_arg("dilation", shape(&op.pool_spec.dilations())),
            named_arg("stride", shape(&op.pool_spec.strides())),
            named_arg("border", string("constant")),
            named_arg("groups", numeric(op.group)),
            named_arg("padding", padding),
        ],
    );
    wire = ast.force_assign(format!("{}_output", node.name), &wire);
    if op.pool_spec.data_format == NHWC || op.pool_spec.data_format == HWC {
        let mut perm: TVec<usize> = (0..op.pool_spec.rank() + 2).collect();
        perm[1..].rotate_left(1);
        wire = invoke("transpose", &[wire], &[named_arg("axes", shape(&perm))])
    }
    if !op.pool_spec.data_format.has_n() {
        wire = invoke("squeeze", &[wire], &[named_arg("axis", numeric(0))]);
    }
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
            &[a, b],
            &[
                named_arg("transposeA", RValue::Literal(Literal::Logical(op.a_trans))),
                named_arg("transposeB", RValue::Literal(Literal::Logical(op.b_trans))),
            ],
        )
    } else {
        invoke(
            "matmul",
            &[b, a],
            &[
                named_arg("transposeA", RValue::Literal(Literal::Logical(!op.b_trans))),
                named_arg("transposeB", RValue::Literal(Literal::Logical(!op.a_trans))),
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
