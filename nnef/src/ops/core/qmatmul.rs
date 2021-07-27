use std::str::FromStr;

use crate::deser::CoerceFrom;
use crate::deser::Value;
use crate::internal::*;
use crate::ser::*;
use tract_core::ops::matmul::mir_quant::QParamKind;
use tract_core::ops::matmul::mir_quant_unary::QMatMulUnary;
use tract_core::ops::matmul::MatMulQParams;
use tract_core::ops::matmul::QMatMul;
use Datum;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(
        TypeId::of::<tract_core::ops::matmul::mir_quant_unary::QMatMulUnary>(),
        qmatmul_unary_dump,
    );
    registry
        .register_dumper(TypeId::of::<tract_core::ops::matmul::mir_quant::QMatMul>(), qmatmul_dump);
    registry.register_primitive("tract_core_qmatmul", &qmatmul_parameters(), qmatmul_load);
}

fn qmatmul_parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("A"),
        TypeName::Scalar.tensor().named("B"),
        TypeName::Scalar.tensor().named("bias").default(0),
        TypeName::Logical.spec().named("transposeA"),
        TypeName::Logical.spec().named("transposeB"),
        TypeName::Logical.spec().named("transposeB"),
        TypeName::Integer.spec().named("a0"),
        TypeName::Scalar.spec().named("a_scale"),
        TypeName::Integer.spec().named("b0"),
        TypeName::Scalar.spec().named("b_scale"),
        TypeName::Integer.spec().named("c0"),
        TypeName::Scalar.spec().named("c_scale"),
        TypeName::String.spec().named("output_type"),
    ]
}

pub fn qparams_to_rvalues(
    params: &MatMulQParams,
    node_inputs: &[OutletId],
    ast_mapping: &HashMap<OutletId, Arc<RValue>>,
) -> TractResult<[Option<RValue>; 6]> {
    macro_rules! attr_to_rvalue {
        ($a:ident, $typ:ty) => {
            match &params.$a {
                QParamKind::Attr(t) => {
                    Some(numeric(t.cast_to_dt(<$typ>::datum_type())?.to_scalar::<$typ>()?))
                }
                QParamKind::FromInput(i) => Some((*ast_mapping[&node_inputs[*i]]).clone()),
                QParamKind::FromQType => None,
            }
        };
    }

    let a0 = attr_to_rvalue!(a0, i32);
    let a_scale = attr_to_rvalue!(a_scale, f32);
    let b0 = attr_to_rvalue!(b0, i32);
    let b_scale = attr_to_rvalue!(b_scale, f32);
    let c0 = attr_to_rvalue!(c0, i32);
    let c_scale = attr_to_rvalue!(c_scale, f32);

    Ok([a0, a_scale, b0, b_scale, c0, c_scale])
}

fn qmatmul_dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    if node.outputs[0].fact.datum_type.is_quantized() {
        return Ok(None);
    }
    let op = node.op_as::<QMatMul>().unwrap();
    let a = ast.mapping[&node.inputs[0]].clone();
    let b = ast.mapping[&node.inputs[1]].clone();
    let bias = ast.mapping[&node.inputs[2]].clone();

    let [a0, a_scale, b0, b_scale, c0, c_scale] =
        qparams_to_rvalues(&op.params, &node.inputs, &ast.mapping)?;
    let mut named_args = vec![
        ("A", (*a).clone()),
        ("B", (*b).clone()),
        ("bias", (*bias).clone()),
        ("transposeA", logical(op.a_trans)),
        ("transposeB", logical(op.b_trans)),
        ("transposeC", logical(op.c_trans)),
        ("output_type", string(format!("{:?}", op.output_type))),
    ];
    macro_rules! push {
        ($a: ident) => {
            if let Some($a) = $a {
                named_args.push((stringify!($a), $a))
            }
        };
    }
    push!(a0);
    push!(a_scale);
    push!(b0);
    push!(b_scale);
    push!(c0);
    push!(c_scale);
    Ok(Some(invocation("tract_core_qmatmul", &[], &named_args)))
}

fn qmatmul_unary_dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op_as::<QMatMulUnary>().unwrap();
    let a = ast.konst_variable(format!("{}.a", node.name), &op.a)?;
    let b = ast.mapping[&node.inputs[0]].clone();

    let [a0, a_scale, b0, b_scale, c0, c_scale] =
        qparams_to_rvalues(&op.params, &node.inputs, &ast.mapping)?;

    let mut named_args = vec![
        ("A", (*a).clone()),
        ("B", (*b).clone()),
        ("transposeA", logical(op.a_trans)),
        ("transposeB", logical(op.b_trans)),
        ("transposeC", logical(op.c_trans)),
        ("output_type", string(format!("{:?}", op.output_type))),
    ];
    macro_rules! push {
        ($a: ident) => {
            if let Some($a) = $a {
                named_args.push((stringify!($a), $a))
            }
        };
    }
    push!(a0);
    push!(a_scale);
    push!(b0);
    push!(b_scale);
    push!(c0);
    push!(c_scale);
    if let Some(bias) = &op.bias {
        named_args
            .push(("bias", (&*ast.konst_variable(format!("{}.bias", node.name), bias)?).clone()));
    }

    Ok(Some(invocation("tract_core_qmatmul", &[], &*named_args)))
}

pub fn values_to_qparams(
    a0: Option<Value>,
    a_scale: Option<Value>,
    b0: Option<Value>,
    b_scale: Option<Value>,
    c0: Option<Value>,
    c_scale: Option<Value>,
    inputs: &mut Vec<OutletId>,
    builder: &mut ModelBuilder,
) -> TractResult<MatMulQParams> {
    macro_rules! value_to_attr {
        ($a:ident, $typ:ty) => {
            if let Some($a) = $a {
                if let Ok(t) = Arc::<Tensor>::coerce(builder, &$a) {
                    QParamKind::Attr(
                        t.cast_to_dt(<$typ>::datum_type())?.into_owned().into_arc_tensor(),
                    )
                } else {
                    let outlet_id = OutletId::coerce(builder, &$a)?;
                    inputs.push(outlet_id);
                    QParamKind::FromInput(inputs.len() - 1)
                }
            } else {
                QParamKind::FromQType
            }
        };
    }
    let a0 = value_to_attr!(a0, i32);
    let a_scale = value_to_attr!(a_scale, f32);
    let b0 = value_to_attr!(b0, i32);
    let b_scale = value_to_attr!(b_scale, f32);
    let c0 = value_to_attr!(c0, i32);
    let c_scale = value_to_attr!(c_scale, f32);

    Ok(MatMulQParams { a0, a_scale, b0, b_scale, c0, c_scale })
}

fn qmatmul_load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let a: OutletId = invocation.named_arg_as(builder, "A")?;
    let b: OutletId = invocation.named_arg_as(builder, "B")?;
    let bias: OutletId = invocation.named_arg_as(builder, "bias")?;
    let a_trans: bool = invocation.named_arg_as(builder, "transposeA")?;
    let b_trans: bool = invocation.named_arg_as(builder, "transposeB")?;
    let c_trans: bool = invocation.named_arg_as(builder, "transposeC")?;
    let a0: Option<Value> = invocation.named_arg_as(builder, "a0").ok();
    let a_scale: Option<Value> = invocation.named_arg_as(builder, "a_scale").ok();
    let b0: Option<Value> = invocation.named_arg_as(builder, "b0").ok();
    let b_scale: Option<Value> = invocation.named_arg_as(builder, "b_scale").ok();
    let c0: Option<Value> = invocation.named_arg_as(builder, "c0").ok();
    let c_scale: Option<Value> = invocation.named_arg_as(builder, "c_scale").ok();
    let output_type =
        DatumType::from_str(&*invocation.named_arg_as::<String>(builder, "output_type")?)?;
    let mut inputs = vec![a, b, bias];
    let params = values_to_qparams(a0, a_scale, b0, b_scale, c0, c_scale, &mut inputs, builder)?;
    builder.wire(QMatMul { a_trans, b_trans, c_trans, output_type, params }, &inputs)
}
