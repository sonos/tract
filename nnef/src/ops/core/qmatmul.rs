use std::str::FromStr;

use crate::deser::CoerceFrom;
use crate::deser::Value;
use crate::internal::*;
use crate::ser::*;
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
) -> TractResult<[RValue; 6]> {
    macro_rules! attr_to_rvalue {
        ($a:ident, $typ:ty) => {
            match &params.$a {
                AttrOrInput::Attr(t) => {
                    numeric(t.cast_to_dt(<$typ>::datum_type())?.to_scalar::<$typ>()?)
                }
                AttrOrInput::Input(i) => (*ast_mapping[&node_inputs[*i]]).clone(),
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

    Ok(Some(invocation(
        "tract_core_qmatmul",
        &[],
        &[
            ("A", (*a).clone()),
            ("B", (*b).clone()),
            ("bias", (*bias).clone()),
            ("transposeA", logical(op.a_trans)),
            ("transposeB", logical(op.b_trans)),
            ("transposeC", logical(op.c_trans)),
            ("a0", a0),
            ("a_scale", a_scale),
            ("b0", b0),
            ("b_scale", b_scale),
            ("c0", c0),
            ("c_scale", c_scale),
            ("output_type", string(format!("{:?}", op.output_type))),
        ],
    )))
}

fn qmatmul_unary_dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op_as::<QMatMulUnary>().unwrap();
    let a = ast.konst_variable(format!("{}.a", node.name), &op.a)?;
    let b = ast.mapping[&node.inputs[0]].clone();

    let [a0, a_scale, b0, b_scale, c0, c_scale] =
        qparams_to_rvalues(&op.params, &node.inputs, &ast.mapping)?;

    let mut args = vec![
        ("A", (*a).clone()),
        ("B", (*b).clone()),
        ("transposeA", logical(op.a_trans)),
        ("transposeB", logical(op.b_trans)),
        ("transposeC", logical(op.c_trans)),
        ("a0", a0),
        ("a_scale", a_scale),
        ("b0", b0),
        ("b_scale", b_scale),
        ("c0", c0),
        ("c_scale", c_scale),
        ("output_type", string(format!("{:?}", op.output_type))),
    ];

    if let Some(bias) = &op.bias {
        args.push(("bias", (&*ast.konst_variable(format!("{}.bias", node.name), bias)?).clone()));
    }

    Ok(Some(invocation("tract_core_qmatmul", &[], &*args)))
}

pub fn values_to_qparams(
    a0: Value,
    a_scale: Value,
    b0: Value,
    b_scale: Value,
    c0: Value,
    c_scale: Value,
    inputs: &mut Vec<OutletId>,
    builder: &mut ModelBuilder,
) -> TractResult<MatMulQParams> {
    macro_rules! value_to_attr {
        ($a:ident, $typ:ty) => {
            if let Ok(t) = Arc::<Tensor>::coerce(builder, &$a) {
                AttrOrInput::Attr(
                    t.cast_to_dt(<$typ>::datum_type())?.into_owned().into_arc_tensor(),
                )
            } else {
                let outlet_id = OutletId::coerce(builder, &$a)?;
                inputs.push(outlet_id);
                AttrOrInput::Input(inputs.len() - 1)
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
    let a0: Value = invocation.named_arg_as(builder, "a0")?;
    let a_scale: Value = invocation.named_arg_as(builder, "a_scale")?;
    let b0: Value = invocation.named_arg_as(builder, "b0")?;
    let b_scale: Value = invocation.named_arg_as(builder, "b_scale")?;
    let c0: Value = invocation.named_arg_as(builder, "c0")?;
    let c_scale: Value = invocation.named_arg_as(builder, "c_scale")?;
    let output_type =
        DatumType::from_str(&*invocation.named_arg_as::<String>(builder, "output_type")?)?;
    let mut inputs = vec![a, b, bias];
    let params = values_to_qparams(a0, a_scale, b0, b_scale, c0, c_scale, &mut inputs, builder)?;
    builder.wire(QMatMul { a_trans, b_trans, c_trans, output_type, params }, &inputs)
}
