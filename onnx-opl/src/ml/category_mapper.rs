use std::hash::Hash;
use tract_nnef::internal::*;
use tract_nnef::tract_core::itertools::Itertools;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_ml_category_mapper_to_int",
        &parameters_to_int(),
        load_to_int,
    );
    registry.register_primitive(
        "tract_onnx_ml_category_mapper_to_string",
        &parameters_to_string(),
        load_to_string,
    );
    registry.register_dumper(TypeId::of::<CategoryMapper<i64, String>>(), dump_to_string);
    registry.register_dumper(TypeId::of::<CategoryMapper<String, i64>>(), dump_to_int);
}

#[derive(Clone, Debug)]
pub struct CategoryMapper<Src: Datum + Hash + Eq, Dst: Datum + Hash> {
    pub hash: HashMap<Src, Dst>,
    pub default: Dst,
}

impl<Src: Datum + Hash + Eq + Ord, Dst: Datum + Hash> DynHash for CategoryMapper<Src, Dst> {
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        dyn_hash(self, hasher)
    }
}

impl<Src: Datum + Hash + Eq + Ord, Dst: Datum + Hash> Hash for CategoryMapper<Src, Dst> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.hash.iter().sorted_by_key(|s| s.0).for_each(|v| std::hash::Hash::hash(&v, state));
        std::hash::Hash::hash(&self.default, state)
    }
}

impl<Src: Datum + Hash + Eq + Ord, Dst: Datum + Hash> Op for CategoryMapper<Src, Dst> {
    fn name(&self) -> Cow<str> {
        format!("CategoryMapper<{:?},{:?}>", Src::datum_type(), Dst::datum_type()).into()
    }

    fn op_families(&self) -> &'static [&'static str] {
        &["onnx-ml"]
    }
    op_as_typed_op!();
}

impl<Src: Datum + Hash + Eq + Ord, Dst: Datum + Hash> EvalOp for CategoryMapper<Src, Dst> {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let input = input.to_array_view::<Src>()?;
        let output = input.map(|v| self.hash.get(v).unwrap_or(&self.default).clone());
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl<Src: Datum + Hash + Eq + Ord, Dst: Datum + Hash> TypedOp for CategoryMapper<Src, Dst> {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        assert!(
            (Src::datum_type() == String::datum_type() && Dst::datum_type() == i64::datum_type())
                || (Dst::datum_type() == String::datum_type()
                    && Src::datum_type() == i64::datum_type())
        );
        Ok(tvec!(TypedFact::dt_shape(Dst::datum_type(), inputs[0].shape.iter())))
    }

    as_op!();
}

fn parameters_to_int() -> Vec<Parameter> {
    vec![
        TypeName::String.tensor().named("input"),
        TypeName::String.tensor().named("keys"),
        TypeName::Scalar.tensor().named("values"),
        TypeName::Scalar.named("default").default(-1),
    ]
}

fn parameters_to_string() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Scalar.tensor().named("keys"),
        TypeName::String.tensor().named("values"),
        TypeName::String.named("default").default(""),
    ]
}

fn dump_to_string(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let to_string = node.op_as::<CategoryMapper<i64, String>>().context("wrong op")?;
    let (keys, values) =
        to_string.hash.iter().map(|(k, v)| (*k, v.clone())).unzip::<i64, String, Vec<_>, Vec<_>>();
    let keys = ast.konst_variable(format!("{}_keys", node.name), &rctensor1(&keys));
    let values = ast.konst_variable(format!("{}_values", node.name), &rctensor1(&values));
    Ok(Some(invocation(
        "tract_onnx_ml_category_mapper_to_string",
        &[input, keys, values],
        &[("default", string(to_string.default.clone()))],
    )))
}

fn dump_to_int(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let from_string = node.op_as::<CategoryMapper<String, i64>>().context("wrong op")?;
    let (keys, values) = from_string
        .hash
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .unzip::<String, i64, Vec<_>, Vec<_>>();
    let keys = ast.konst_variable(format!("{}_keys", node.name), &rctensor1(&keys));
    let values = ast.konst_variable(format!("{}_values", node.name), &rctensor1(&values));
    Ok(Some(invocation(
        "tract_onnx_ml_category_mapper_to_int",
        &[input, keys, values],
        &[("default", numeric(from_string.default))],
    )))
}

fn load_to_string(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input = invocation.named_arg_as(builder, "input")?;
    let default = invocation.named_arg_as(builder, "default")?;
    let keys: Arc<Tensor> = invocation.named_arg_as(builder, "keys")?;
    let values: Arc<Tensor> = invocation.named_arg_as(builder, "values")?;
    let hash = keys
        .as_slice::<i64>()?
        .iter()
        .copied()
        .zip(values.as_slice::<String>()?.iter().cloned())
        .collect();
    let op = CategoryMapper::<i64, String> { hash, default };
    builder.wire(op, &[input])
}

fn load_to_int(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input = invocation.named_arg_as(builder, "input")?;
    let default = invocation.named_arg_as(builder, "default")?;
    let keys: Arc<Tensor> = invocation.named_arg_as(builder, "keys")?;
    let values: Arc<Tensor> = invocation.named_arg_as(builder, "values")?;
    let hash = keys
        .as_slice::<String>()?
        .iter()
        .cloned()
        .zip(values.as_slice::<i64>()?.iter().copied())
        .collect();
    let op = CategoryMapper::<String, i64> { hash, default };
    builder.wire(op, &[input])
}
