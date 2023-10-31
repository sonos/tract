pub use super::tree::{Aggregate, Cmp, TreeEnsemble, TreeEnsembleData};
use tract_nnef::internal::*;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_ml_tree_ensemble_classifier",
        &parameters(),
        &[("output", TypeName::Scalar.tensor())],
        load,
    );
    registry.register_dumper(dump);
}

pub fn parse_aggregate(s: &str) -> TractResult<Aggregate> {
    match s {
        "SUM" => Ok(Aggregate::Sum),
        "AVERAGE" => Ok(Aggregate::Avg),
        "MAX" => Ok(Aggregate::Max),
        "MIN" => Ok(Aggregate::Min),
        _ => bail!("Invalid aggregate function: {}", s),
    }
}

#[derive(Debug, Clone, Hash)]
pub struct TreeEnsembleClassifier {
    pub ensemble: TreeEnsemble,
}

impl Op for TreeEnsembleClassifier {
    fn name(&self) -> Cow<str> {
        "TreeEnsembleClassifier".into()
    }

    op_as_typed_op!();
}

impl EvalOp for TreeEnsembleClassifier {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let input = input.cast_to::<f32>()?;
        let input = input.to_array_view::<f32>()?;
        let scores = self.ensemble.eval(input)?;
        Ok(tvec!(scores.into_tvalue()))
    }
}

impl TypedOp for TreeEnsembleClassifier {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let n = &inputs[0].shape[0];
        Ok(tvec!(f32::fact(&[n.clone(), self.ensemble.n_classes().into()])))
    }

    as_op!();
}

fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Scalar.tensor().named("trees"),
        TypeName::Scalar.tensor().named("nodes"),
        TypeName::Scalar.tensor().named("leaves"),
        TypeName::Integer.named("max_used_feature"),
        TypeName::Integer.named("n_classes"),
        TypeName::String.named("aggregate_fn"),
    ]
}

fn dump(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &TreeEnsembleClassifier,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let trees = ast.konst_variable(format!("{}_trees", node.name), &op.ensemble.data.trees)?;
    let nodes = ast.konst_variable(format!("{}_nodes", node.name), &op.ensemble.data.nodes)?;
    let leaves = ast.konst_variable(format!("{}_leaves", node.name), &op.ensemble.data.leaves)?;
    let agg = match op.ensemble.aggregate_fn {
        Aggregate::Min => "MIN",
        Aggregate::Max => "MAX",
        Aggregate::Sum => "SUM",
        Aggregate::Avg => "AVERAGE",
    };
    Ok(Some(invocation(
        "tract_onnx_ml_tree_ensemble_classifier",
        &[input, trees, nodes, leaves],
        &[
            ("max_used_feature", numeric(op.ensemble.max_used_feature)),
            ("n_classes", numeric(op.ensemble.n_classes)),
            ("aggregate_fn", string(agg)),
        ],
    )))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let trees = invocation.named_arg_as(builder, "trees")?;
    let nodes = invocation.named_arg_as(builder, "nodes")?;
    let leaves = invocation.named_arg_as(builder, "leaves")?;
    let max_used_feature = invocation.named_arg_as(builder, "max_used_feature")?;
    let n_classes = invocation.named_arg_as(builder, "n_classes")?;
    let aggregate_fn: String = invocation.named_arg_as(builder, "aggregate_fn")?;
    let aggregate_fn = parse_aggregate(&aggregate_fn)?;
    let data = TreeEnsembleData { trees, nodes, leaves };
    let ensemble = TreeEnsemble { data, n_classes, max_used_feature, aggregate_fn };
    let op = TreeEnsembleClassifier { ensemble };
    builder.wire(op, &[input])
}
