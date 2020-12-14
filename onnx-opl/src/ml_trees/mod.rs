use std::collections::HashMap;
use std::iter;
use tract_nnef::internal::*;

mod tree;

use self::tree::{Aggregate, Cmp, Leaf, PostTransform, Tree, TreeEnsemble, TreeNode};

fn parse_node_mode(s: &str) -> TractResult<Option<Cmp>> {
    match s {
        "BRANCH_LEQ" => Ok(Some(Cmp::LessEqual)),
        "BRANCH_LT" => Ok(Some(Cmp::Less)),
        "BRANCH_GTE" => Ok(Some(Cmp::GreaterEqual)),
        "BRANCH_GT" => Ok(Some(Cmp::Greater)),
        "BRANCH_EQ" => Ok(Some(Cmp::Equal)),
        "BRANCH_NEQ" => Ok(Some(Cmp::NotEqual)),
        "LEAF" => Ok(None),
        _ => bail!("Unsupported mode node: {}", s),
    }
}

fn parse_post_transform(s: &str) -> TractResult<Option<PostTransform>> {
    match s {
        "NONE" => Ok(None),
        "SOFTMAX" => Ok(Some(PostTransform::Softmax)),
        "LOGISTIC" => Ok(Some(PostTransform::Logistic)),
        "SOFTMAX_ZERO" => Ok(Some(PostTransform::SoftmaxZero)),
        "PROBIT" => bail!("PROBIT unsupported"),
        _ => bail!("Invalid post transform: {}", s),
    }
}

fn parse_aggregate(s: &str) -> TractResult<Aggregate> {
    match s {
        "SUM" => Ok(Aggregate::Sum),
        "AVERAGE" => Ok(Aggregate::Avg),
        "MAX" => Ok(Aggregate::Max),
        "MIN" => Ok(Aggregate::Min),
        _ => bail!("Invalid aggregate function: {}", s),
    }
}

pub struct NodesData {
    pub n_classes: usize,
    pub base_values: Option<Vec<f32>>,
    pub classlabels: Option<Tensor>,
    pub node_ids: Vec<usize>,
    pub tree_ids: Vec<usize>,
    pub feature_ids: Vec<usize>,
    pub true_ids: Vec<usize>,
    pub false_ids: Vec<usize>,
    pub node_values: Vec<f32>,
    pub nan_is_true: Vec<bool>,
    pub node_modes: Vec<String>,
    pub post_transform: Option<String>,
    pub aggregate_fn: String,
    pub leaf_node_ids: Vec<usize>,
    pub leaf_tree_ids: Vec<usize>,
    pub leaf_class_ids: Vec<usize>,
    pub leaf_weights: Vec<f32>,
}

impl NodesData {
    pub fn build_tree_ensemble(&self) -> TractResult<TreeEnsemble> {
        // parse node data from protobuf
        let n_nodes = self.tree_ids.len();

        // build the leaf map and collect all leaves
        let n_trees = *self.tree_ids.last().unwrap() + 1;
        let n_leaves = self.leaf_class_ids.len();
        let mut leaf_map: HashMap<(usize, usize), (usize, usize)> = HashMap::default();
        let mut leaves: Vec<Vec<Leaf>> = iter::repeat_with(Vec::default).take(n_trees).collect();

        for i in 0..n_leaves {
            let leaf_tree_id = self.leaf_tree_ids[i];
            let leaf_node_id = self.leaf_node_ids[i];
            let leaf_class_id = self.leaf_class_ids[i];
            let leaf_weight = self.leaf_weights[i];

            let tree_leaves = &mut leaves[leaf_tree_id];
            leaf_map
                .entry((leaf_tree_id, leaf_node_id))
                .or_insert_with(|| (tree_leaves.len(), tree_leaves.len()))
                .1 += 1;
            tree_leaves.push(Leaf::new(leaf_class_id, leaf_weight));
        }

        // collect all nodes

        let mut nodes: Vec<Vec<TreeNode>> = vec![vec!(); n_trees];

        let mut prev_tree_id = -1i64 as usize;
        let mut prev_node_id = 0;

        for i in 0..n_nodes {
            let tree_id = self.tree_ids[i];
            let node_id = self.node_ids[i];
            let node_mode = &self.node_modes[i];

            if tree_id != prev_tree_id {
                tract_core::anyhow::ensure!(
                    node_id == 0,
                    "node ids must start from 0 for each tree"
                )
            } else {
                tract_core::anyhow::ensure!(
                    node_id == prev_node_id + 1,
                    "node ids must increase by 1"
                )
            }
            let tree_node = if node_mode != "" {
                let feature_id = self.feature_ids[i];
                let value = self.node_values[i];
                let true_id = self.true_ids[i];
                let false_id = self.false_ids[i];
                let nan_is_true = self.nan_is_true[i];
                TreeNode::new_branch(
                    parse_node_mode(node_mode)?.unwrap(),
                    feature_id,
                    value,
                    true_id,
                    false_id,
                    nan_is_true,
                )
            } else {
                if let Some(&(start, end)) = leaf_map.get(&(tree_id, node_id)) {
                    TreeNode::new_leaf(start, end)
                } else {
                    bail!("leaf not found: tree_id={}, node_id={}", tree_id, node_id)
                }
            };
            nodes[tree_id].push(tree_node);

            prev_tree_id = tree_id;
            prev_node_id = node_id;
        }

        // build the trees

        let mut trees: Vec<Tree> = Vec::default();

        for i in 0..n_trees {
            let tree = Tree::build(self.n_classes, &nodes[i], &leaves[i])
                .with_context(|| format!("Building tree {}", i))?;
            trees.push(tree);
        }

        // build the tree ensemble

        let base_scores = self.base_values.as_ref().map(Vec::as_slice);
        let post_transform =
            self.post_transform.as_deref().map(parse_post_transform).transpose()?.unwrap_or(None);
        let aggregate_fn = parse_aggregate(&self.aggregate_fn)?;

        let ensemble = TreeEnsemble::build(&trees, aggregate_fn, post_transform, base_scores)?;
        Ok(ensemble)
    }
}

#[derive(Debug, Clone, Hash)]
pub struct TreeEnsembleClassifier {
    pub ensemble: TreeEnsemble,
    pub class_labels: Tensor,
}

impl_dyn_hash!(TreeEnsembleClassifier);

impl Op for TreeEnsembleClassifier {
    fn name(&self) -> Cow<str> {
        "TreeEnsembleClassifier".into()
    }

    fn op_families(&self) -> &'static [&'static str] {
        &["onnx-ml"]
    }

    op_as_typed_op!();
}

impl EvalOp for TreeEnsembleClassifier {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let input = input.cast_to::<f32>()?;
        let input = input.to_array_view::<f32>()?;
        let scores = self.ensemble.eval(input)?;
        let tops: Vec<usize> = scores
            .view()
            .into_dimensionality::<tract_ndarray::Ix2>()?
            .outer_iter()
            .map(|scores| {
                scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| (a.1).partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less))
                    .unwrap()
                    .0 as usize
            })
            .collect();
        let categ = tops
            .iter()
            .map(|&ix| self.class_labels.slice(0, ix, ix + 1))
            .collect::<TractResult<Vec<Tensor>>>()?;
        let categ = Tensor::stack_tensors(0, &categ)?;
        Ok(tvec!(categ.into_arc_tensor(), scores.into_arc_tensor()))
    }
}

impl TypedOp for TreeEnsembleClassifier {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let n = &inputs[0].shape[0];
        Ok(tvec!(
            TypedFact::dt_shape(self.class_labels.datum_type(), [n.clone()].as_ref()),
            TypedFact::dt_shape(
                f32::datum_type(),
                [n.clone(), self.class_labels.len().to_dim()].as_ref()
            )
        ))
    }

    as_op!();
}
