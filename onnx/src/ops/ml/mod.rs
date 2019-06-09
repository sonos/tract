use std::collections::HashMap;
use std::iter;

use crate::model::{ OnnxOpRegister, ParsingContext };
use crate::pb::NodeProto;
use crate::pb_helpers::{AttrTVecType, OptionExt, TryCollect};

use tract_core::internal::*;

mod tree;

use self::tree::{Aggregate, Cmp, Leaf, Node, PostTransform, Tree, TreeEnsemble};

fn get_vec_attr<'a, T>(node: &'a NodeProto, attr: &str, n: usize) -> TractResult<Vec<T>>
where
    T: AttrTVecType<'a>,
{
    let vec = node.get_attr_vec(attr)?;
    node.expect_attr(attr, vec.len() == n, || format!("length {}, got {}", vec.len(), n))?;
    Ok(vec)
}

fn get_vec_attr_opt<'a, T>(node: &'a NodeProto, attr: &str, n: usize) -> TractResult<Option<Vec<T>>>
where
    T: AttrTVecType<'a>,
{
    match node.get_attr_opt_vec(attr)? {
        Some(vec) => {
            node.expect_attr(attr, vec.len() == n, || {
                format!("length {} (or undefined), got {}", vec.len(), n)
            })?;
            Ok(Some(vec))
        }
        None => Ok(None),
    }
}

fn parse_node_mode(s: &str) -> Option<Option<Cmp>> {
    match s {
        "BRANCH_LEQ" => Some(Some(Cmp::LessEqual)),
        "BRANCH_LT" => Some(Some(Cmp::Less)),
        "BRANCH_GTE" => Some(Some(Cmp::GreaterEqual)),
        "BRANCH_GT" => Some(Some(Cmp::Greater)),
        "BRANCH_EQ" => Some(Some(Cmp::Equal)),
        "BRANCH_NEQ" => Some(Some(Cmp::NotEqual)),
        "BRANCH_LEAF" => Some(None),
        _ => None,
    }
}

fn parse_post_transform(s: &str) -> Option<Option<PostTransform>> {
    match s {
        "NONE" => Some(None),
        "SOFTMAX" => Some(Some(PostTransform::Softmax)),
        "LOGISTIC" => Some(Some(PostTransform::Logistic)),
        "SOFTMAX_ZERO" => Some(Some(PostTransform::SoftmaxZero)),
        "PROBIT" => None, // we don't support it for now
        _ => None,
    }
}

fn parse_aggregate(s: &str) -> Option<Aggregate> {
    match s {
        "SUM" => Some(Aggregate::Sum),
        "AVERAGE" => Some(Aggregate::Avg),
        "MAX" => Some(Aggregate::Max),
        "MIN" => Some(Aggregate::Min),
        _ => None,
    }
}

fn parse_tree_ensemble(node: &NodeProto, is_classifier: bool) -> TractResult<TreeEnsemble> {
    // parse n_classes from protobuf

    let (n_classes, attr) = if is_classifier {
        let n_ints = node.get_attr_opt_slice::<i64>("classlabels_int64s")?.map(|l| l.len());
        let n_strs = node.get_attr_opt_tvec::<&str>("classlabels_strings")?.map(|l| l.len());
        match (n_ints, n_strs) {
            (Some(n), None) => (n, "classlabels_ints64s"),
            (None, Some(n)) => (n, "classlabels_strings"),
            (None, None) => {
                node.bail("cannot find neither 'classlabels_int64s' not 'classlabels_strings'")?
            }
            (Some(_), Some(_)) => {
                node.bail("only one of 'classlabels_int64s' and 'classlabels_strings' can be set")?
            }
        }
    } else {
        (node.get_attr("n_targets")?, "n_targets")
    };
    node.expect_attr(attr, n_classes != 0, "at least one class/target")?;

    // parse base_values from protobuf

    let base_values = get_vec_attr_opt::<f32>(node, "base_values", n_classes)?;

    // parse post_transform from protobuf

    let post_transform = node
        .get_attr_opt("post_transform")?
        .and_try(|s| node.check_value("post_transform", parse_post_transform(s).ok_or(s)))?
        .unwrap_or(None);

    // parse aggregate_fn from protobuf (for regressors)

    let aggregate_fn = if is_classifier {
        Aggregate::Sum
    } else {
        node.get_attr_opt("aggregate")?
            .and_try(|s| node.check_value("aggregate", parse_aggregate(s).ok_or(s)))?
            .unwrap_or(Aggregate::Sum)
    };

    // parse node data from protobuf

    let n_nodes = node.get_attr_slice::<i64>("nodes_featureids")?.len();
    node.expect_attr("nodes_featureids", n_nodes != 0, "at least one node")?;

    let node_ids = get_vec_attr::<usize>(node, "nodes_nodeids", n_nodes)?;
    let tree_ids = get_vec_attr::<usize>(node, "nodes_treeids", n_nodes)?;
    let feature_ids = get_vec_attr::<usize>(node, "nodes_featureids", n_nodes)?;
    let true_ids = get_vec_attr::<usize>(node, "nodes_truenodeids", n_nodes)?;
    let false_ids = get_vec_attr::<usize>(node, "nodes_falsenodeids", n_nodes)?;
    let node_values = get_vec_attr::<f32>(node, "nodes_values", n_nodes)?;
    let nan_is_true = get_vec_attr_opt::<bool>(node, "nodes_missing_value_tracks_true", n_nodes)?
        .unwrap_or_else(|| iter::repeat(false).take(n_nodes).collect());
    let node_modes = node.check_value(
        "nodes_modes",
        get_vec_attr::<&str>(node, "nodes_modes", n_nodes)?
            .into_iter()
            .map(|s| node.check_value("nodes_modes", parse_node_mode(s).ok_or(s)))
            .try_collect::<Vec<_>>(),
    )?;

    // parse leaf data from protobuf

    let leaf_prefix = if is_classifier { "class" } else { "target" };
    let cls = |name| format!("{}_{}", leaf_prefix, name);

    let n_leaves = node.get_attr_slice::<i64>(&cls("ids"))?.len();
    node.expect_attr(&cls("ids"), n_leaves != 0, "at least one leaf")?;

    let leaf_node_ids = get_vec_attr::<usize>(node, &cls("nodeids"), n_leaves)?;
    let leaf_tree_ids = get_vec_attr::<usize>(node, &cls("treeids"), n_leaves)?;
    let leaf_class_ids = get_vec_attr::<usize>(node, &cls("ids"), n_leaves)?;
    let leaf_weights = get_vec_attr::<f32>(node, &cls("weights"), n_leaves)?;

    // check tree ids and count the trees

    let inc_by_1 = |x: &[_]| x.iter().zip(x.iter().skip(1)).all(|(&x, &y)| y == x || y == x + 1);
    node.expect_attr("nodes_treeids", inc_by_1(&tree_ids), "tree ids to increase by 1")?;
    node.expect_attr(&cls("treeids"), inc_by_1(&leaf_tree_ids), "leaf tree ids to increase by 1")?;
    node.expect_attr("nodes_treeids", tree_ids[0] == 0, "tree ids to start from 0")?;
    node.expect_attr(&cls("treeids"), leaf_tree_ids[0] == 0, "leaf tree ids to start from 0")?;
    let n_trees = *tree_ids.last().unwrap();
    node.expect(leaf_tree_ids.last() == Some(&n_trees), "mismatching # of trees (nodes/leaves)")?;

    // build the leaf map and collect all leaves

    let mut leaf_map: HashMap<(usize, usize), (usize, usize)> = HashMap::default();
    let mut leaves: Vec<Vec<Leaf>> = iter::repeat_with(Vec::default).take(n_trees).collect();

    let mut prev_leaf_tree_id = -1i64 as usize;
    let mut prev_leaf_node_id = 0;

    for i in 0..n_leaves {
        let leaf_tree_id = leaf_tree_ids[i];
        let leaf_node_id = leaf_node_ids[i];
        let leaf_class_id = leaf_class_ids[i];
        let leaf_weight = leaf_weights[i];

        if leaf_tree_id == prev_leaf_tree_id {
            node.expect(leaf_node_id >= prev_leaf_node_id, "leaf node ids must be increasing")?;
        }
        let tree_leaves = &mut leaves[leaf_tree_id];
        leaf_map
            .entry((leaf_tree_id, leaf_node_id))
            .or_insert_with(|| (tree_leaves.len(), tree_leaves.len()))
            .1 += 1;
        tree_leaves.push(Leaf::new(leaf_class_id, leaf_weight));

        prev_leaf_tree_id = leaf_tree_id;
        prev_leaf_node_id = leaf_node_id;
    }

    // collect all nodes

    let mut nodes: Vec<Vec<Node>> = Vec::default();

    let mut prev_tree_id = -1i64 as usize;
    let mut prev_node_id = 0;

    for i in 0..n_nodes {
        let tree_id = tree_ids[i];
        let node_id = node_ids[i];
        let node_mode = node_modes[i];

        if tree_id != prev_tree_id {
            node.expect(node_id == 0, "node ids must start from 0 for each tree")?;
        } else {
            node.expect(node_id == prev_node_id + 1, "node ids must increase by 1")?;
        }
        let tree_node = if let Some(cmp) = node_mode {
            let feature_id = feature_ids[i];
            let value = node_values[i];
            let true_id = true_ids[i];
            let false_id = false_ids[i];
            let nan_is_true = nan_is_true[i];
            Node::new_branch(cmp, feature_id, value, true_id, false_id, nan_is_true)
        } else {
            if let Some(&(start, end)) = leaf_map.get(&(tree_id, node_id)) {
                Node::new_leaf(start, end)
            } else {
                node.bail(&format!("leaf not found: tree_id={}, node_id={}", tree_id, node_id))?
            }
        };
        nodes[tree_id].push(tree_node);

        prev_tree_id = tree_id;
        prev_node_id = node_id;
    }

    // build the trees

    let mut trees: Vec<Tree> = Vec::default();

    for i in 0..n_trees {
        let tree = Tree::build(n_classes, &nodes[i], &leaves[i])
            .or_else(|err| node.bail(&format!("{}", err)))?;
        trees.push(tree);
    }

    // build the tree ensemble

    let base_scores = base_values.as_ref().map(Vec::as_slice);
    let ensemble = TreeEnsemble::build(&trees, aggregate_fn, post_transform, base_scores)
        .or_else(|err| node.bail(&format!("{}", err)))?;

    Ok(ensemble)
}

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("TreeEnsembleClassifier", TreeEnsembleClassifier::parse);
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClassLabels {
    Ints(Vec<i64>),
    Strings(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct TreeEnsembleClassifier {
    ensemble: TreeEnsemble,
    class_labels: ClassLabels,
}

impl TreeEnsembleClassifier {
    fn parse(_ctx: &ParsingContext, node: &NodeProto) -> TractResult<Box<InferenceOp>> {
        let ensemble = parse_tree_ensemble(node, true)?;
        let class_labels = match node.get_attr_opt_slice::<i64>("classlabels_int64s")? {
            Some(int_labels) => ClassLabels::Ints(int_labels.into()),
            _ => {
                let str_labels = node.get_attr_opt_vec::<String>("classlabels_strings")?;
                ClassLabels::Strings(str_labels.unwrap())
            }
        };
        Ok(Box::new(Self { ensemble, class_labels }))
    }
}

impl Op for TreeEnsembleClassifier {
    fn name(&self) -> Cow<str> {
        "onnx-ml.TreeEnsembleClassifier".into()
    }
}

impl StatelessOp for TreeEnsembleClassifier {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unimplemented!()
    }
}

impl InferenceRulesOp for TreeEnsembleClassifier {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self, s: &mut Solver<'r>, inputs: &'p [TensorProxy], outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 2)?;

        s.equals(&outputs[0].rank, 1)?;
        s.equals(&outputs[0].rank, inputs[1].rank.bex() - 1)?;
        s.equals(&outputs[1].rank, &inputs[1].rank)?;

        s.given(&inputs[0].rank, move |s, rank| {
            if rank == 1 && rank != 2 {
                bail!("First input rank must be 1 or 2");
            }
            if rank == 2 {
                s.equals(&inputs[0].shape[0], &outputs[0].shape[0])?;
                s.equals(&inputs[0].shape[0], &outputs[1].shape[0])?;
            }
            // FIXME: do we somehow know Nfeatures ?
            // s.equals(&inputs[0].shape[rank-1], );
            s.equals(&outputs[1].shape[rank as usize - 1], self.ensemble.n_classes().to_dim())?;
            Ok(())
        })?;


        s.given(&inputs[0].datum_type, move |_, dt| {
            Ok(match dt {
                DatumType::F32 | DatumType::F64 | DatumType::I64 | DatumType::I32 => (),
                _ => bail!("invalid input type for tree ensemble classifier: {:?}", dt),
            })
        })?;
        match self.class_labels {
            ClassLabels::Ints(_) => s.equals(&outputs[0].datum_type, &DatumType::I64)?,
            ClassLabels::Strings(_) => s.equals(&outputs[0].datum_type, &DatumType::String)?,
        };
        s.equals(&outputs[1].datum_type, DatumType::F32)?;

        Ok(())
    }

    inference_op_as_op!();
}
