use std::iter;
use tract_onnx_opl::ml_trees::Cmp;
use tract_onnx_opl::ml_trees::*;

use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use crate::pb_helpers::AttrTVecType;

use tract_hir::internal::*;

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

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("TreeEnsembleClassifier", tree_classifier);
}

fn tree_classifier(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let ensemble = parse_nodes_data(node, true)?;
    let class_labels = parse_class_data(node)?;
    Ok((inference_wrap(TreeEnsembleClassifier { ensemble, class_labels }, 2, rules), vec![]))
}

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

fn parse_class_data(node: &NodeProto) -> TractResult<Tensor> {
    // parse n_classes from protobuf
    let ints = node.get_attr_opt_slice::<i64>("classlabels_int64s")?;
    let strs = node.get_attr_opt_tvec::<&str>("classlabels_strings")?;
    match (ints, strs) {
        (Some(n), None) => Ok(tensor1(n)),
        (None, Some(n)) => Ok(tensor1(&n.iter().map(|d| d.to_string()).collect::<Vec<_>>())),
        (None, None) => {
            bail!("cannot find neither 'classlabels_int64s' not 'classlabels_strings'")
        }
        (Some(_), Some(_)) => {
            bail!("only one of 'classlabels_int64s' and 'classlabels_strings' can be set")
        }
    }
}

fn parse_nodes_data(node: &NodeProto, is_classifier: bool) -> TractResult<TreeEnsemble> {
    // parse n_classes from protobuf
    let n_classes = if is_classifier {
        let ints = node.get_attr_opt_slice::<i64>("classlabels_int64s")?;
        let strs = node.get_attr_opt_tvec::<&str>("classlabels_strings")?;
        match (ints, strs) {
            (Some(n), None) => n.len(),
            (None, Some(n)) => n.len(),
            (None, None) => {
                bail!("cannot find neither 'classlabels_int64s' not 'classlabels_strings'")
            }
            (Some(_), Some(_)) => {
                bail!("only one of 'classlabels_int64s' and 'classlabels_strings' can be set")
            }
        }
    } else {
        node.get_attr("n_targets")?
    };
    let n_nodes = node.get_attr_slice::<i64>("nodes_featureids")?.len();
    node.expect_attr("nodes_featureids", n_nodes != 0, "at least one node")?;

    // parse base_values from protobuf
    let base_values = get_vec_attr_opt::<f32>(node, "base_values", n_classes)?;

    let node_ids = get_vec_attr::<usize>(node, "nodes_nodeids", n_nodes)?;
    let tree_ids = get_vec_attr::<usize>(node, "nodes_treeids", n_nodes)?;
    let feature_ids = get_vec_attr::<usize>(node, "nodes_featureids", n_nodes)?;
    let true_ids = get_vec_attr::<usize>(node, "nodes_truenodeids", n_nodes)?;
    let false_ids = get_vec_attr::<usize>(node, "nodes_falsenodeids", n_nodes)?;
    let node_values = get_vec_attr::<f32>(node, "nodes_values", n_nodes)?;
    let nan_is_true = get_vec_attr_opt::<bool>(node, "nodes_missing_value_tracks_true", n_nodes)?
        .unwrap_or_else(|| iter::repeat(false).take(n_nodes).collect());
    let node_modes: Vec<Option<Cmp>> = get_vec_attr::<&str>(node, "nodes_modes", n_nodes)?
        .iter()
        .map(|s| parse_node_mode(s))
        .collect::<TractResult<_>>()?;

    let n_features = feature_ids.iter().max().copied().unwrap_or(0) + 1;

    // parse post_transform from protobuf
    let post_transform =
        node.get_attr_opt("post_transform")?.map(parse_post_transform).transpose()?.unwrap_or(None);

    // parse aggregate_fn from protobuf (for regressors)
    let aggregate_fn = parse_aggregate(if is_classifier {
        "SUM"
    } else {
        node.get_attr_opt("aggregate")?.unwrap_or("SUM")
    })?;

    // parse leaf data from protobuf
    let leaf_prefix = if is_classifier { "class" } else { "target" };
    let cls = |name| format!("{}_{}", leaf_prefix, name);

    let n_leaves = node.get_attr_slice::<i64>(&cls("ids"))?.len();
    node.expect_attr(&cls("ids"), n_leaves != 0, "at least one leaf")?;

    let leaf_node_ids = get_vec_attr::<usize>(node, &cls("nodeids"), n_leaves)?;
    let leaf_tree_ids = get_vec_attr::<usize>(node, &cls("treeids"), n_leaves)?;
    let leaf_class_ids = get_vec_attr::<usize>(node, &cls("ids"), n_leaves)?;
    let leaf_weights = get_vec_attr::<f32>(node, &cls("weights"), n_leaves)?;

    let inc_by_1 = |x: &[_]| x.iter().zip(x.iter().skip(1)).all(|(&x, &y)| y == x || y == x + 1);
    node.expect_attr("nodes_treeids", inc_by_1(&tree_ids), "tree ids to increase by 1")?;
    node.expect_attr(&cls("treeids"), inc_by_1(&leaf_tree_ids), "leaf tree ids to increase by 1")?;
    node.expect_attr("nodes_treeids", tree_ids[0] == 0, "tree ids to start from 0")?;
    node.expect_attr(&cls("treeids"), leaf_tree_ids[0] == 0, "leaf tree ids to start from 0")?;
    let n_trees = *tree_ids.last().unwrap() + 1;
    node.expect(
        leaf_tree_ids.last() == Some(&(n_trees - 1)),
        "mismatching # of trees (nodes/leaves)",
    )?;

    let mut node_order: Vec<usize> = (0usize..node_ids.len()).collect();
    node_order.sort_by_key(|&ix| (tree_ids[ix], node_ids[ix]));

    let mut leaf_order: Vec<usize> = (0usize..leaf_node_ids.len()).collect();
    leaf_order.sort_by_key(|&ix| (leaf_tree_ids[ix], leaf_node_ids[ix]));

    let mut trees = vec![];
    let mut nodes: Vec<u32> = vec![];
    let mut leaves: Vec<u32> = vec![];
    let mut current_tree_id = None;
    let mut in_leaf_ix = 0;
    for n in node_order.into_iter() {
        let node_id = node_ids[n];
        let tree_id = tree_ids[n];
        if Some(tree_id) != current_tree_id {
            current_tree_id = Some(tree_id);
            trees.push(nodes.len() as u32 / 5);
        }
        if let Some(mode) = node_modes[n] {
            let mut row = [0u32; 5];
            row[0] = feature_ids[n] as u32;
            row[1] = true_ids[n] as u32 + trees.last().unwrap();
            row[2] = false_ids[n] as u32 + trees.last().unwrap();
            row[3] = node_values[n].to_bits();
            row[4] = (0x0100u32 * nan_is_true[n] as u32) | mode as u32;
            nodes.extend(row.iter());
        } else {
            let mut row = [0u32; 5];
            row[0] = leaves.len() as u32 / 2;
            loop {
                if in_leaf_ix >= leaf_order.len()
                    || leaf_tree_ids[leaf_order[in_leaf_ix]] != tree_id
                    || leaf_node_ids[leaf_order[in_leaf_ix]] != node_id
                {
                    break;
                }
                let leaf_ix = leaf_order[in_leaf_ix];
                leaves.push(leaf_class_ids[leaf_ix] as u32);
                leaves.push(leaf_weights[leaf_ix].to_bits());
                in_leaf_ix += 1;
            }
            row[1] = leaves.len() as u32 / 2;
            nodes.extend(row.iter());
        };
    }
    let trees = tensor1(&*trees);
    let nodes = tensor1(&*nodes).into_shape(&[nodes.len() / 5, 5])?;
    let leaves = tensor1(&*leaves).into_shape(&[leaves.len() / 2, 2])?;
    let data = TreeEnsembleData { trees, nodes, leaves };
    TreeEnsemble::build(data, n_features, n_classes, aggregate_fn, post_transform, base_values)
}

fn rules<'r, 'p, 's>(
    op: &'s dyn Op,
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    let op = op.downcast_ref::<TreeEnsembleClassifier>().context("Wrong op")?;

    check_input_arity(&inputs, 1)?;
    check_output_arity(&outputs, 2)?;

    s.equals(&outputs[0].datum_type, op.class_labels.datum_type())?;
    s.equals(&outputs[1].datum_type, DatumType::F32)?;

    s.equals(&outputs[0].rank, 1)?;
    s.equals(&outputs[1].rank, 2)?;
    s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?;
    s.equals(&outputs[1].shape[0], &inputs[0].shape[0])?;
    s.equals(&outputs[1].shape[1], &op.class_labels.len().to_dim())?;

    /*
    s.given(&inputs[0].rank, move |s, rank| {
    if rank < 1 || rank > 2 {
    bail!("First input rank must be 1 or 2");
    }
    if rank == 2 {
    s.equals(&inputs[0].shape[0], &outputs[0].shape[0])?;
    s.equals(&inputs[0].shape[0], &outputs[1].shape[0])?;
    }
    s.given(&inputs[0].shape[rank as usize - 1], move |_, feats| {
    self.ensemble.check_n_features(feats.to_usize()?)
    })?;
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
    */

    Ok(())
}

/*
fn nboutputs(&self) -> TractResult<usize> {
Ok(2)
}
*/
