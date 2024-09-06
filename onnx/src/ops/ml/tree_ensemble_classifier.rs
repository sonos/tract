use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use crate::pb_helpers::*;
use std::iter;
use tract_hir::internal::*;
use tract_hir::ops::array::{Slice, TypedConcat};
use tract_onnx_opl::ml::tree::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("TreeEnsembleClassifier", tree_classifier);
}

fn tree_classifier(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let ensemble = parse_nodes_data(node, true)?;
    let class_labels = parse_class_data(node)?;
    let base_class_score =
        get_vec_attr_opt::<f32>(node, "base_values", ensemble.n_classes())?.map(|t| rctensor1(&t));
    let post_transform =
        node.get_attr_opt("post_transform")?.map(parse_post_transform).transpose()?.unwrap_or(None);

    // even numbers in leaves are categories id target of leaf contrib
    let binary_result_layout = class_labels.len() < 3
        && ensemble
            .data
            .leaves
            .as_slice::<u32>()?
            .iter()
            .enumerate()
            .all(|(ix, v)| ix % 2 == 1 || *v == 0);

    Ok((
        expand(TreeEnsembleClassifier {
            ensemble,
            class_labels,
            base_class_score,
            post_transform,
            binary_result_layout,
        }),
        vec![],
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PostTransform {
    Softmax,
    Logistic,
    // SoftmaxZero,
    // Probit, // probit, especially multinomial, is p.i.t.a. - so let's ignore it for now
}

pub fn parse_post_transform(s: &str) -> TractResult<Option<PostTransform>> {
    match s {
        "NONE" => Ok(None),
        "SOFTMAX" => Ok(Some(PostTransform::Softmax)),
        "LOGISTIC" => Ok(Some(PostTransform::Logistic)),
        "PROBIT" | "SOFTMAX_ZERO" => bail!("PROBIT and SOFTMAX_ZERO unsupported"),
        _ => bail!("Invalid post transform: {}", s),
    }
}

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

fn parse_class_data(node: &NodeProto) -> TractResult<Arc<Tensor>> {
    // parse n_classes from protobuf
    let ints = node.get_attr_opt_slice::<i64>("classlabels_int64s")?;
    let strs = node.get_attr_opt_tvec::<&str>("classlabels_strings")?;
    match (ints, strs) {
        (Some(n), None) => Ok(rctensor1(n)),
        (None, Some(n)) => Ok(rctensor1(&n.iter().map(|d| d.to_string()).collect::<Vec<_>>())),
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

    let max_used_features = feature_ids.iter().max().copied().unwrap_or(0);

    use tract_onnx_opl::ml::tree_ensemble_classifier::parse_aggregate;
    // parse post_transform from protobuf
    // parse aggregate_fn from protobuf (for regressors)
    let aggregate_fn = parse_aggregate(if is_classifier {
        "SUM"
    } else {
        node.get_attr_opt("aggregate")?.unwrap_or("SUM")
    })?;

    // parse leaf data from protobuf
    let leaf_prefix = if is_classifier { "class" } else { "target" };
    let cls = |name| format!("{leaf_prefix}_{name}");

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
    let trees = rctensor1(&trees);
    let nodes = tensor1(&nodes).into_shape(&[nodes.len() / 5, 5])?.into_arc_tensor();
    let leaves = tensor1(&leaves).into_shape(&[leaves.len() / 2, 2])?.into_arc_tensor();
    let data = TreeEnsembleData { trees, nodes, leaves };
    TreeEnsemble::build(data, max_used_features, n_classes, aggregate_fn)
}

#[derive(Debug, Clone, Hash)]
pub struct TreeEnsembleClassifier {
    pub ensemble: TreeEnsemble,
    pub class_labels: Arc<Tensor>,
    pub base_class_score: Option<Arc<Tensor>>,
    pub post_transform: Option<PostTransform>,
    pub binary_result_layout: bool,
}



impl Expansion for TreeEnsembleClassifier {
    fn name(&self) -> Cow<str> {
        "TreeEnsembleClassifier".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("binary result layout kludge: {:?}", self.binary_result_layout)])
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 2)?;

        s.equals(&outputs[0].datum_type, self.class_labels.datum_type())?;
        s.equals(&outputs[1].datum_type, DatumType::F32)?;

        s.equals(&outputs[0].rank, 1)?;
        s.equals(&outputs[1].rank, 2)?;
        s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?;
        s.equals(&outputs[1].shape[0], &inputs[0].shape[0])?;
        if self.binary_result_layout {
            s.equals(&outputs[1].shape[1], 2.to_dim())?;
        } else {
            s.equals(&outputs[1].shape[1], self.class_labels.len().to_dim())?;
        }

        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        use tract_core::ops::nn::*;

        let mut scores = model.wire_node(
            format!("{prefix}.classifier"),
            tract_onnx_opl::ml::tree_ensemble_classifier::TreeEnsembleClassifier {
                ensemble: self.ensemble.clone(),
            },
            inputs,
        )?;
        if let Some(base_class_score) = self.base_class_score.as_deref() {
            let base = base_class_score.clone().broadcast_into_rank(2)?.into_arc_tensor();
            let base = model.add_const(prefix.to_string() + ".base", base)?;
            scores = model.wire_node(
                format!("{prefix}.base_class_score"),
                tract_core::ops::math::add(),
                &[scores[0], base],
            )?;
        }
        match self.post_transform {
            None => (),
            Some(PostTransform::Softmax) => {
                scores = tract_hir::ops::nn::LayerSoftmax::new(1, false).wire(
                    &format!("{prefix}.softmax"),
                    model,
                    &scores,
                )?;
            }
            Some(PostTransform::Logistic) => {
                scores = model.wire_node(
                    format!("{prefix}.logistic"),
                    tract_core::ops::nn::sigmoid(),
                    &scores,
                )?;
            }
        }
        let processed_scores = scores.clone();
        if self.binary_result_layout {
            scores = model.wire_node(
                format!("{prefix}.binary_result_slice"),
                Slice::new(1, 0, 1),
                &scores,
            )?;
            let one = model.add_const(prefix.to_string() + ".one", rctensor2(&[[1f32]]))?;
            let complement = model.wire_node(
                format!("{prefix}.binary_result_complement"),
                tract_core::ops::math::sub(),
                &[one, scores[0]],
            )?;
            scores = model.wire_node(
                format!("{prefix}.binary_result"),
                TypedConcat::new(1),
                &[complement[0], scores[0]],
            )?;
        }
        let winners = model.wire_node(
            format!("{prefix}.argmax"),
            Reduce::new(tvec!(1), Reducer::ArgMax(false)),
            &processed_scores,
        )?;
        let reduced = model.wire_node(
            format!("{prefix}.rm_axis"),
            tract_core::ops::change_axes::AxisOp::Rm(1),
            &winners,
        )?;
        let casted = model.wire_node(
            format!("{prefix}.casted"),
            tract_core::ops::cast::cast(i32::datum_type()),
            &reduced,
        )?;
        let labels = model.wire_node(
            format!("{prefix}.labels"),
            tract_onnx_opl::ml::DirectLookup::new(
                self.class_labels.clone(),
                Tensor::zero_dt(self.class_labels.datum_type(), &[])?.into_arc_tensor(),
            )?,
            &casted,
        )?[0];
        Ok(tvec!(labels, scores[0]))
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(2)
    }
}
