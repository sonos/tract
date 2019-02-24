use std::cmp::Ordering;
use std::fmt::{self, Debug, Display};
use std::iter;

use tract_core::ops::prelude::*;

use ndarray::{
    Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMut1, Axis, Ix1, Ix2,
};
use num_traits::AsPrimitive;
use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    LessEqual,
    Less,
    GreaterEqual,
    Greater,
    Equal,
    NotEqual,
}

impl Cmp {
    pub fn compare<T>(&self, x: T, y: T) -> bool
    where
        T: PartialOrd + Copy,
    {
        match self {
            Cmp::LessEqual => x <= y,
            Cmp::Less => x < y,
            Cmp::GreaterEqual => x >= y,
            Cmp::Greater => x > y,
            Cmp::Equal => x == y,
            Cmp::NotEqual => x != y,
        }
    }
}

impl Display for Cmp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Cmp::LessEqual => "<=",
            Cmp::Less => "<",
            Cmp::GreaterEqual => ">=",
            Cmp::Greater => ">",
            Cmp::Equal => "==",
            Cmp::NotEqual => "!=",
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BranchNode {
    pub cmp: Cmp, // TODO: perf: most real forests have only 1 type of comparison
    pub feature_id: usize,
    pub value: f32,
    pub true_id: usize,
    pub false_id: usize,
    pub nan_is_true: bool,
}

impl BranchNode {
    pub fn get_child_id(&self, feature: f32) -> usize {
        let condition =
            if feature.is_nan() { self.nan_is_true } else { self.cmp.compare(feature, self.value) };
        if condition {
            self.true_id
        } else {
            self.false_id
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LeafNode {
    pub start_id: usize,
    pub end_id: usize,
}

#[derive(Copy, Clone, Debug)]
pub enum Node {
    Branch(BranchNode),
    Leaf(LeafNode),
}

impl Node {
    pub fn new_branch(
        cmp: Cmp, feature_id: usize, value: f32, true_id: usize, false_id: usize, nan_is_true: bool,
    ) -> Self {
        Node::Branch(BranchNode { cmp, feature_id, value, true_id, false_id, nan_is_true })
    }

    pub fn new_leaf(start_id: usize, end_id: usize) -> Self {
        Node::Leaf(LeafNode { start_id, end_id })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Leaf {
    class_id: usize,
    weight: f32,
}

impl Leaf {
    pub fn new(class_id: usize, weight: f32) -> Leaf {
        Leaf { class_id, weight }
    }
}

pub trait AggregateFn: Default {
    fn aggregate(&mut self, score: f32, total: &mut f32);

    fn post_aggregate(&mut self, _total: &mut f32) {}
}

#[derive(Clone, Copy, Default, Debug)]
pub struct SumFn;

impl AggregateFn for SumFn {
    fn aggregate(&mut self, score: f32, total: &mut f32) {
        *total += score;
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct AvgFn {
    count: usize,
}

impl AggregateFn for AvgFn {
    fn aggregate(&mut self, score: f32, total: &mut f32) {
        *total += score;
        self.count += 1;
    }

    fn post_aggregate(&mut self, total: &mut f32) {
        if self.count > 1 {
            *total /= self.count as f32;
        }
        self.count = 0;
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct MaxFn;

impl AggregateFn for MaxFn {
    fn aggregate(&mut self, score: f32, total: &mut f32) {
        *total = total.max(score);
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct MinFn;

impl AggregateFn for MinFn {
    fn aggregate(&mut self, score: f32, total: &mut f32) {
        *total = total.min(score);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aggregate {
    Sum,
    Avg,
    Max,
    Min,
}

impl Default for Aggregate {
    fn default() -> Self {
        Aggregate::Sum
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostTransform {
    Softmax,
    Logistic,
    SoftmaxZero,
    // Probit, // probit, especially multinomial, is p.i.t.a. - so let's ignore it for now
}

impl PostTransform {
    pub fn apply(&self, scores: &mut ArrayViewMut1<f32>) {
        match self {
            PostTransform::Softmax => {
                // TODO: stability w.r.t. nan/inf/-inf?
                let max: f32 = scores
                    .iter()
                    .cloned()
                    .max_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
                    .unwrap_or(0.0);
                let mut norm = 0.;
                scores.map_inplace(|a| {
                    *a = (*a - max).exp();
                    norm += *a;
                });
                scores.map_inplace(|a| *a = *a / norm);
            }
            PostTransform::Logistic => {
                scores.map_inplace(|a| {
                    let v = 1. / (1. + f32::exp(-f32::abs(*a)));
                    *a = if a.is_sign_negative() { 1. - v } else { v };
                });
            }
            PostTransform::SoftmaxZero => {
                // TODO: stability w.r.t. nan/inf/-inf?
                let max: f32 = scores
                    .iter()
                    .cloned()
                    .filter(|&a| a != 0.)
                    .max_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
                    .unwrap_or(0.0);
                let mut norm = 0.;
                scores.map_inplace(|a| {
                    if *a != 0. {
                        *a = (*a - max).exp();
                        norm += *a;
                    }
                });
                scores.map_inplace(|a| *a = *a / norm);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tree {
    n_classes: usize,
    nodes: Vec<Node>, // TODO: can this be a slice/view into ensemble's contiguous storage?
    leaves: Vec<Leaf>, // TODO: store as a collection of direct slices instead of indices?
}

fn ensure<O, T>(
    t: &str, index: usize, obj: O, an: &str, a: T, cmp: Cmp, bn: &str, b: T,
) -> TractResult<()>
where
    O: Debug,
    T: Display + PartialOrd + Copy,
{
    let cond = cmp.compare(a, b);
    ensure!(cond, "Invalid {} #{}: {} = {} {} {} = {} ({:?})", t, index, an, a, cmp, bn, b, obj);
    Ok(())
}

impl Tree {
    pub fn build(n_classes: usize, nodes: &[Node], leaves: &[Leaf]) -> TractResult<Self> {
        use self::Cmp::{Equal, Less, LessEqual};

        let n_nodes = nodes.len();
        ensure!(n_nodes != 0, "Invalid tree: no nodes");
        let n_leaves = leaves.len();
        ensure!(n_leaves != 0, "Invalid tree: no leaves");

        let mut has_parents: Vec<_> = iter::repeat(false).take(n_nodes).collect();
        let mut leaf_coverage: Vec<_> = iter::repeat(0).take(n_leaves).collect();
        for (i, node) in nodes.iter().enumerate() {
            match node {
                Node::Branch(ref b) => {
                    ensure("node", i, node, "feature_id", b.feature_id, Less, "n_nodes", n_nodes)?;
                    ensure("node", i, node, "true_id", b.true_id, Less, "n_nodes", n_nodes)?;
                    ensure("node", i, node, "false_id", b.false_id, Less, "n_nodes", n_nodes)?;
                    has_parents[b.true_id] = true;
                    has_parents[b.false_id] = true;
                }
                Node::Leaf(ref l) => {
                    ensure("node", i, node, "start_id", l.start_id, Less, "end_id", l.end_id)?;
                    ensure("node", i, node, "end_id", l.start_id, LessEqual, "n_leaves", n_leaves)?;
                    for j in l.start_id..l.end_id {
                        leaf_coverage[j] += 1;
                    }
                }
            }
        }

        ensure!(!has_parents[0], "Invalid tree: expected node #0 to have no parents (root)");
        let n_orphans = has_parents.iter().skip(1).filter(|&x| !*x).count();
        ensure!(n_orphans == 0, "Invalid tree: {} orphan nodes", n_orphans);

        let n_orphan_leaves = leaf_coverage.iter().filter(|&x| *x == 0).count();
        ensure!(n_orphan_leaves == 0, "Invalid tree: {} orphan leaves", n_orphan_leaves);

        for (i, leaf) in leaves.iter().enumerate() {
            ensure("leaf", i, leaf, "class_id", leaf.class_id, Less, "n_classes", n_classes)?;
            // TODO: be more strict and check for nan/inf/-inf here, or not?
            let w_finite = leaf.weight.is_finite();
            ensure("leaf", i, leaf, "weight.is_finite()", w_finite, Equal, "true", true)?;
        }

        Ok(Self { n_classes, nodes: nodes.into(), leaves: leaves.into() })
    }

    pub fn branches(&self) -> impl Iterator<Item = &BranchNode> {
        self.nodes.iter().filter_map(|node| match node {
            Node::Branch(ref branch) => Some(branch),
            _ => None,
        })
    }

    pub fn max_feature_id(&self) -> usize {
        self.branches().map(|b| b.feature_id).max().unwrap_or(0)
    }

    unsafe fn get_leaves_unchecked<T>(&self, input: &ArrayView1<T>) -> &[Leaf]
    where
        T: AsPrimitive<f32>,
    {
        let mut node_id = 0;
        loop {
            let node = self.nodes.get_unchecked(node_id);
            match node {
                Node::Branch(ref b) => {
                    let feature = *input.uget(b.feature_id);
                    node_id = b.get_child_id(feature.as_());
                }
                Node::Leaf(ref l) => return &self.leaves[l.start_id..l.end_id],
            }
        }
    }

    unsafe fn eval_unchecked<A, T>(
        &self, input: &ArrayView1<T>, output: &mut ArrayViewMut1<f32>, aggs: &mut [A],
    ) where
        A: AggregateFn,
        T: AsPrimitive<f32>,
    {
        for leaf in self.get_leaves_unchecked(input) {
            let agg_fn = aggs.get_unchecked_mut(leaf.class_id);
            agg_fn.aggregate(leaf.weight, output.uget_mut(leaf.class_id));
        }
    }
}

#[derive(Clone, Debug)]
pub struct TreeEnsemble {
    trees: Vec<Tree>,
    max_feature_id: usize,
    n_classes: usize,
    aggregate_fn: Aggregate, // TODO: should this be an argument to eval()?
    post_transform: Option<PostTransform>, // TODO: should this be an argument to eval()?
}

impl TreeEnsemble {
    pub fn build(
        trees: &[Tree], aggregate_fn: Aggregate, post_transform: Option<PostTransform>,
    ) -> TractResult<Self> {
        ensure!(trees.len() > 0, "Invalid tree ensemble: cannot be empty");
        let max_feature_id = trees.iter().map(Tree::max_feature_id).max().unwrap_or(0);
        let n_classes = trees[0].n_classes;
        for tree in trees.iter().skip(1) {
            ensure!(
                tree.n_classes == n_classes,
                "Invalid tree ensemble: n_classes must be the same (got {} and {})",
                n_classes,
                tree.n_classes
            );
        }
        Ok(Self { trees: trees.into(), max_feature_id, n_classes, aggregate_fn, post_transform })
    }

    unsafe fn eval_one_unchecked<A, T>(
        &self, input: &ArrayView1<T>, output: &mut ArrayViewMut1<f32>, aggs: &mut [A],
    ) where
        A: AggregateFn,
        T: AsPrimitive<f32>,
    {
        for tree in &self.trees {
            tree.eval_unchecked(input, output, aggs);
        }
        for i in 0..self.n_classes {
            aggs.get_unchecked_mut(i).post_aggregate(output.uget_mut(i));
        }
        if let Some(post_transform) = self.post_transform {
            post_transform.apply(output);
        }
    }

    fn check_n_features(&self, n_features: usize) -> TractResult<()> {
        Ok(ensure!(
            n_features > self.max_feature_id,
            "Invalid input shape: got {} features, expected > {}",
            n_features,
            self.max_feature_id
        ))
    }

    fn eval_2d<'i, A, T>(&self, input: &ArrayView2<T>) -> TractResult<Array2<f32>>
    where
        A: AggregateFn,
        T: AsPrimitive<f32>,
    {
        self.check_n_features(input.shape()[1])?;
        let n = input.shape()[0];
        let mut output = Array2::zeros((n, self.n_classes));
        let mut aggs: SmallVec<[A; 16]> =
            iter::repeat_with(Default::default).take(self.n_classes).collect();
        for i in 0..n {
            unsafe {
                self.eval_one_unchecked::<A, T>(
                    &input.index_axis(Axis(0), i),
                    &mut output.index_axis_mut(Axis(0), i),
                    &mut aggs,
                );
            }
        }
        Ok(output)
    }

    fn eval_1d<'i, A, T>(&self, input: &ArrayView1<T>) -> TractResult<Array1<f32>>
    where
        A: AggregateFn,
        T: AsPrimitive<f32>,
    {
        self.check_n_features(input.len())?;
        let mut output = Array1::zeros(self.n_classes);
        let mut aggs: SmallVec<[A; 16]> =
            iter::repeat_with(Default::default).take(self.n_classes).collect();
        unsafe {
            self.eval_one_unchecked::<A, T>(input, &mut output.view_mut(), &mut aggs);
        }
        Ok(output)
    }

    pub fn eval<'i, I, T>(&self, input: I) -> TractResult<ArrayD<f32>>
    where
        I: Into<ArrayViewD<'i, T>>, // TODO: accept generic dimensions, not just IxDyn
        T: AsPrimitive<f32>,
    {
        let input = input.into();
        if let Ok(input) = input.view().into_dimensionality::<Ix1>() {
            Ok(match self.aggregate_fn {
                Aggregate::Sum => self.eval_1d::<SumFn, T>(&input),
                Aggregate::Avg => self.eval_1d::<AvgFn, T>(&input),
                Aggregate::Min => self.eval_1d::<MinFn, T>(&input),
                Aggregate::Max => self.eval_1d::<MaxFn, T>(&input),
            }?
            .into_dyn())
        } else if let Ok(input) = input.view().into_dimensionality::<Ix2>() {
            Ok(match self.aggregate_fn {
                Aggregate::Sum => self.eval_2d::<SumFn, T>(&input),
                Aggregate::Avg => self.eval_2d::<AvgFn, T>(&input),
                Aggregate::Min => self.eval_2d::<MinFn, T>(&input),
                Aggregate::Max => self.eval_2d::<MaxFn, T>(&input),
            }?
            .into_dyn())
        } else {
            bail!("Invalid input dimensionality for tree ensemble: {:?}", input.shape());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    fn generate_gbm_trees() -> Vec<Tree> {
        vec![
            Tree::build(
                3,
                &[
                    Node::new_branch(Cmp::LessEqual, 2, 3.15, 1, 2, true),
                    Node::new_branch(Cmp::LessEqual, 1, 3.35, 3, 4, true),
                    Node::new_leaf(0, 1),
                    Node::new_leaf(1, 2),
                    Node::new_leaf(2, 3),
                ],
                &[Leaf::new(0, -0.075), Leaf::new(0, 0.13928571), Leaf::new(0, 0.15)],
            )
            .unwrap(),
            Tree::build(
                3,
                &[
                    Node::new_branch(Cmp::LessEqual, 2, 1.8, 1, 2, true),
                    Node::new_leaf(0, 1),
                    Node::new_branch(Cmp::LessEqual, 3, 1.65, 3, 4, true),
                    Node::new_branch(Cmp::LessEqual, 2, 4.45, 5, 6, true),
                    Node::new_branch(Cmp::LessEqual, 2, 5.35, 7, 8, true),
                    Node::new_leaf(1, 2),
                    Node::new_leaf(2, 3),
                    Node::new_leaf(3, 4),
                    Node::new_leaf(4, 5),
                ],
                &[
                    Leaf::new(1, -0.075),
                    Leaf::new(1, 0.13548388),
                    Leaf::new(1, 0.110869564),
                    Leaf::new(1, -0.052500002),
                    Leaf::new(1, -0.075),
                ],
            )
            .unwrap(),
            Tree::build(
                3,
                &[
                    Node::new_branch(Cmp::LessEqual, 3, 1.65, 1, 2, true),
                    Node::new_branch(Cmp::LessEqual, 2, 4.45, 3, 4, true),
                    Node::new_branch(Cmp::LessEqual, 2, 5.35, 5, 6, true),
                    Node::new_leaf(0, 1),
                    Node::new_leaf(1, 2),
                    Node::new_leaf(2, 3),
                    Node::new_leaf(3, 4),
                ],
                &[
                    Leaf::new(2, -0.075),
                    Leaf::new(2, -0.035869565),
                    Leaf::new(2, 0.1275),
                    Leaf::new(2, 0.15),
                ],
            )
            .unwrap(),
            Tree::build(
                3,
                &[
                    Node::new_branch(Cmp::LessEqual, 2, 3.15, 1, 2, true),
                    Node::new_branch(Cmp::LessEqual, 1, 3.35, 3, 4, true),
                    Node::new_branch(Cmp::LessEqual, 2, 4.45, 5, 6, true),
                    Node::new_leaf(0, 1),
                    Node::new_leaf(1, 2),
                    Node::new_leaf(2, 3),
                    Node::new_branch(Cmp::LessEqual, 2, 5.35, 7, 8, true),
                    Node::new_leaf(3, 4),
                    Node::new_leaf(4, 5),
                ],
                &[
                    Leaf::new(0, 0.12105576),
                    Leaf::new(0, 0.1304589),
                    Leaf::new(0, -0.07237862),
                    Leaf::new(0, -0.07226522),
                    Leaf::new(0, -0.07220469),
                ],
            )
            .unwrap(),
            Tree::build(
                3,
                &[
                    Node::new_branch(Cmp::LessEqual, 3, 0.45, 1, 2, true),
                    Node::new_branch(Cmp::LessEqual, 2, 1.45, 3, 4, true),
                    Node::new_branch(Cmp::LessEqual, 3, 1.65, 5, 6, true),
                    Node::new_leaf(0, 1),
                    Node::new_leaf(1, 2),
                    Node::new_branch(Cmp::LessEqual, 2, 4.45, 7, 8, true),
                    Node::new_branch(Cmp::LessEqual, 2, 5.35, 9, 10, true),
                    Node::new_leaf(2, 3),
                    Node::new_leaf(3, 4),
                    Node::new_leaf(4, 5),
                    Node::new_leaf(5, 6),
                ],
                &[
                    Leaf::new(1, -0.07226842),
                    Leaf::new(1, -0.07268012),
                    Leaf::new(1, 0.119391434),
                    Leaf::new(1, 0.097440675),
                    Leaf::new(1, -0.049815115),
                    Leaf::new(1, -0.07219931),
                ],
            )
            .unwrap(),
            Tree::build(
                3,
                &[
                    Node::new_branch(Cmp::LessEqual, 2, 4.75, 1, 2, true),
                    Node::new_branch(Cmp::LessEqual, 1, 2.75, 3, 4, true),
                    Node::new_branch(Cmp::LessEqual, 2, 5.15, 7, 8, true),
                    Node::new_leaf(0, 1),
                    Node::new_branch(Cmp::LessEqual, 2, 4.15, 5, 6, true),
                    Node::new_leaf(1, 2),
                    Node::new_leaf(2, 3),
                    Node::new_leaf(3, 4),
                    Node::new_leaf(4, 5),
                ],
                &[
                    Leaf::new(2, -0.061642267),
                    Leaf::new(2, -0.0721846),
                    Leaf::new(2, -0.07319043),
                    Leaf::new(2, 0.076814815),
                    Leaf::new(2, 0.1315959),
                ],
            )
            .unwrap(),
        ]
    }

    fn generate_gbm_ensemble(post_transform: Option<PostTransform>) -> TreeEnsemble {
        // converted manually from LightGBM, fitted on iris dataset
        let trees = generate_gbm_trees();
        TreeEnsemble::build(&trees, Aggregate::Sum, post_transform).unwrap()
    }

    fn generate_gbm_input() -> Array2<f32> {
        arr2(&[
            [5.1, 3.5, 1.4, 0.2],
            [5.4, 3.7, 1.5, 0.2],
            [5.4, 3.4, 1.7, 0.2],
            [4.8, 3.1, 1.6, 0.2],
            [5.0, 3.5, 1.3, 0.3],
            [7.0, 3.2, 4.7, 1.4],
            [5.0, 2.0, 3.5, 1.0],
            [5.9, 3.2, 4.8, 1.8],
            [5.5, 2.4, 3.8, 1.1],
            [5.5, 2.6, 4.4, 1.2],
            [6.3, 3.3, 6.0, 2.5],
            [6.5, 3.2, 5.1, 2.0],
            [6.9, 3.2, 5.7, 2.3],
            [7.4, 2.8, 6.1, 1.9],
            [6.7, 3.1, 5.6, 2.4],
        ])
    }

    fn generate_gbm_raw_output() -> Array2<f32> {
        arr2(&[
            [0.28045893, -0.14726841, -0.14718461],
            [0.28045893, -0.14768013, -0.14718461],
            [0.28045893, -0.14768013, -0.14718461],
            [0.26034147, -0.14768013, -0.14718461],
            [0.28045893, -0.14726841, -0.14718461],
            [-0.14726523, 0.20831025, -0.10905999],
            [-0.14737862, 0.25487530, -0.13664228],
            [-0.14726523, -0.10231511, 0.20431481],
            [-0.14737862, 0.25487530, -0.13664228],
            [-0.14737862, 0.25487530, -0.13664228],
            [-0.14720470, -0.14719930, 0.28159590],
            [-0.14726523, -0.10231511, 0.20431481],
            [-0.14720470, -0.14719930, 0.28159590],
            [-0.14720470, -0.14719930, 0.28159590],
            [-0.14720470, -0.14719930, 0.28159590],
        ])
    }

    fn generate_gbm_softmax_output_approx() -> Array2<f32> {
        arr2(&[
            [0.43402156, 0.28297736, 0.28300108],
            [0.43407212, 0.28289383, 0.28303405],
            [0.43407212, 0.28289383, 0.28303405],
            [0.42913692, 0.28536082, 0.28550226],
            [0.43402156, 0.28297736, 0.28300108],
            [0.28852151, 0.41172066, 0.29975782],
            [0.28522654, 0.42646814, 0.28830533],
            [0.28840992, 0.30166975, 0.40992033],
            [0.28522654, 0.42646814, 0.28830533],
            [0.28522654, 0.42646814, 0.28830533],
            [0.28285181, 0.28285333, 0.43429486],
            [0.28840992, 0.30166975, 0.40992033],
            [0.28285181, 0.28285333, 0.43429486],
            [0.28285181, 0.28285333, 0.43429486],
            [0.28285181, 0.28285333, 0.43429486],
        ])
    }

    #[test]
    fn test_tree_ensemble() {
        let ensemble = generate_gbm_ensemble(None);
        let input = generate_gbm_input();
        let output = ensemble.eval(&input.view().into_dyn()).unwrap();
        assert_eq!(output, generate_gbm_raw_output().into_dyn());

        let ensemble = generate_gbm_ensemble(Some(PostTransform::Softmax));
        let output = ensemble.eval(&input.view().into_dyn()).unwrap();
        assert!(output.all_close(&generate_gbm_softmax_output_approx(), 1e-7));
    }
}
