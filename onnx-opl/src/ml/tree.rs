use std::convert::TryFrom;
use std::convert::TryInto;
use std::fmt::{self, Debug, Display};
use std::iter;

use tract_nnef::internal::*;

use tract_ndarray::{
    Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMut1, Axis, Ix1, Ix2,
};

use tract_num_traits::AsPrimitive;

macro_rules! ensure {
    ($cond: expr, $($rest: expr),* $(,)?) => {
        if !$cond {
            bail!($($rest),*)
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cmp {
    Equal = 1,
    NotEqual = 2,
    Less = 3,
    Greater = 4,
    LessEqual = 5,
    GreaterEqual = 6,
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
    pub fn to_u8(&self) -> u8 {
        unsafe { std::mem::transmute(*self) }
    }
}

impl TryFrom<u8> for Cmp {
    type Error = TractError;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if (1..=5).contains(&value) {
            unsafe { Ok(std::mem::transmute::<u8, Cmp>(value)) }
        } else {
            bail!("Invalid value for Cmp: {}", value);
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

#[derive(Debug, Clone, Hash)]
pub struct TreeEnsembleData {
    // u32, [Ntrees], root row of each tree in nodes array (in rows)
    pub trees: Arc<Tensor>,
    // u32, [_, 5],
    // 5th number is flags: last byte is comparator, 0 for leaves, transmuted Cmp for the internal nodes
    //                      is_nan is 0x100 bit
    // intern nodes:    feature_id, true_id, false_id, value.to_bits(),
    //                  comp | (0x100 if nan_is_true)
    // leaves:          start row, end row in leaves array, 3 zeros for padding
    pub nodes: Arc<Tensor>,
    // categ,
    pub leaves: Arc<Tensor>,
}

impl Display for TreeEnsembleData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tree = self.trees.as_slice::<u32>().unwrap();
        for t in 0..tree.len() {
            let last_node = tree.get(t + 1).cloned().unwrap_or(self.nodes.len() as u32 / 5);
            writeln!(f, "Tree {}, nodes {:?}", t, tree[t]..last_node)?;
            for n in tree[t]..last_node {
                unsafe {
                    let node = self.get_unchecked(n as _);
                    if let TreeNode::Leaf(leaf) = node {
                        for vote in leaf.start_id..leaf.end_id {
                            let cat = self.leaves.as_slice::<u32>().unwrap()[vote * 2];
                            let contrib = self.leaves.as_slice::<u32>().unwrap()[vote * 2 + 1];
                            let contrib = f32::from_bits(contrib);
                            writeln!(f, "{n} categ:{cat} add:{contrib}")?;
                        }
                    } else {
                        writeln!(f, "{} {:?}", n, self.get_unchecked(n as _))?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl TreeEnsembleData {
    unsafe fn get_unchecked(&self, node: usize) -> TreeNode {
        let row = &self.nodes.as_slice_unchecked::<u32>()[node * 5..][..5];
        if let Ok(cmp) = ((row[4] & 0xFF) as u8).try_into() {
            let feature_id = row[0];
            let true_id = row[1];
            let false_id = row[2];
            let value = f32::from_bits(row[3]);
            let nan_is_true = (row[4] & 0x0100) != 0;
            TreeNode::Branch(BranchNode { cmp, feature_id, value, true_id, false_id, nan_is_true })
        } else {
            TreeNode::Leaf(LeafNode { start_id: row[0] as usize, end_id: row[1] as usize })
        }
    }

    unsafe fn get_leaf_unchecked<T>(&self, tree: usize, input: &ArrayView1<T>) -> LeafNode
    where
        T: AsPrimitive<f32>,
    {
        let mut node_id = self.trees.as_slice_unchecked::<u32>()[tree] as usize;
        loop {
            let node = self.get_unchecked(node_id);
            match node {
                TreeNode::Branch(ref b) => {
                    let feature = *input.uget(b.feature_id as usize);
                    node_id = b.get_child_id(feature.as_());
                }
                TreeNode::Leaf(l) => return l,
            }
        }
    }

    unsafe fn eval_unchecked<A, T>(
        &self,
        tree: usize,
        input: &ArrayView1<T>,
        output: &mut ArrayViewMut1<f32>,
        aggs: &mut [A],
    ) where
        A: AggregateFn,
        T: AsPrimitive<f32>,
    {
        let leaf = self.get_leaf_unchecked(tree, input);
        for leaf in self
            .leaves
            .to_array_view_unchecked::<u32>()
            .outer_iter()
            .skip(leaf.start_id)
            .take(leaf.end_id - leaf.start_id)
        {
            let class_id = leaf[0] as usize;
            let weight = f32::from_bits(leaf[1]);
            let agg_fn = aggs.get_unchecked_mut(class_id);
            agg_fn.aggregate(weight, output.uget_mut(class_id));
        }
    }
}

#[derive(Copy, Clone)]
struct BranchNode {
    pub cmp: Cmp, // TODO: perf: most real forests have only 1 type of comparison
    pub feature_id: u32,
    pub value: f32,
    pub true_id: u32,
    pub false_id: u32,
    pub nan_is_true: bool,
}

impl std::fmt::Debug for BranchNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "if feat({}) {} {} then {} else {}",
            self.feature_id, self.cmp, self.value, self.true_id, self.false_id
        )
    }
}

impl BranchNode {
    pub fn get_child_id(&self, feature: f32) -> usize {
        let condition =
            if feature.is_nan() { self.nan_is_true } else { self.cmp.compare(feature, self.value) };
        if condition {
            self.true_id as usize
        } else {
            self.false_id as usize
        }
    }
}

#[derive(Copy, Clone, Debug, Hash)]
struct LeafNode {
    pub start_id: usize,
    pub end_id: usize,
}

#[derive(Copy, Clone, Debug)]
enum TreeNode {
    Branch(BranchNode),
    Leaf(LeafNode),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Aggregate {
    #[default]
    Sum,
    Avg,
    Max,
    Min,
}

#[derive(Clone, Debug, Hash)]
pub struct TreeEnsemble {
    pub data: TreeEnsembleData,
    pub max_used_feature: usize,
    pub n_classes: usize,
    pub aggregate_fn: Aggregate, // TODO: should this be an argument to eval()?
}

impl TreeEnsemble {
    pub fn build(
        data: TreeEnsembleData,
        max_used_feature: usize,
        n_classes: usize,
        aggregate_fn: Aggregate,
    ) -> TractResult<Self> {
        Ok(Self { data, max_used_feature, n_classes, aggregate_fn })
    }

    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    unsafe fn eval_one_unchecked<A, T>(
        &self,
        input: &ArrayView1<T>,
        output: &mut ArrayViewMut1<f32>,
        aggs: &mut [A],
    ) where
        A: AggregateFn,
        T: AsPrimitive<f32>,
    {
        for t in 0..self.data.trees.len() {
            self.data.eval_unchecked(t, input, output, aggs)
        }
        for i in 0..self.n_classes {
            aggs.get_unchecked_mut(i).post_aggregate(output.uget_mut(i));
        }
    }

    pub fn check_n_features(&self, n_features: usize) -> TractResult<()> {
        ensure!(
            n_features > self.max_used_feature,
            "Invalid input shape: input has {} features, tree ensemble use feature #{}",
            n_features,
            self.max_used_feature
        );
        Ok(())
    }

    fn eval_2d<A, T>(&self, input: &ArrayView2<T>) -> TractResult<Array2<f32>>
    where
        A: AggregateFn,
        T: AsPrimitive<f32>,
    {
        self.check_n_features(input.shape()[1])?;
        let n = input.shape()[0];
        let mut output = Array2::zeros((n, self.n_classes));
        let mut aggs: tract_smallvec::SmallVec<[A; 16]> =
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

    fn eval_1d<A, T>(&self, input: &ArrayView1<T>) -> TractResult<Array1<f32>>
    where
        A: AggregateFn,
        T: AsPrimitive<f32>,
    {
        self.check_n_features(input.len())?;
        let mut output = Array1::zeros(self.n_classes);
        let mut aggs: tract_smallvec::SmallVec<[A; 16]> =
            iter::repeat_with(Default::default).take(self.n_classes).collect();
        unsafe {
            self.eval_one_unchecked::<A, T>(input, &mut output.view_mut(), &mut aggs);
        }
        Ok(output)
    }

    pub fn eval<'i, I, T>(&self, input: I) -> TractResult<ArrayD<f32>>
    where
        I: Into<ArrayViewD<'i, T>>, // TODO: accept generic dimensions, not just IxDyn
        T: Datum + AsPrimitive<f32>,
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
    use tract_ndarray::prelude::*;

    fn b(
        node_offset: usize,
        cmp: Cmp,
        feat: usize,
        v: f32,
        left: usize,
        right: usize,
        nan_is_true: bool,
    ) -> [u32; 5] {
        [
            feat as u32,
            (node_offset + left) as u32,
            (node_offset + right) as u32,
            v.to_bits(),
            cmp as u32 | if nan_is_true { 0x100 } else { 0 },
        ]
    }

    fn l(leaf_offset: usize, start_id: usize, end_id: usize) -> [u32; 5] {
        [(leaf_offset + start_id) as u32, (leaf_offset + end_id) as u32, 0, 0, 0]
    }

    fn w(categ: usize, weight: f32) -> [u32; 2] {
        [categ as u32, weight.to_bits()]
    }

    fn generate_gbm_trees() -> TreeEnsembleData {
        let trees = rctensor1(&[0u32, 5u32, 14, 21, 30, 41]);
        let nodes = rctensor2(&[
            b(0, Cmp::LessEqual, 2, 3.15, 1, 2, true),
            b(0, Cmp::LessEqual, 1, 3.35, 3, 4, true),
            l(0, 0, 1),
            l(0, 1, 2),
            l(0, 2, 3),
            //
            b(5, Cmp::LessEqual, 2, 1.8, 1, 2, true),
            l(3, 0, 1),
            b(5, Cmp::LessEqual, 3, 1.65, 3, 4, true),
            b(5, Cmp::LessEqual, 2, 4.45, 5, 6, true),
            b(5, Cmp::LessEqual, 2, 5.35, 7, 8, true),
            l(3, 1, 2),
            l(3, 2, 3),
            l(3, 3, 4),
            l(3, 4, 5),
            //
            b(14, Cmp::LessEqual, 3, 1.65, 1, 2, true),
            b(14, Cmp::LessEqual, 2, 4.45, 3, 4, true),
            b(14, Cmp::LessEqual, 2, 5.35, 5, 6, true),
            l(8, 0, 1),
            l(8, 1, 2),
            l(8, 2, 3),
            l(8, 3, 4),
            //
            b(21, Cmp::LessEqual, 2, 3.15, 1, 2, true),
            b(21, Cmp::LessEqual, 1, 3.35, 3, 4, true),
            b(21, Cmp::LessEqual, 2, 4.45, 5, 6, true),
            l(12, 0, 1),
            l(12, 1, 2),
            l(12, 2, 3),
            b(21, Cmp::LessEqual, 2, 5.35, 7, 8, true),
            l(12, 3, 4),
            l(12, 4, 5),
            //
            b(30, Cmp::LessEqual, 3, 0.45, 1, 2, true),
            b(30, Cmp::LessEqual, 2, 1.45, 3, 4, true),
            b(30, Cmp::LessEqual, 3, 1.65, 5, 6, true),
            l(17, 0, 1),
            l(17, 1, 2),
            b(30, Cmp::LessEqual, 2, 4.45, 7, 8, true),
            b(30, Cmp::LessEqual, 2, 5.35, 9, 10, true),
            l(17, 2, 3),
            l(17, 3, 4),
            l(17, 4, 5),
            l(17, 5, 6),
            //
            b(41, Cmp::LessEqual, 2, 4.75, 1, 2, true),
            b(41, Cmp::LessEqual, 1, 2.75, 3, 4, true),
            b(41, Cmp::LessEqual, 2, 5.15, 7, 8, true),
            l(23, 0, 1),
            b(41, Cmp::LessEqual, 2, 4.15, 5, 6, true),
            l(23, 1, 2),
            l(23, 2, 3),
            l(23, 3, 4),
            l(23, 4, 5),
        ]);
        assert_eq!(nodes.shape(), &[50, 5]);
        let leaves = rctensor2(&[
            w(0, -0.075),
            w(0, 0.13928571),
            w(0, 0.15),
            //
            w(1, -0.075),
            w(1, 0.13548388),
            w(1, 0.110869564),
            w(1, -0.052500002),
            w(1, -0.075),
            //
            w(2, -0.075),
            w(2, -0.035869565),
            w(2, 0.1275),
            w(2, 0.15),
            //
            w(0, 0.12105576),
            w(0, 0.1304589),
            w(0, -0.07237862),
            w(0, -0.07226522),
            w(0, -0.07220469),
            //
            w(1, -0.07226842),
            w(1, -0.07268012),
            w(1, 0.119391434),
            w(1, 0.097440675),
            w(1, -0.049815115),
            w(1, -0.07219931),
            //
            w(2, -0.061642267),
            w(2, -0.0721846),
            w(2, -0.07319043),
            w(2, 0.076814815),
            w(2, 0.1315959),
        ]);
        assert_eq!(leaves.shape(), &[28, 2]);
        TreeEnsembleData { nodes, trees, leaves }
    }

    fn generate_gbm_ensemble() -> TreeEnsemble {
        // converted manually from LightGBM, fitted on iris dataset
        let trees = generate_gbm_trees();
        TreeEnsemble::build(trees, 4, 3, Aggregate::Sum).unwrap()
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
            [-0.14737862, 0.254_875_3, -0.13664228],
            [-0.14726523, -0.10231511, 0.20431481],
            [-0.14737862, 0.254_875_3, -0.13664228],
            [-0.14737862, 0.254_875_3, -0.13664228],
            [-0.147_204_7, -0.147_199_3, 0.281_595_9],
            [-0.14726523, -0.10231511, 0.20431481],
            [-0.147_204_7, -0.147_199_3, 0.281_595_9],
            [-0.147_204_7, -0.147_199_3, 0.281_595_9],
            [-0.147_204_7, -0.147_199_3, 0.281_595_9],
        ])
    }

    #[test]
    #[ignore]
    fn test_tree_ensemble() {
        let ensemble = generate_gbm_ensemble();
        let input = generate_gbm_input();
        let output = ensemble.eval(input.view().into_dyn()).unwrap();
        assert_eq!(output, generate_gbm_raw_output().into_dyn());
    }
}
