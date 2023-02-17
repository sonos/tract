use super::*;

use proptest::prelude::*;
use proptest::strategy::{BoxedStrategy, Strategy};

fn tensor(shape: &[usize]) -> BoxedStrategy<Tensor> {
    let len = shape.iter().product::<usize>();
    let shape = shape.to_vec();
    proptest::collection::vec((0..5i8).prop_map(|i| i as f32), len..=len)
        .prop_map(move |vec| ArrayD::from_shape_vec(shape.clone(), vec).unwrap().into_tensor())
        .boxed()
}

fn strat_for_b_axes(rank: usize) -> impl Strategy<Value = (usize, usize)> {
    assert!(rank >= 2);
    (0..rank)
        .prop_flat_map(move |bn| (Just(bn), (0..rank - 1)))
        .prop_map(|(bn, raw_bk)| (bn, raw_bk + (raw_bk >= bn) as usize))
}

#[derive(Clone, Debug)]
struct ChangeAxisMatmulProblem {
    input: Tensor,
    change: AxisOp,
    matmul: MatMulUnary,
}

impl Arbitrary for ChangeAxisMatmulProblem {
    type Parameters = bool; // only use Moves
    type Strategy = BoxedStrategy<ChangeAxisMatmulProblem>;
    fn arbitrary_with(only_moves: bool) -> Self::Strategy {
        proptest::collection::vec(1..6usize, 2..5)
            .prop_flat_map(move |shape_input| {
                let op = if only_moves {
                    arbitrary_move_axis_for_rank(shape_input.len()).boxed()
                } else {
                    AxisOp::arbitrary_with(shape_input.clone().into()).boxed()
                };
                (tensor(&shape_input), op)
            })
            .prop_flat_map(|(input, change)| {
                let mut matmul_input_shape: TVec<usize> = input.shape().into();
                change.change_shape_array(&mut matmul_input_shape, false).unwrap();
                (Just(input), Just(change), Just(matmul_input_shape))
            })
            .prop_filter("rank must be >= 2", |(_, _, matmul_input_shape)| {
                matmul_input_shape.len() >= 2
            })
            .prop_flat_map(|(input, change, matmul_input_shape)| {
                (
                    Just(input),
                    Just(change),
                    Just(matmul_input_shape.clone()),
                    strat_for_b_axes(matmul_input_shape.len()),
                    1usize..=6,
                )
            })
            .prop_flat_map(|(input, change, matmul_input_shape, (b_k, b_n), m)| {
                let k = matmul_input_shape[b_k];
                (Just((input, change, matmul_input_shape, b_k, b_n)), tensor(&[m, k]))
            })
            .prop_map(|((input, change, matmul_input_shape, b_k, b_n), a)| {
                let mut axes = MatMulAxes::default_for_rank(matmul_input_shape.len());
                let change = change.canonical().into_owned();
                axes.b_n = b_n;
                axes.b_k = b_k;
                axes.c_m = b_k;
                axes.c_n = b_n;
                ChangeAxisMatmulProblem {
                    input,
                    change,
                    matmul: MatMulUnary {
                        a: a.broadcast_into_rank(matmul_input_shape.len())
                            .unwrap()
                            .into_arc_tensor(),
                        axes,
                    },
                }
            })
            .boxed()
    }
}

impl ChangeAxisMatmulProblem {
    fn model(&self) -> TypedModel {
        let mut model = TypedModel::default();
        let source = model.add_source("source", f32::fact(self.input.shape())).unwrap();
        let changed = model.wire_node("change", self.change.clone(), &[source]).unwrap();
        let output = model.wire_node("mm", self.matmul.clone(), &changed).unwrap();
        model.set_output_outlets(&output).unwrap();
        model
    }

    fn reference(&self) -> Tensor {
        let model = self.model();
        let mut outputs = model.into_runnable().unwrap().run(tvec!(self.input.clone().into())).unwrap();
        outputs.remove(0).into_tensor()
    }

    fn assert_killed_change(&self) {
        let dec = self.model().into_decluttered().unwrap();
        assert_eq!(dec.nodes().len(), 2); // just source and matmul
        assert!(dec.node(1).op_is::<MatMulUnary>());
    }

    // if swapping operator fails (not swappable) we get no output here
    fn swapped(&self) -> Option<Tensor> {
        let model = self.model();
        self.matmul
            .change_axes(&model, &model.nodes[2], InOut::In(0), &self.change.recip())
            .unwrap()
            .map(|changed_mm| {
                let mut model = TypedModel::default();
                let source = model.add_source("source", f32::fact(self.input.shape())).unwrap();
                let mut wire = model
                    .wire_node(
                        "mm",
                        changed_mm.substitute_op.clone().unwrap_or(Box::new(self.matmul.clone())),
                        &[source],
                    )
                    .unwrap();
                if let Some(change_after) = changed_mm
                    .wire_changes
                    .iter()
                    .find(|(io, _change)| *io == InOut::Out(0))
                    .map(|(_io, change)| change)
                {
                    wire = model.wire_node("change", change_after.recip(), &wire).unwrap();
                }
                model.set_output_outlets(&wire).unwrap();
                let mut outputs =
                    model.into_runnable().unwrap().run(tvec!(self.input.clone().into())).unwrap();
                outputs.remove(0).into_tensor()
            })
    }
}

pub fn arbitrary_move_axis_for_rank(rank: usize) -> BoxedStrategy<AxisOp> {
    (0..rank, 0..rank - 1).prop_map(|(a, b)| AxisOp::Move(a, b + (b >= a) as usize)).boxed()
}

proptest! {
    #[test]
    fn proptest_validity(pb in any::<ChangeAxisMatmulProblem>()) {
        pb.reference();
    }

    #[test]
    fn proptest_equals(pb in any::<ChangeAxisMatmulProblem>()) {
        if let Some(swapped) = pb.swapped() {
            prop_assert_eq!(swapped, pb.reference());
        }
    }

    /* FIXME: consider this one if we get a super generic definition of matmul (~einsum)
    #[test]
    fn proptest_move_are_absorbed(pb in any_with::<ChangeAxisMatmulProblem>(true)) {
        pb.assert_killed_change()
    }
    */
}

#[test]
fn rm0() {
    let pb = ChangeAxisMatmulProblem {
        input: Tensor::zero::<f32>(&[3, 1, 1]).unwrap(),
        change: AxisOp::Rm(1),
        matmul: MatMulUnary {
            a: Tensor::zero::<f32>(&[1, 3]).unwrap().into_arc_tensor(),
            axes: MatMulAxes { a_m: 0, a_k: 1, b_k: 0, b_n: 1, c_m: 0, c_n: 1 },
        },
    };
    assert_eq!(pb.swapped().unwrap(), pb.reference());
}

#[test]
#[ignore] // induce permutation around k axis
fn rm1_0() {
    // 3,1,4,9 -> k=3,n=4,9 -> 9,m=1,n=4, 9 and 4 have changed positions
    let pb = ChangeAxisMatmulProblem {
        input: Tensor::zero::<f32>(&[3, 1, 4, 9]).unwrap(),
        change: AxisOp::Rm(1),
        matmul: MatMulUnary {
            a: Tensor::zero::<f32>(&[1, 1, 3]).unwrap().into_arc_tensor(),
            axes: MatMulAxes { a_m: 1, a_k: 2, b_k: 0, b_n: 1, c_m: 1, c_n: 2 },
        },
    };
    assert_eq!(pb.swapped().unwrap(), pb.reference());
}

#[test]
#[ignore] // induce permutation around k axis
fn rm1_1() {
    let pb = ChangeAxisMatmulProblem {
        input: Tensor::zero::<f32>(&[2, 1, 3, 5]).unwrap(),
        change: AxisOp::Rm(1),
        matmul: MatMulUnary {
            a: Tensor::zero::<f32>(&[1, 1, 2]).unwrap().into_arc_tensor(),
            axes: MatMulAxes { a_m: 1, a_k: 2, b_k: 0, b_n: 1, c_m: 1, c_n: 2 },
        },
    };
    assert_eq!(pb.swapped().unwrap(), pb.reference());
}

#[test]
fn add2() {
    let pb = ChangeAxisMatmulProblem {
        input: Tensor::zero::<f32>(&[5, 2]).unwrap(),
        change: AxisOp::Add(2),
        matmul: MatMulUnary {
            a: Tensor::zero::<f32>(&[1, 1, 5]).unwrap().into_arc_tensor(),
            axes: MatMulAxes { a_m: 1, a_k: 2, b_k: 0, b_n: 1, c_m: 1, c_n: 2 },
        },
    };
    assert_eq!(pb.swapped().unwrap(), pb.reference());
}

#[test]
fn reshape0() {
    let pb = ChangeAxisMatmulProblem {
        input: Tensor::zero::<f32>(&[3, 5, 2, 2]).unwrap(),
        change: AxisOp::Reshape(1, tvec!(5.into(), 2.into()), tvec!(10.into())),
        matmul: MatMulUnary {
            a: Tensor::zero::<f32>(&[1, 1, 3]).unwrap().into_arc_tensor(),
            axes: MatMulAxes { a_m: 1, a_k: 2, b_k: 0, b_n: 2, c_m: 1, c_n: 2 },
        },
    };
    assert_eq!(pb.swapped().unwrap(), pb.reference());
}

#[test]
fn move_kn_nk() {
    let pb = ChangeAxisMatmulProblem {
        input: Tensor::zero::<f32>(&[2, 2]).unwrap(),
        change: AxisOp::Move(1, 0),
        matmul: MatMulUnary {
            a: Tensor::zero::<f32>(&[1, 2]).unwrap().into_arc_tensor(),
            axes: MatMulAxes::default(),
        },
    };
    pb.assert_killed_change()
}

#[test]
fn move_nak_kna() {
    let pb = ChangeAxisMatmulProblem {
        input: Tensor::zero::<f32>(&[2, 5, 3]).unwrap(), // n a k
        change: AxisOp::Move(2, 0),                      // -> k n a
        matmul: MatMulUnary {
            a: Tensor::zero::<f32>(&[1, 1, 3]).unwrap().into_arc_tensor(),
            axes: MatMulAxes { a_m: 1, a_k: 2, b_k: 0, b_n: 1, c_m: 1, c_n: 2 },
        },
    };
    pb.assert_killed_change()
}

#[test]
fn move_01() {
    let pb = ChangeAxisMatmulProblem {
        input: Tensor::zero::<f32>(&[3, 2]).unwrap(),
        change: AxisOp::Move(0, 1),
        matmul: MatMulUnary {
            a: Tensor::zero::<f32>(&[1, 2]).unwrap().into_arc_tensor(),
            axes: MatMulAxes { a_m: 0, a_k: 1, b_k: 0, b_n: 1, c_m: 0, c_n: 1 },
        },
    };
    assert_eq!(pb.swapped().unwrap(), pb.reference());
}

#[test]
fn move_01_bis() {
    let pb = ChangeAxisMatmulProblem {
        input: Tensor::zero::<f32>(&[1, 5, 2, 5]).unwrap(),
        change: AxisOp::Move(0, 1),
        matmul: MatMulUnary {
            a: Tensor::zero::<f32>(&[1, 1, 1, 2]).unwrap().into_arc_tensor(),
            axes: MatMulAxes { a_m: 2, a_k: 3, b_k: 2, b_n: 3, c_m: 2, c_n: 3 },
        },
    };
    assert_eq!(pb.swapped().unwrap(), pb.reference());
}
