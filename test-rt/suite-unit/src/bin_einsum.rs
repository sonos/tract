use std::{fmt, ops::Mul};

use infra::{Test, TestResult, TestSuite};
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;
use tract_core::internal::*;
use tract_ndarray::{ArrayD, Axis, Dimension};

use tract_core::ops::einsum::EinSum;
use tract_num_traits::{One, Zero};

#[derive(Debug, Clone)]
pub struct BinEinsumProblemParams {
    pub force_unique_non_trivial_m_n: bool,
    pub no_trivial_axes: bool,
    pub force_max_one_iter_axis: bool,
    pub max_dims: usize,
}

impl Default for BinEinsumProblemParams {
    fn default() -> BinEinsumProblemParams {
        BinEinsumProblemParams {
            force_unique_non_trivial_m_n: false,
            no_trivial_axes: false,
            force_max_one_iter_axis: false,
            max_dims: 8,
        }
    }
}

#[derive(Clone)]
pub struct BinEinsumProblem {
    expr: AxesMapping,
    a: Tensor,
    b: Tensor,
    a_constant: bool,
    b_constant: bool,
    unicast_add_constant: Option<Tensor>,
}

impl std::fmt::Debug for BinEinsumProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} A:{:?} B:{:?} a_constant:{:?} b_constant:{:?} unicast_add_constant:{:?}",
            self.expr, self.a, self.b, self.a_constant, self.b_constant, self.unicast_add_constant
        )
    }
}

impl Arbitrary for BinEinsumProblem {
    type Parameters = BinEinsumProblemParams;
    type Strategy = BoxedStrategy<BinEinsumProblem>;

    fn arbitrary_with(params: Self::Parameters) -> Self::Strategy {
        let supp_m_n_axes_range =
            if params.force_unique_non_trivial_m_n { 0..1usize } else { 0..2usize };
        assert!(params.max_dims >= 3);
        let remaining = params.max_dims - 3; // At least 1 m, and n

        supp_m_n_axes_range
            .clone()
            .prop_flat_map(move |supp_m_axes| {
                let remaining = remaining - supp_m_axes;
                let n_axes_range = if remaining < supp_m_n_axes_range.end {
                    0..(remaining + 1)
                } else {
                    supp_m_n_axes_range.clone()
                };
                let iter_axes_range =
                    if params.force_max_one_iter_axis { 0..2usize } else { 0..3usize };
                n_axes_range.prop_flat_map(move |supp_n_axes| {
                    let remaining = remaining - supp_n_axes;
                    let iter_axes_range = if remaining < iter_axes_range.end {
                        0..(remaining + 1)
                    } else {
                        iter_axes_range.clone()
                    };
                    iter_axes_range.clone().prop_flat_map(move |iter_axes| {
                        let remaining = remaining - iter_axes;
                        let trivial_m_n_axes_range =
                            if params.no_trivial_axes { 0..1usize } else { 0..2usize };
                        let trivial_m_axes_range = if remaining < trivial_m_n_axes_range.end {
                            0..(remaining + 1)
                        } else {
                            trivial_m_n_axes_range.clone()
                        };
                        trivial_m_axes_range.clone().prop_flat_map(move |trivial_m_axes| {
                            let remaining = remaining - trivial_m_axes;
                            let trivial_n_axes_range = if remaining < trivial_m_n_axes_range.end {
                                0..(remaining + 1)
                            } else {
                                trivial_m_n_axes_range.clone()
                            };
                            trivial_n_axes_range.clone().prop_flat_map(move |trivial_n_axes| {
                                Just((
                                    supp_m_axes,
                                    supp_n_axes,
                                    iter_axes,
                                    trivial_m_axes,
                                    trivial_n_axes,
                                ))
                            })
                        })
                    })
                })
            })
            .prop_map(|(supp_m_axes, supp_n_axes, iter_axes, trivial_m_axes, trivial_n_axes)| {
                let m_axes: String = ('a'..).take(supp_m_axes).collect();
                let trivial_m_axes: String = ('e'..).take(trivial_m_axes).collect();
                let n_axes: String = ('h'..).take(supp_n_axes).collect();
                let trivial_n_axes: String = ('o'..).take(trivial_n_axes).collect();
                let iter_axes: String = ('w'..).take(iter_axes).collect();
                let a_axes: Vec<char> =
                    (m_axes.clone() + "m" + &trivial_m_axes + &iter_axes + "k").chars().collect();
                let b_axes: Vec<char> =
                    (n_axes.clone() + "n" + &trivial_n_axes + &iter_axes + "k").chars().collect();
                let c_axes: Vec<char> =
                    (m_axes + &n_axes + "mn" + &trivial_m_axes + &trivial_n_axes + &iter_axes)
                        .chars()
                        .collect();
                (Just(a_axes), Just(b_axes), Just(c_axes))
            })
            .prop_flat_map(|(a, b, c)| (a.prop_shuffle(), b.prop_shuffle(), c.prop_shuffle()))
            .prop_map(|(a, b, c)| {
                let a: String = a.into_iter().collect();
                let b: String = b.into_iter().collect();
                let c: String = c.into_iter().collect();
                let expr: AxesMapping = format!("{a},{b}->{c}").parse().unwrap();
                eprintln!("{expr}");
                expr
            })
            .prop_flat_map(|expr| {
                let dims = expr.iter_all_axes().count();
                (Just(expr), proptest::collection::vec(1..4usize, dims..=dims))
            })
            .prop_flat_map(|(expr, axis_dims)| {
                let shape_a: TVec<usize> = expr
                    .axes(InOut::In(0))
                    .map(|axis| {
                        expr.iter_all_axes()
                            .position(|x| (x == axis) && !('m'..='v').contains(&axis.repr))
                            .map(|dim| axis_dims[dim])
                            .unwrap_or(1)
                    })
                    .collect();
                let shape_b: TVec<usize> = expr
                    .axes(InOut::In(1))
                    .map(|axis| {
                        expr.iter_all_axes()
                            .position(|x| (x == axis) && !('m'..='v').contains(&axis.repr))
                            .map(|dim| axis_dims[dim])
                            .unwrap_or(1)
                    })
                    .collect();
                let shape_output: TVec<usize> = expr
                    .axes(InOut::Out(0))
                    .map(|axis| {
                        expr.iter_all_axes()
                            .position(|x| (x == axis) && !('m'..='v').contains(&axis.repr))
                            .map(|dim| axis_dims[dim])
                            .unwrap_or(1)
                    })
                    .collect();
                let unicast_add_constant = proptest::option::of(tensor(&shape_output));
                (Just(expr), tensor(&shape_a), tensor(&shape_b), 0..3usize, unicast_add_constant)
            })
            .prop_map(|(expr, a, b, a_b_constant, unicast_add_constant)| {
                let a_constant = (a_b_constant & 0x1) != 0;
                let b_constant = (a_b_constant & 0x2) != 0;
                BinEinsumProblem { expr, a, b, a_constant, b_constant, unicast_add_constant }
            })
            .boxed()
    }
}

pub fn tensor(shape: &[usize]) -> BoxedStrategy<Tensor> {
    let len = shape.iter().product::<usize>();
    let shape: Vec<usize> = shape.into();
    proptest::collection::vec((-10i8..=10i8).prop_map(|i| i as f32), len..=len)
        .prop_map(move |vec| ArrayD::from_shape_vec(shape.clone(), vec).unwrap().into_tensor())
        .boxed()
}

impl BinEinsumProblem {
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let a = if self.a_constant {
            model.add_const("a", self.a.clone())?
        } else {
            model.add_source("a", TypedFact::shape_and_dt_of(&self.a))?
        };
        let b = if self.b_constant {
            model.add_const("b", self.b.clone())?
        } else {
            model.add_source("b", TypedFact::shape_and_dt_of(&self.b))?
        };

        let mut output = model.wire_node(
            "einsum",
            EinSum { axes: self.expr.clone(), operating_dt: f32::datum_type(), q_params: None },
            &[a, b],
        )?;

        if let Some(c) = &self.unicast_add_constant {
            let c = model.add_const("c", c.clone())?;
            output = model.wire_node("add", tract_core::ops::math::add(), &[output[0], c])?;
        }

        model.set_output_outlets(&output)?;

        //let test = model.node_by_name("einsum")?.op.as_op().downcast_ref::<EinSum>().unwrap();

        model = model.into_decluttered()?;
        //let test1 = model.node_by_name("einsum")?.op.as_op().downcast_ref::<EinSum>().unwrap();
        //dbg!(&test1.axes);
        Ok(model)
    }

    fn output_shape(&self) -> TVec<usize> {
        self.expr
            .axes(InOut::Out(0))
            .map(|axis| {
                let dim_in_a = axis.inputs[0].first().map(|pos| self.a.shape()[*pos]).unwrap_or(1);
                let dim_in_b = axis.inputs[1].first().map(|pos| self.b.shape()[*pos]).unwrap_or(1);
                dim_in_a.max(dim_in_b)
            })
            .collect()
    }

    fn reference<Acc: Datum + Copy + Zero + One + Mul<Acc, Output = Acc>>(&self) -> ArrayD<Acc> {
        let output_shape = self.output_shape();

        let a = self.a.cast_to::<Acc>().unwrap();
        let b = self.b.cast_to::<Acc>().unwrap();

        let a = a.to_array_view::<Acc>().unwrap();
        let b = b.to_array_view::<Acc>().unwrap();

        let k_axes: TVec<_> = self
            .expr
            .iter_all_axes()
            .filter(|axis| {
                axis.outputs[0].is_empty() && axis.inputs[0].len() == 1 && axis.inputs[1].len() == 1
            })
            .collect();

        let summing_shape: TVec<usize> = k_axes
            .iter()
            .map(|axis| {
                let dim_in_a = axis.inputs[0].first().map(|pos| self.a.shape()[*pos]).unwrap_or(1);
                let dim_in_b = axis.inputs[1].first().map(|pos| self.b.shape()[*pos]).unwrap_or(1);
                dim_in_a.max(dim_in_b)
            })
            .collect();

        let output = tract_ndarray::ArrayD::<Acc>::from_shape_fn(&*output_shape, |coords| {
            let coords = coords.as_array_view();
            let mut a = a.clone();
            let mut b = b.clone();
            for (axis, x) in self.expr.axes(InOut::Out(0)).zip(coords.iter()) {
                if let Some(pos) = axis.inputs[0].first() {
                    a.collapse_axis(Axis(*pos), if a.shape()[*pos] > 1 { *x } else { 0 });
                }

                if let Some(pos) = axis.inputs[1].first() {
                    b.collapse_axis(Axis(*pos), if b.shape()[*pos] > 1 { *x } else { 0 });
                }
            }

            let mut sum: Acc = Acc::zero();
            for sum_coords in tract_ndarray::indices(&*summing_shape) {
                let mut a = a.clone();
                let mut b = b.clone();

                let sum_coords = sum_coords.as_array_view();
                for (axis, x) in k_axes.iter().zip(sum_coords) {
                    a.collapse_axis(Axis(axis.inputs[0][0]), *x);
                    b.collapse_axis(Axis(axis.inputs[1][0]), *x);
                }

                let product = *a.iter().next().unwrap() * *b.iter().next().unwrap();
                sum = sum + product;
            }
            sum
        });
        if let Some(unicast_const) = self.unicast_add_constant.clone() {
            output + unicast_const.into_array::<Acc>().unwrap()
        } else {
            output
        }
    }
}

impl Test for BinEinsumProblem {
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference::<f32>().into_tensor();
        //dbg!(&reference);
        let mut model = self.tract()?;

        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let mut inputs = tvec![];
        if !self.a_constant {
            inputs.push(self.a.clone().into());
        }
        if !self.b_constant {
            inputs.push(self.b.clone().into());
        }
        let mut output = runtime.prepare(model)?.run(inputs)?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    suite.add_arbitrary::<BinEinsumProblem>("proptest", BinEinsumProblemParams::default());

    suite.add(
        "unicast_0",
        BinEinsumProblem {
            expr: "ak,gk->ag".parse().unwrap(),
            a: Tensor::zero::<f32>(&[1, 2]).unwrap(),
            b: Tensor::zero::<f32>(&[1, 2]).unwrap(),
            a_constant: false,
            b_constant: false,
            unicast_add_constant: Some(Tensor::zero::<f32>(&[1, 1]).unwrap()),
        },
    );

    suite.add(
        "unicast_1",
        BinEinsumProblem {
            expr: "ak,gk->ag".parse().unwrap(),
            a: Tensor::zero::<f32>(&[2, 1]).unwrap(),
            b: Tensor::zero::<f32>(&[2, 1]).unwrap(),
            a_constant: false,
            b_constant: false,
            unicast_add_constant: Some(tensor2(&[[0f32, 0.], [0., 1.]])),
        },
    );

    suite.add(
        "unicast_2",
        BinEinsumProblem {
            expr: "abk,gk->abg".parse().unwrap(),
            a: Tensor::zero::<f32>(&[2, 2, 1]).unwrap(),
            b: Tensor::zero::<f32>(&[1, 1]).unwrap(),
            a_constant: false,
            b_constant: false,
            unicast_add_constant: Some(tensor3(&[[[0f32], [0.]], [[0.], [1.]]])),
        },
    );

    suite.add(
        "trivial_0",
        BinEinsumProblem {
            expr: "ak,gk->ag".parse().unwrap(),
            a: tensor2(&[[1f32]]),
            b: tensor2(&[[0f32], [1f32]]),
            a_constant: false,
            b_constant: false,
            unicast_add_constant: None,
        },
    );

    suite.add(
        "trivial_1",
        BinEinsumProblem {
            expr: "akb,gk->gba".parse().unwrap(),
            a: tensor3(&[[[0f32], [0f32]]]),
            b: tensor2(&[[0f32, 0f32]]),
            a_constant: true,
            b_constant: false,
            unicast_add_constant: None,
        },
    );

    suite.add(
        "supp_axis_bug_0",
        BinEinsumProblem {
            expr: "bmk, abkn->bn".parse().unwrap(),
            a: Tensor::zero::<f32>(&[32, 1, 25]).unwrap(),
            b: Tensor::zero::<f32>(&[1, 32, 25, 64]).unwrap(),
            a_constant: false,
            b_constant: false,
            unicast_add_constant: None,
        },
    );

    suite.add(
        "m_axis_select_bug_0",
        BinEinsumProblem {
            expr: "wmkx,wknx->xmnw".parse().unwrap(),
            a: tensor4(&[[[[0f32, 0f32], [0f32, -1f32]]]]),
            b: tensor4(&[[[[0f32, 0f32]], [[0f32, 1f32]]]]),
            a_constant: false,
            b_constant: false,
            unicast_add_constant: None,
        },
    );

    // TODO: fix ensure_mkn() to handle multiple n axes
    //suite.add(
    //    "multiple_n_axes",
    //    BinEinsumProblem {
    //        expr: "kwa,gkwh->gahw".parse().unwrap(),
    //        a: Tensor::zero::<f32>(&[1, 2, 1]).unwrap(),
    //        b: Tensor::zero::<f32>(&[2, 1, 2, 2]).unwrap(),
    //        a_constant: false,
    //        b_constant: false,
    //        unicast_add_constant: None,
    //    }
    //);
    Ok(suite)
}
