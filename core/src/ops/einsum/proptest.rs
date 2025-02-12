use std::fmt;

use crate::internal::*;
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;
use tract_ndarray::ArrayD;

use crate::axes::AxesMapping;

use super::EinSum;

#[derive(Clone)]
struct BinEinsumProblem {
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
    type Parameters = ();
    type Strategy = BoxedStrategy<BinEinsumProblem>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (1..3usize, 1..3usize, 0..3usize)
            .prop_map(|(m_axes, n_axes, iter_axes)| {
                let m_axes: String = ('a'..).take(m_axes).collect();
                let n_axes: String = ('g'..).take(n_axes).collect();
                let iter_axes: String = ('w'..).take(iter_axes).collect();
                let a_axes: Vec<char> = (m_axes.clone() + &iter_axes + "k").chars().collect();
                let b_axes: Vec<char> = (n_axes.clone() + &iter_axes + "k").chars().collect();
                let c_axes: Vec<char> = (m_axes + &n_axes + &iter_axes).chars().collect();
                (Just(a_axes), Just(b_axes), Just(c_axes))
            })
            .prop_flat_map(|(a, b, c)| (a.prop_shuffle(), b.prop_shuffle(), c.prop_shuffle()))
            .prop_map(|(a, b, c)| {
                let a: String = a.into_iter().collect();
                let b: String = b.into_iter().collect();
                let c: String = c.into_iter().collect();
                let expr: AxesMapping = format!("{a},{b}->{c}").parse().unwrap();
                expr
            })
            .prop_flat_map(|expr| {
                let dims = expr.iter_all_axes().count();
                (Just(expr), proptest::collection::vec(1..4usize, dims..=dims))
            })
            .prop_flat_map(|(expr, axis_dims)| {
                let shape_a: TVec<usize> = expr
                    .axes(InOut::In(0))
                    .map(|axis| expr.iter_all_axes().position(|x| x == axis).unwrap())
                    .map(|dim| axis_dims[dim])
                    .collect();
                let shape_b: TVec<usize> = expr
                    .axes(InOut::In(1))
                    .map(|axis| expr.iter_all_axes().position(|x| x == axis).unwrap())
                    .map(|dim| axis_dims[dim])
                    .collect();
                let shape_output: TVec<usize> = expr
                    .axes(InOut::Out(0))
                    .map(|axis| expr.iter_all_axes().position(|x| x == axis).unwrap())
                    .map(|dim| axis_dims[dim])
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
    fn check(&self) -> TractResult<()> {
        let mut model = TypedModel::default();
        let mut inputs = tvec!();
        let a = if self.a_constant {
            model.add_const("a", self.a.clone())?
        } else {
            inputs.push(self.a.clone().into_tvalue());
            model.add_source("a", TypedFact::shape_and_dt_of(&self.a))?
        };
        let b = if self.b_constant {
            model.add_const("b", self.b.clone())?
        } else {
            inputs.push(self.b.clone().into_tvalue());
            model.add_source("b", TypedFact::shape_and_dt_of(&self.b))?
        };
        let mut output = model.wire_node(
            "einsum",
            EinSum { axes: self.expr.clone(), operating_dt: f32::datum_type(), q_params: None },
            &[a, b],
        )?;
        if let Some(c) = &self.unicast_add_constant {
            let c = model.add_const("c", c.clone())?;
            output = model.wire_node("add", crate::ops::math::add(), &[output[0], c])?;
        }
        model.set_output_outlets(&output)?;
        model = model.into_decluttered()?;
        let expected = model.clone().into_runnable()?.run(inputs.clone())?.remove(0);
        let optimised = model.clone().into_optimized()?;
        //dbg!(&optimised);
        let found = optimised.into_runnable()?.run(inputs.clone())?.remove(0);
        found.close_enough(&expected, Approximation::Close)
    }
}

proptest::proptest! {
    #[test]
    fn prop(pb in any::<BinEinsumProblem>()) {
        pb.check().unwrap();
    }
}

#[test]
fn unicast_0() {
    BinEinsumProblem {
        expr: "ak,gk->ag".parse().unwrap(),
        a: Tensor::zero::<f32>(&[1, 2]).unwrap(),
        b: Tensor::zero::<f32>(&[1, 2]).unwrap(),
        a_constant: false,
        b_constant: false,
        unicast_add_constant: Some(Tensor::zero::<f32>(&[1, 1]).unwrap()),
    }
    .check()
    .unwrap()
}

#[test]
fn unicast_1() {
    BinEinsumProblem {
        expr: "ak,gk->ag".parse().unwrap(),
        a: Tensor::zero::<f32>(&[2, 1]).unwrap(),
        b: Tensor::zero::<f32>(&[2, 1]).unwrap(),
        a_constant: false,
        b_constant: false,
        unicast_add_constant: Some(tensor2(&[[0f32, 0.], [0., 1.]])),
    }
    .check()
    .unwrap()
}

#[test]
fn unicast_2() {
    BinEinsumProblem {
        expr: "abk,gk->abg".parse().unwrap(),
        a: Tensor::zero::<f32>(&[2, 2, 1]).unwrap(),
        b: Tensor::zero::<f32>(&[1, 1]).unwrap(),
        a_constant: false,
        b_constant: false,
        unicast_add_constant: Some(tensor3(&[[[0f32], [0.]], [[0.], [1.]]])),
    }
    .check()
    .unwrap()
}
