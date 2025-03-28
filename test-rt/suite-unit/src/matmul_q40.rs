use std::fmt;

use infra::{Test, TestResult, TestSuite};
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;
use tract_core::internal::*;
use tract_core::ndarray::Ix2;
use tract_core::ops::array::{Pad, PadMode};
use tract_core::ops::konst::Const;
use tract_core::tract_linalg::block_quant::{BlockQuant, BlockQuantFact, BlockQuantValue, Q4_0};
use tract_ndarray::{ArrayD, Axis};

use tract_core::ops::einsum::EinSum;

#[derive(Debug, Clone, Default)]
pub struct MatmulQ40ProblemParams {
    weights_in_b: bool,
}

#[derive(Clone)]
pub struct MatmulQ40Problem {
    a: Tensor,
    b: Tensor,
    weights_in_b: bool,
}

impl std::fmt::Debug for MatmulQ40Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a:{:?} b:{:?} weights_in_b:{:?}", self.a, self.b, self.weights_in_b)
    }
}

impl Arbitrary for MatmulQ40Problem {
    type Parameters = MatmulQ40ProblemParams;
    type Strategy = BoxedStrategy<MatmulQ40Problem>;

    fn arbitrary_with(params: Self::Parameters) -> Self::Strategy {
        (1..20usize, 1..20usize, 1..20usize)
            .prop_flat_map(|(m, k, n)| {
                let a = tensor(&[m, k]);
                let b = tensor(&[n, k]);

                (a, b)
            })
            .prop_map(move |(a, b)| MatmulQ40Problem { a, b, weights_in_b: params.weights_in_b })
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

impl MatmulQ40Problem {
    fn pad_tensor(a: &Tensor, k_axis: usize) -> TractResult<Tensor> {
        let (mn, k) = (a.shape()[1 - k_axis], a.shape()[k_axis]);
        let shape =
            if k_axis == 0 { [k.next_multiple_of(32), mn] } else { [mn, k.next_multiple_of(32)] };
        let mut padded_a = Tensor::zero::<f32>(&shape)?;
        padded_a
            .to_array_view_mut::<f32>()?
            .slice_axis_move(Axis(k_axis), (0..k).into())
            .assign(&a.to_array_view::<f32>()?);

        Ok(padded_a)
    }

    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();

        let padded_a = Self::pad_tensor(&self.a, 1)?;

        let quant_a = Q4_0.quant_f32(padded_a.as_slice::<f32>()?)?;

        let bqf = BlockQuantFact::new(Box::new(Q4_0), padded_a.shape().into());
        let bqv = BlockQuantValue { value: Arc::new(quant_a), fact: bqf.clone() };

        let opaque_a = tensor0(Opaque(Arc::new(bqv))).into_arc_tensor();

        let a =
            model.wire_node("a", Const::new_with_opaque_fact(opaque_a, Box::new(bqf))?, &[])?[0];
        let b = model.add_source("b", TypedFact::shape_and_dt_of(&self.b))?;

        let k = self.b.shape()[1];
        let padded_b = model.wire_node(
            "pad_b",
            Pad::new(
                vec![(0, 0), (0, k.next_multiple_of(32) - k)],
                PadMode::Constant(rctensor0(0f32)),
            ),
            &[b],
        )?[0];

        let inputs = if !self.weights_in_b { [a, padded_b] } else { [padded_b, a] };
        let output = model.wire_node(
            "einsum",
            EinSum { axes: "mk,nk->mn".parse()?, operating_dt: f32::datum_type(), q_params: None },
            &inputs,
        )?;

        model.set_output_outlets(&output)?;
        //let test = model.node_by_name("einsum")?.op.as_op().downcast_ref::<EinSum>().unwrap();

        //let test1 = model.node_by_name("einsum")?.op.as_op().downcast_ref::<EinSum>().unwrap();
        //dbg!(&test1.axes);
        Ok(model)
    }

    fn reference(&self) -> TractResult<Tensor> {
        let padded_a = Self::pad_tensor(&self.a, 1)?;
        let quant_dequant_a = Q4_0.simulate_precision_loss(padded_a, 1)?;

        let mut a_view = quant_dequant_a
            .to_array_view::<f32>()?
            .slice_axis_move(Axis(1), (0..self.a.shape()[1]).into());
        let mut b_view = self.b.to_array_view::<f32>()?;

        if self.weights_in_b {
            (a_view, b_view) = (b_view, a_view);
        }
        let c = a_view.into_dimensionality::<Ix2>()?.dot(&b_view.into_dimensionality::<Ix2>()?.t());
        Ok(c.into_tensor())
    }
}

impl Test for MatmulQ40Problem {
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference()?;
        //dbg!(&reference);
        let mut model = self.tract()?;

        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let mut inputs = tvec![];

        inputs.push(self.b.clone().into());

        let mut output = runtime.prepare(model)?.run(inputs)?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    suite.add_arbitrary::<MatmulQ40Problem>("proptest", MatmulQ40ProblemParams::default());
    suite.add_arbitrary::<MatmulQ40Problem>(
        "proptest_weights_in_b",
        MatmulQ40ProblemParams { weights_in_b: true },
    );

    suite.add_arbitrary::<MatmulQ40Problem>(
        "proptest_weights_in_b",
        MatmulQ40ProblemParams { weights_in_b: true },
    );

    suite.add(
        "minimal_inputs",
        MatmulQ40Problem { a: tensor2(&[[0f32]]), b: tensor2(&[[0f32]]), weights_in_b: false },
    );

    suite.add(
        "minimal_matvec",
        MatmulQ40Problem {
            a: tensor2(&[[-1f32]]),
            b: tensor2(&[[0f32], [-1f32]]),
            weights_in_b: false,
        },
    );

    suite.add(
        "minimal_matvec_weights_in_b_0",
        MatmulQ40Problem {
            a: tensor2(&[[0f32, 1f32]]),
            b: tensor2(&[[0f32, 1f32]]),
            weights_in_b: true,
        },
    );

    //  a:1,1,F32 0 b:1,1,F32 0
    suite.add(
        "minimal_matvec_weights_in_b_1",
        MatmulQ40Problem { a: tensor2(&[[0f32]]), b: tensor2(&[[0f32]]), weights_in_b: true },
    );

    suite.add(
        "minimal_matvec_weights_in_b_2",
        MatmulQ40Problem {
            a: tensor2(&[[0f32], [0f32]]),
            b: tensor2(&[[0f32]]),
            weights_in_b: true,
        },
    );

    Ok(suite)
}
