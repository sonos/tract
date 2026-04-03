use std::fmt;

use infra::{Test, TestResult, TestSuite};
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;
use tract_core::internal::*;
use tract_core::ndarray::Ix2;
use tract_core::ops::array::{Pad, PadMode};
use tract_core::ops::konst::Const;
use tract_core::tract_linalg::block_quant::{
    BlockQuant, BlockQuantFact, BlockQuantStorage, Q4_0, Q8_1,
};
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
        (1..10usize, 1..128usize, 1..10usize)
            .prop_flat_map(|(m, k, n)| {
                let a = mm_q40_tensor(&[m, k]);
                let b = mm_q40_tensor(&[n, k]);

                (a, b)
            })
            .prop_map(move |(a, b)| MatmulQ40Problem { a, b, weights_in_b: params.weights_in_b })
            .boxed()
    }
}

fn mm_q40_tensor(shape: &[usize]) -> BoxedStrategy<Tensor> {
    let len = shape.iter().product::<usize>();
    let shape: Vec<usize> = shape.into();
    proptest::collection::vec((-100i8..=100i8).prop_map(|i| i as f32 / 100f32), len..=len)
        .prop_map(move |vec| ArrayD::from_shape_vec(shape.clone(), vec).unwrap().into_tensor())
        .boxed()
}

impl MatmulQ40Problem {
    fn pad_tensor(a: &Tensor, k_axis: usize) -> TractResult<Tensor> {
        let (mn, k) = (a.shape()[1 - k_axis], a.shape()[k_axis]);
        let shape =
            if k_axis == 0 { [k.next_multiple_of(32), mn] } else { [mn, k.next_multiple_of(32)] };
        let mut padded_a = Tensor::zero::<f32>(&shape)?;
        {
            let mut padded_a_plain = padded_a.try_as_plain_mut()?;
            padded_a_plain
                .to_array_view_mut::<f32>()?
                .slice_axis_move(Axis(k_axis), (0..k).into())
                .assign(&a.to_plain_array_view::<f32>()?);
        };

        Ok(padded_a)
    }

    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();

        let padded_a = Self::pad_tensor(&self.a, 1)?;

        let quant_a = Q4_0.quant_f32(padded_a.try_as_plain()?.as_slice::<f32>()?)?;

        let m = padded_a.shape()[0];
        let k = padded_a.shape()[1];
        let bqs = BlockQuantStorage::new(Box::new(Q4_0), m, k, Arc::new(quant_a))?;
        let bqf = BlockQuantFact::new(Box::new(Q4_0), tvec!(1, m, k));
        let packed_a = Arc::new(bqs.into_tensor_with_shape(f32::datum_type(), &[1, m, k]));

        let a =
            model.wire_node("a", Const::new_with_exotic_fact(packed_a, Box::new(bqf))?, &[])?[0];
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
        // Block-quant tensor is rank 3 [G=1, rows, K]; add group dim to its axes
        let axes_str = if !self.weights_in_b { "gmk,nk->mn" } else { "mk,gnk->mn" };
        let output = model.wire_node(
            "einsum",
            EinSum { axes: axes_str.parse()?, operating_dt: f32::datum_type(), q_params: None },
            &inputs,
        )?;

        model.select_output_outlets(&output)?;
        //let test = model.node_by_name("einsum")?.op.as_op().downcast_ref::<EinSum>().unwrap();

        //let test1 = model.node_by_name("einsum")?.op.as_op().downcast_ref::<EinSum>().unwrap();
        //dbg!(&test1.axes);
        Ok(model)
    }

    fn reference(&self, simulate_q81_activation_loss: bool) -> TractResult<Tensor> {
        let padded_a = Self::pad_tensor(&self.a, 1)?;
        let quant_dequant_a = Q4_0.simulate_precision_loss(padded_a, 1)?;

        // The GGML CUDA kernel internally quantizes activations to Q8_1.
        // When testing against such a runtime, the reference must account
        // for that additional precision loss.
        let quant_dequant_b = if simulate_q81_activation_loss {
            let padded_b = Self::pad_tensor(&self.b, 1)?;
            Q8_1.simulate_precision_loss(padded_b, 1)?
        } else {
            self.b.clone()
        };

        let mut a_view = quant_dequant_a
            .to_plain_array_view::<f32>()?
            .slice_axis_move(Axis(1), (0..self.a.shape()[1]).into());
        let mut b_view = quant_dequant_b
            .to_plain_array_view::<f32>()?
            .slice_axis_move(Axis(1), (0..self.b.shape()[1]).into());

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
        id: &str,
        runtime: &dyn Runtime,
        _approx: Approximation,
    ) -> TestResult {
        let uses_q81_activations = runtime.name().contains("cuda");
        let reference = self.reference(uses_q81_activations)?;
        //dbg!(&reference);
        let mut model = self.tract()?;

        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let mut inputs = tvec![];

        inputs.push(self.b.clone().into());

        let mut output = runtime.prepare(model)?.run(inputs)?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, Approximation::SuperApproximate)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    suite.add_arbitrary::<MatmulQ40Problem>("proptest", MatmulQ40ProblemParams::default());
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

    // Reduced from proptest — k=87 (not a multiple of 32) triggers Q8_1
    // activation quantization mismatch between CUDA GGML kernel and CPU reference.
    suite.add("proptest_reduced_k87", {
        #[rustfmt::skip]
        let a_row2: &[f32] = &[
            -0.69, -0.19, 0.0, 0.07, 0.0, 0.19, 0.0, -0.19, -0.94, 0.0,
            0.82, 0.32, 0.0, 0.32, -0.07, -0.07, 0.69, -0.07, -0.98, 0.19,
            0.56, -0.56, 0.0, -0.68, 0.68, -0.19, -0.07, 0.19, -0.07, -0.19,
            0.8, -0.56, 0.57, -0.07, 0.19, 0.82, -0.32, -0.32, 0.0, 0.07,
            0.0, 0.0, -0.82, 0.07, 0.0, -0.44, 0.44, 0.32, 0.07, 0.57,
            0.57, 0.0, 0.0, 0.57, 0.44, -0.07, 0.0, 0.0, 0.82, 0.69,
            0.32, -0.82, 0.44, 0.99, 0.18, 0.42, 0.66, 0.3, 0.0, 0.66,
            0.78, 0.0, -0.43, 0.18, 0.3, 0.0, 0.0, 0.78, -0.43, 0.66,
            0.0, -0.78, 0.0, -0.95, 0.18, 0.66, 0.3,
        ];
        let mut a_data = vec![0f32; 3 * 87];
        a_data[2 * 87..].copy_from_slice(a_row2);
        let a = Tensor::from_shape(&[3, 87], &a_data).unwrap();

        #[rustfmt::skip]
        let b_row1: &[f32] = &[
            -0.34, 0.54, -0.3, 0.32, -0.3, -0.2, 0.0, -0.08, -1.0, -0.04,
            0.91, 0.43, 0.0, 0.65, -0.34, -0.46, 0.76, 0.0, 0.0, 0.0,
            -0.18, 0.65, 0.1, 0.58, -0.98, 0.54, 0.06, 0.0, 0.0, 0.17,
            -0.71, 0.0, 0.0, 0.0, 0.0, 0.97, 0.0, 0.21, 0.07, -0.01,
            -0.54, -0.12, 0.0, 0.0, 0.2, 0.21, -0.6, -0.09, -0.15, -0.41,
            0.0, 0.0, 0.0, -0.25, 0.0, 0.0, 0.0, 0.0, -0.54, -0.8,
            0.0, -0.28, 0.0, -0.71, 0.98, -0.21, 0.0, 0.0, 0.0, -0.61,
            -0.32, 0.19, 0.16, 0.16, 0.0, 0.0, 0.0, 0.06, 0.36, 0.09,
            0.06, -0.4, -0.04, -0.51, 0.09, 0.73, 0.02,
        ];
        let mut b_data = vec![0f32; 7 * 87];
        b_data[87..2 * 87].copy_from_slice(b_row1);
        let b = Tensor::from_shape(&[7, 87], &b_data).unwrap();

        MatmulQ40Problem { a, b, weights_in_b: false }
    });

    Ok(suite)
}
