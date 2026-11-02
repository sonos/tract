use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use tract_core::internal::*;
use tract_core::ops::konst::Const;
use tract_core::tract_linalg::block_quant::BlockQuant;
use tract_core::tract_linalg::block_quant::BlockQuantFact;
use tract_core::tract_linalg::block_quant::BlockQuantStorage;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_transformers::ops::moe_ffn::RoutedInputMode;
use tract_transformers::ops::moe_ffn::RoutedQ40MatMul;

#[derive(Clone)]
struct BoundsCase {
    token: i64,
    expert: i64,
    constant: bool,
    mode: RoutedInputMode,
}

impl Test for BoundsCase {
    fn run_with_approx(
        &self,
        _: &'static str,
        runtime: &dyn Runtime,
        _: Approximation,
    ) -> TestResult {
        let rows = if self.mode == RoutedInputMode::TokenRows { 2 } else { 1 };
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([rows, 32]))?;
        let storage =
            BlockQuantStorage::new(Box::new(Q4_0), 4, 32, Arc::new(Q4_0.quant_f32(&[1f32; 128])?))?;
        let weights = model.wire_node(
            "weights",
            Const::new_with_exotic_fact(
                Arc::new(storage.into_tensor_with_shape(DatumType::F32, &[2, 2, 32])),
                Box::new(BlockQuantFact::new(Box::new(Q4_0), tvec![2, 2, 32])),
            )?,
            &[],
        )?[0];
        let mut inputs = tvec![Tensor::zero::<f32>(&[rows, 32])?.into_tvalue()];
        let mut ids = tvec![];
        for (name, id) in [("token", self.token), ("expert", self.expert)] {
            ids.push(if self.constant {
                model.add_const(name, tensor1(&[id]))?
            } else {
                inputs.push(tensor1(&[id]).into_tvalue());
                model.add_source(name, i64::fact([1]))?
            });
        }
        let output = model.wire_node(
            "routed",
            RoutedQ40MatMul { input_mode: self.mode },
            &[x, weights, ids[0], ids[1]],
        )?;
        model.select_output_outlets(&output)?;
        let result = runtime.prepare(model).and_then(|plan| plan.run(inputs));
        let valid_token = self.mode == RoutedInputMode::RouteRows || (0..2).contains(&self.token);
        if valid_token && (0..2).contains(&self.expert) {
            result?[0].close_enough(&Tensor::zero::<f32>(&[1, 2])?, Approximation::Exact)?;
        } else {
            ensure!(result.is_err(), "out-of-range route was accepted");
        }
        Ok(())
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    for constant in [false, true] {
        for mode in [RoutedInputMode::TokenRows, RoutedInputMode::RouteRows] {
            for (token, expert) in [(0, 0), (-1, 0), (2, 0), (0, -1), (0, 2)] {
                suite.add(
                    format!("{mode:?}_constant{constant}_token{token}_expert{expert}")
                        .replace('-', "neg"),
                    BoundsCase { token, expert, constant, mode },
                );
            }
        }
    }
    Ok(suite)
}
