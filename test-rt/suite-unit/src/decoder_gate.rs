use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use tract_core::internal::*;
use tract_core::ops::array::Slice;
use tract_core::ops::array::TypedConcat;
use tract_core::ops::math::mul;
use tract_core::ops::math::neg;
use tract_core::ops::nn::sigmoid;

#[derive(Debug, Clone)]
struct DecoderGate {
    width: usize,
    public_reset: bool,
}

impl Test for DecoderGate {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let c = 3;
        let shape = [1, c, 1, self.width];
        let gates_shape = [1, 2 * c, 1, self.width];
        let mut model = TypedModel::default();
        let logits = model.add_source("logits", f16::fact(gates_shape))?;
        let feature = model.add_source("feature-input", f16::fact(shape))?;
        let state = model.add_source("state-input", f16::fact(shape))?;
        let logits = model.wire_node("logits-neg", neg(), &[logits])?[0];
        let feature = model.wire_node("feature", neg(), &[feature])?[0];
        let state = model.wire_node("state", neg(), &[state])?[0];
        let lo = model.wire_node("reset-logits", Slice::new(1, 0, c), &[logits])?[0];
        let hi = model.wire_node("update-logits", Slice::new(1, c, 2 * c), &[logits])?[0];
        let reset = model.wire_node("reset", sigmoid(), &[lo])?[0];
        let update = model.wire_node("update", sigmoid(), &[hi])?[0];
        let gated = model.wire_node("apply-reset", mul(), &[reset, state])?[0];
        let joined = model.wire_node("joined", TypedConcat::new(1), &[feature, gated])?[0];
        let mut outputs = tvec![joined, update];
        if self.public_reset {
            outputs.push(reset);
        }
        model.select_output_outlets(&outputs)?;
        model.properties.insert("tract-rt-test.id".into(), rctensor0(id.to_string()));

        let len = c * self.width;
        let logits: Vec<_> =
            (0..2 * len).map(|i| f16::from_f32(((i * 7) % 29) as f32 / 4.0 - 3.5)).collect();
        let features: Vec<_> = (0..len).map(|i| f16::from_f32(i as f32 / 4.0 - 1.0)).collect();
        let states: Vec<_> =
            (0..len).map(|i| f16::from_f32(((i * 5) % 11) as f32 / 4.0 - 1.25)).collect();
        let gates: Vec<_> =
            logits.iter().map(|x| f16::from_f32(1.0 / (1.0 + x.to_f32().exp()))).collect();
        let expected: Vec<_> = features
            .iter()
            .map(|x| -*x)
            .chain(gates.iter().zip(&states).map(|(g, s)| *g * -*s))
            .collect();
        let runnable = runtime.prepare(model)?;
        if matches!(
            runtime.name().as_ref(),
            "wgpu_translate" | "optimized_wgpu" | "optimized_wgpu_planned"
        ) {
            let graph = runnable.typed_model().context("Decoder gate test needs a typed graph")?;
            let count =
                |name| graph.nodes().iter().filter(|node| node.op.name().as_ref() == name).count();
            let packed = !self.public_reset && len.is_multiple_of(4);
            ensure!(
                count("WgpuDecoderSigmoid") == usize::from(packed),
                "unexpected sigmoid motion (expected {packed})"
            );
            ensure!(
                count("WgpuPackedDecoderGate") == usize::from(packed),
                "unexpected packed reset gate selection (expected {packed})"
            );
        }
        let result = runnable.run(tvec![
            Tensor::from_shape(&gates_shape, &logits)?.into(),
            Tensor::from_shape(&shape, &features)?.into(),
            Tensor::from_shape(&shape, &states)?.into(),
        ])?;
        ensure!(result.len() == outputs.len());
        result[0].close_enough(&Tensor::from_shape(&gates_shape, &expected)?, approx)?;
        result[1].close_enough(&Tensor::from_shape(&shape, &gates[len..])?, approx)?;
        if self.public_reset {
            result[2].close_enough(&Tensor::from_shape(&shape, &gates[..len])?, approx)?;
        }
        Ok(())
    }
}

pub fn suite() -> TestSuite {
    let mut suite = TestSuite::default();
    suite.add("aligned", DecoderGate { width: 4, public_reset: false });
    suite.add("public_reset", DecoderGate { width: 4, public_reset: true });
    suite.add("unaligned", DecoderGate { width: 5, public_reset: false });
    suite
}
