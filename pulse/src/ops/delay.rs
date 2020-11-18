use crate::internal::*;
use tract_pulse_opl::ops::Delay;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<Delay>(), ser_delay)
}

fn ser_delay(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op().downcast_ref::<Delay>().unwrap();
    let wire = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_pulse_delay",
        &[wire],
        &[
            ("axis", numeric(op.axis)),
            ("delay", numeric(op.delay)),
            ("overlap", numeric(op.overlap)),
        ],
    )))
}

impl PulsedOp for Delay {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape[self.axis] += self.overlap;
        fact.delay += self.delay + self.overlap;
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_pulse_delay_over(pulse: usize, delay: usize, overlap: usize) {
        let mut model = PulsedModel::default();
        let fact1 = PulsedFact {
            datum_type: u8::datum_type(),
            shape: tvec![pulse.to_dim()],
            axis: 0,
            dim: stream_dim(),
            delay: 0,
        };
        let source = model.add_source("source", fact1.clone()).unwrap();
        model
            .wire_node("delay", Delay::new(fact1.axis, &(&fact1).into(), delay, overlap), &[source])
            .unwrap();
        model.auto_outputs().unwrap();

        let plan = SimplePlan::new(model).unwrap();
        let mut state = tract_core::plan::SimpleState::new(plan).unwrap();

        for i in 0..5 {
            let input: Vec<u8> = (pulse * i..(pulse * (i + 1))).map(|a| a as u8).collect();
            let expect: Vec<u8> = (pulse * i..(pulse * (i + 1) + overlap))
                .map(|i| i.saturating_sub(delay + overlap) as u8)
                .collect();
            let output = state.run(tvec!(tensor1(&input).into())).unwrap();
            let skip = (delay + overlap).saturating_sub(i * pulse).min(pulse + overlap);
            assert_eq!(&output[0].as_slice::<u8>().unwrap()[skip..], &expect[skip..]);
        }
    }

    #[test]
    fn sub_pulse() {
        test_pulse_delay_over(4, 1, 0);
    }

    #[test]
    fn supra_pulse() {
        test_pulse_delay_over(4, 5, 0);
    }

    #[test]
    fn sub_pulse_context() {
        test_pulse_delay_over(4, 0, 2);
    }

    #[test]
    fn supra_pulse_context() {
        test_pulse_delay_over(4, 0, 6);
    }

    #[test]
    fn test_two_delays() {
        let pulse = 4usize;
        let mut model = PulsedModel::default();
        let fact_0 = PulsedFact {
            datum_type: u8::datum_type(),
            shape: tvec![pulse.to_dim()],
            axis: 0,
            dim: stream_dim(),
            delay: 0,
        };
        let source = model.add_source("source", fact_0.clone()).unwrap();
        let delay_1 = model
            .wire_node("delay-1", Delay::new(fact_0.axis, &(&fact_0).into(), 2, 0), &[source])
            .unwrap()[0];
        let fact_1 = model.outlet_fact(delay_1).unwrap().clone();
        let delay_2 = model
            .wire_node("delay-1", Delay::new(fact_1.axis, &(&fact_1).into(), 2, 0), &[delay_1])
            .unwrap();
        model.set_output_outlets(&delay_2).unwrap();

        let plan = SimplePlan::new(model).unwrap();
        let mut state = tract_core::plan::SimpleState::new(plan).unwrap();

        for i in 0..5 {
            let input: Vec<u8> = (pulse * i..(pulse * (i + 1))).map(|a| a as u8).collect();
            let expect: Vec<u8> =
                (pulse * i..(pulse * (i + 1))).map(|i| i.saturating_sub(4) as u8).collect();
            let skip = 4usize.saturating_sub(i * pulse).min(pulse);
            let output = state.run(tvec!(tensor1(&input).into())).unwrap();
            assert_eq!(&output[0].as_slice::<u8>().unwrap()[skip..], &expect[skip..]);
        }
    }
}
