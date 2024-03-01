use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ops::quant::{offset_i8_as_u8, offset_u8_as_i8};

use crate::q_helpers::qtensor;

#[derive(Debug, Clone, Default)]
struct QFlavoursProblem {
    input: Tensor,
}

impl Arbitrary for QFlavoursProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<QFlavoursProblem>;
    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        ((1..20usize), any::<bool>())
            .prop_flat_map(|(len, signed)| {
                let dt = if signed { DatumType::I8 } else { DatumType::U8 };
                (Just(dt), qtensor(vec![1], dt), qtensor(vec![len], dt))
            })
            .prop_map(|(dt, zp, values)| {
                let zp = zp.into_tensor().cast_to_scalar::<i32>().unwrap();
                let mut values = values.into_tensor();
                let dt = dt.quantize(QParams::ZpScale { zero_point: zp, scale: 1f32 });
                unsafe {
                    values.set_datum_type(dt);
                }
                QFlavoursProblem { input: values }
            })
            .boxed()
    }
}

impl Test for QFlavoursProblem {
    fn run_with_approx(
        &self,
        _suite: &str,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let wire = model.add_source("input", TypedFact::shape_and_dt_of(&self.input))?;
        let output = if self.input.datum_type().is_signed() {
            model.wire_node("flavour", offset_i8_as_u8(), &[wire])?
        } else {
            model.wire_node("flavour", offset_u8_as_i8(), &[wire])?
        };
        model.set_output_outlets(&output)?;
        let output = runtime
            .prepare(model)?
            .run(tvec![self.input.clone().into_tvalue()])?
            .remove(0)
            .into_tensor();
        dbg!(&output);
        let reference = self.input.cast_to::<f32>()?;
        let comparison = output.cast_to::<f32>()?;
        comparison.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<QFlavoursProblem>("proptest", ());
    suite.add(
        "trivial_0",
        QFlavoursProblem {
            input: tensor0(0u8)
                .cast_to_dt(
                    u8::datum_type().quantize(QParams::ZpScale { zero_point: 0, scale: 1. }),
                )
                .unwrap()
                .into_owned(),
        },
    );
    Ok(suite)
}
