use std::cmp::Ordering;

use infra::{Test, TestSuite};
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::ndarray::Axis;
use tract_core::ndarray::IxDyn;
use tract_core::ops::array::Topk;

/// Top-k along one axis, checked against a plain stable sort: the largest (or smallest)
/// values first, ties to the lowest index, NaN after every number. Values come from a
/// handful of levels plus NaN and infinities, so ties and NaNs are the common case.
#[derive(Debug, Clone)]
pub struct TopkProblem {
    pub input: ArrayD<f32>,
    pub axis: usize,
    pub k: usize,
    pub largest: bool,
    pub dt: DatumType,
    /// Feed both outputs to another op instead of making them the model's outputs: the
    /// values are negated, the indices cast to f32.
    pub consumed: bool,
}

fn level(ix: u8) -> f32 {
    match ix {
        0 => f32::NAN,
        1 => f32::INFINITY,
        2 => f32::NEG_INFINITY,
        n => n as f32 - 6.0,
    }
}

impl Arbitrary for TopkProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<TopkProblem>;
    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        vec(1usize..8, 1..4usize)
            .prop_flat_map(|shape| {
                let len = shape.iter().product::<usize>();
                let rank = shape.len();
                let dt = prop_oneof![Just(DatumType::F32), Just(DatumType::F16)];
                (Just(shape), vec(0u8..10, len..=len), 0..rank, any::<bool>(), dt)
            })
            .prop_flat_map(|(shape, levels, axis, largest, dt)| {
                let n = shape[axis];
                (Just(shape), Just(levels), Just(axis), 1..=n, Just(largest), Just(dt))
            })
            .prop_map(|(shape, levels, axis, k, largest, dt)| TopkProblem {
                input: ArrayD::from_shape_vec(
                    IxDyn(&shape),
                    levels.into_iter().map(level).collect(),
                )
                .unwrap(),
                axis,
                k,
                largest,
                dt,
                consumed: false,
            })
            .boxed()
    }
}

impl TopkProblem {
    fn reference(&self) -> (ArrayD<f32>, ArrayD<i64>) {
        let mut shape = self.input.shape().to_vec();
        shape[self.axis] = self.k;
        let mut values = ArrayD::<f32>::zeros(IxDyn(&shape));
        let mut indices = ArrayD::<i64>::zeros(IxDyn(&shape));
        let lanes = self.input.lanes(Axis(self.axis));
        let out_lanes = values.lanes_mut(Axis(self.axis)).into_iter();
        let ix_lanes = indices.lanes_mut(Axis(self.axis)).into_iter();
        for ((lane, mut v), mut i) in lanes.into_iter().zip(out_lanes).zip(ix_lanes) {
            let mut entries: Vec<(usize, f32)> = lane.iter().copied().enumerate().collect();
            entries.sort_by(|(_, a), (_, b)| match (a.is_nan(), b.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                _ if self.largest => b.partial_cmp(a).unwrap(),
                _ => a.partial_cmp(b).unwrap(),
            });
            for (slot, (ix, value)) in entries.into_iter().take(self.k).enumerate() {
                v[slot] = value;
                i[slot] = ix as i64;
            }
        }
        (values, indices)
    }
}

impl Test for TopkProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        _approx: Approximation,
    ) -> infra::TestResult {
        let (mut values, indices) = self.reference();
        let mut indices = indices.into_tensor();
        if self.consumed {
            values.mapv_inplace(|v| -v);
            indices = indices.cast_to::<f32>()?.into_owned();
        }
        let mut model = TypedModel::default();
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let input = model.add_source("input", self.dt.fact(self.input.shape()))?;
        let k = model.add_const("k", rctensor0(self.k as i64))?;
        let op = Topk { axis: self.axis, largest: self.largest, fallback_k: self.k.to_dim() };
        let mut outputs = model.wire_node("topk", op, &[input, k])?;
        if self.consumed {
            outputs[0] = model.wire_node("neg", tract_core::ops::math::neg(), &[outputs[0]])?[0];
            outputs[1] = model.wire_node(
                "cast",
                tract_core::ops::cast::cast(f32::datum_type()),
                &[outputs[1]],
            )?[0];
        }
        model.select_output_outlets(&outputs)?;
        let input = self.input.clone().into_tensor().cast_to_dt(self.dt)?.into_owned();
        let mut found = runtime.prepare(model)?.run(tvec![input.into_tvalue()])?;
        let found_indices = found.remove(1).into_tensor();
        let found_values = found.remove(0).into_tensor();
        let values = values.into_tensor().cast_to_dt(self.dt)?.into_owned();
        found_values.close_enough(&values, Approximation::Exact)?;
        found_indices.close_enough(&indices, Approximation::Exact)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<TopkProblem>("proptest", ());
    let rows =
        |rows: usize, len: usize, k: usize, largest: bool, f: &dyn Fn(usize) -> f32| TopkProblem {
            input: ArrayD::from_shape_fn(IxDyn(&[rows, len]), |ix| f(ix[0] * 31 + ix[1] * 7919)),
            axis: 1,
            k,
            largest,
            dt: DatumType::F32,
            consumed: false,
        };
    let row = |len, k, largest, f: &dyn Fn(usize) -> f32| rows(2, len, k, largest, f);
    let spread = |i: usize| (i % 97) as f32;
    suite.add_test("long_row_largest", row(9000, 300, true, &spread));
    suite.add_test("long_row_smallest", row(9000, 300, false, &spread));
    suite.add_test("long_row_all_equal", row(9000, 300, true, &|_| 1.0));
    suite.add_test(
        "long_row_f16",
        TopkProblem { dt: DatumType::F16, ..row(9000, 300, true, &spread) },
    );
    suite.add_test("signed_zeros", row(64, 40, true, &|i| if i % 2 == 0 { 0.0 } else { -0.0 }));
    suite.add_test("all_nan", row(17, 5, true, &|_| f32::NAN));
    suite.add_test("k_is_the_whole_row", row(37, 37, false, &spread));
    let scattered = |i: usize| (i as u64 * 2654435761 % 2039) as f32;
    for len in [2047, 2048, 2049, 4096, 32000] {
        suite.add_test(format!("row_{len}_k1"), row(len, 1, true, &scattered));
        suite.add_test(format!("row_{len}_k250"), row(len, 250, false, &scattered));
    }
    suite.add_test("row_70000_k1024", row(70000, 1024, true, &scattered));
    suite.add_test("many_rows", rows(300, 50, 7, true, &spread));
    suite.add_test(
        "consumed_outputs",
        TopkProblem { consumed: true, ..row(2048, 250, true, &scattered) },
    );
    Ok(suite)
}
