use infra::{Test, TestSuite};
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ops::array::{Pad, PadMode};

#[derive(Debug, Clone)]
struct PadProblem {
    shape: Vec<usize>,
    pads: Vec<(usize, usize)>,
    dt: DatumType,
    fill: f32,
}

impl Arbitrary for PadProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: ()) -> Self::Strategy {
        vec((0usize..5, 0usize..3, 0usize..3), 0..5)
            .prop_map(|axes| PadProblem {
                shape: axes.iter().map(|a| a.0).collect(),
                pads: axes.iter().map(|a| (a.1, a.2)).collect(),
                dt: DatumType::F32,
                fill: -3.0,
            })
            .boxed()
    }
}

impl PadProblem {
    fn reference<T: Datum + Copy>(&self, input: &Tensor, fill: &Tensor) -> TractResult<Tensor> {
        let input = input.to_plain_array_view::<T>()?;
        let shape: Vec<usize> =
            self.shape.iter().zip(&self.pads).map(|(d, (a, b))| d + a + b).collect();
        let mut output = tract_ndarray::ArrayD::from_elem(shape, fill.cast_to_scalar::<T>()?);
        for (coord, value) in output.indexed_iter_mut() {
            let mut source = Vec::with_capacity(self.shape.len());
            for axis in 0..self.shape.len() {
                let Some(ix) = coord[axis].checked_sub(self.pads[axis].0) else { break };
                if ix >= self.shape[axis] {
                    break;
                }
                source.push(ix);
            }
            if source.len() == self.shape.len() {
                *value = input[tract_ndarray::IxDyn(&source)];
            }
        }
        Ok(output.into_tensor())
    }
}

impl Test for PadProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        _approx: Approximation,
    ) -> infra::TestResult {
        let values: Vec<f32> =
            (0..self.shape.iter().product()).map(|i| (i % 19) as f32 - 9.0).collect();
        let input = Tensor::from_shape(&self.shape, &values)?.cast_to_dt(self.dt)?.into_owned();
        let fill = tensor0(self.fill);
        let reference =
            dispatch_numbers!(Self::reference(self.dt.unquantized())(self, &input, &fill))?;
        let mut model = TypedModel::default();
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let wire = model.add_source("input", TypedFact::shape_and_dt_of(&input))?;
        let output = model.wire_node(
            "pad",
            Pad { pads: self.pads.clone(), mode: PadMode::Constant(fill.into_arc_tensor()) },
            &[wire],
        )?;
        model.select_output_outlets(&output)?;
        let prepared = runtime.prepare(model)?;
        let output = prepared.run(tvec![input.into_tvalue()])?.remove(0).into_tensor();
        ensure!(output.datum_type() == self.dt);
        ensure!(output.shape() == reference.shape());
        ensure!(output.as_bytes() == reference.as_bytes(), "Constant pad changed values");
        Ok(())
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<PadProblem>("proptest", ());
    for (name, shape, pads) in [
        ("scalar", vec![], vec![]),
        ("identity", vec![2, 3], vec![(0, 0), (0, 0)]),
        ("vector", vec![3], vec![(2, 1)]),
        ("outer", vec![2, 3, 4], vec![(2, 1), (0, 0), (0, 0)]),
        ("middle", vec![2, 3, 4], vec![(0, 0), (1, 2), (0, 0)]),
        ("all_axes", vec![2, 3, 4], vec![(1, 2), (2, 0), (0, 3)]),
        ("empty_inner", vec![2, 0], vec![(1, 2), (3, 1)]),
        ("empty_outer", vec![0, 3], vec![(1, 2), (2, 1)]),
        ("empty_output", vec![2, 0, 3], vec![(0, 0), (0, 0), (1, 1)]),
        ("nchw", vec![2, 3, 130, 129], vec![(0, 0), (0, 0), (1, 2), (3, 1)]),
        ("nhwc", vec![2, 130, 129, 3], vec![(0, 0), (1, 2), (3, 1), (0, 0)]),
    ] {
        for dt in [DatumType::F16, DatumType::F32, DatumType::F64, DatumType::I8, DatumType::I32] {
            suite.add(
                format!("{name}_{:?}", dt.unquantized()),
                PadProblem { shape: shape.clone(), pads: pads.clone(), dt, fill: -3.0 },
            );
        }
    }
    for (name, fill) in [("zero", 0.0f32), ("negative_zero", -0.0f32)] {
        for dt in [
            DatumType::F16,
            DatumType::F32,
            DatumType::F64,
            DatumType::QI8(QParams::ZpScale { zero_point: 5, scale: 0.25 }),
        ] {
            suite.add(
                format!("{name}_{:?}", dt.unquantized()),
                PadProblem { shape: vec![2, 3], pads: vec![(1, 2), (2, 1)], dt, fill },
            );
        }
    }
    Ok(suite)
}
