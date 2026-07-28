use infra::{Test, TestResult, TestSuite};
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::ndarray::Dimension;
use tract_core::ops::array::GatherElements;

/// `data` is filled with its own flat offsets, so the expected value at a given
/// output coordinate is the offset the op is supposed to read from, computed
/// here from the shape alone.
#[derive(Debug, Clone)]
struct GatherElementsProblem {
    data_shape: Vec<usize>,
    indices: ArrayD<i64>,
    axis: usize,
}

impl GatherElementsProblem {
    fn data(&self) -> TractResult<Tensor> {
        let len = self.data_shape.iter().product::<usize>();
        let values = (0..len).map(|ix| ix as f32).collect::<Vec<_>>();
        Tensor::from_shape(&self.data_shape, &values)
    }

    fn reference(&self) -> ArrayD<f32> {
        let strides = natural_strides(&self.data_shape);
        ArrayD::from_shape_fn(self.indices.shape(), |coords| {
            let index = self.indices[&coords];
            let resolved =
                if index < 0 { index + self.data_shape[self.axis] as i64 } else { index } as usize;
            let offset: isize = coords
                .slice()
                .iter()
                .enumerate()
                .map(|(coord_axis, &coord)| {
                    strides[coord_axis]
                        * if coord_axis == self.axis { resolved } else { coord } as isize
                })
                .sum();
            offset as f32
        })
    }

    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let data = model.add_source("data", f32::fact(&self.data_shape))?;
        let indices = model.add_const("indices", self.indices.clone())?;
        let output =
            model.wire_node("gather_elements", GatherElements::new(self.axis), &[data, indices])?;
        model.select_output_outlets(&output)?;
        Ok(model)
    }
}

impl Arbitrary for GatherElementsProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<GatherElementsProblem>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        vec(1usize..6, 1usize..5)
            .prop_flat_map(|data_shape| {
                let rank = data_shape.len();
                (Just(data_shape), 0..rank)
            })
            .prop_flat_map(|(data_shape, axis)| {
                // Off `axis`, indices are allowed to be smaller than data, which
                // is what takes the op off the contiguous last-axis fast path.
                // `Just(dim)` is drawn explicitly so equal leading dimensions
                // stay common as rank grows instead of being a coincidence.
                let indices_shape = data_shape
                    .iter()
                    .enumerate()
                    .map(|(ax, &dim)| {
                        if ax == axis {
                            (1usize..6).boxed()
                        } else {
                            prop_oneof![Just(dim), 1..=dim].boxed()
                        }
                    })
                    .collect::<Vec<_>>();
                (Just(data_shape), Just(axis), indices_shape)
            })
            .prop_flat_map(|(data_shape, axis, indices_shape)| {
                let len = indices_shape.iter().product::<usize>();
                let axis_len = data_shape[axis] as i64;
                (
                    Just(data_shape),
                    Just(axis),
                    Just(indices_shape),
                    vec(-axis_len..axis_len, len..=len),
                )
            })
            .prop_map(|(data_shape, axis, indices_shape, indices)| GatherElementsProblem {
                indices: ArrayD::from_shape_vec(indices_shape, indices).unwrap(),
                data_shape,
                axis,
            })
            .boxed()
    }
}

impl Test for GatherElementsProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference().into_tensor();
        let mut model = self.tract()?;
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let mut output = runtime.prepare(model)?.run(tvec!(self.data()?.into_tvalue()))?;
        output.remove(0).into_tensor().close_enough(&reference, approx)
    }
}

fn problem(
    data_shape: &[usize],
    axis: usize,
    indices_shape: &[usize],
    indices: impl Fn(usize) -> i64,
) -> GatherElementsProblem {
    let len = indices_shape.iter().product::<usize>();
    let indices = (0..len).map(indices).collect::<Vec<_>>();
    GatherElementsProblem {
        data_shape: data_shape.into(),
        indices: ArrayD::from_shape_vec(indices_shape.to_vec(), indices).unwrap(),
        axis,
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<GatherElementsProblem>("proptest", ());

    suite.add("last_axis", problem(&[2, 3, 8], 2, &[2, 3, 5], |ix| (ix * 3 % 8) as i64));
    suite.add(
        "last_axis_negative",
        problem(&[2, 3, 8], 2, &[2, 3, 5], |ix| (ix * 3 % 8) as i64 - 8),
    );
    suite.add("last_axis_rank_1", problem(&[8], 0, &[5], |ix| (ix * 3 % 8) as i64));
    suite.add("first_axis", problem(&[4, 8], 0, &[3, 8], |ix| (ix % 4) as i64));
    // A reduced OUTERMOST dim leaves the flat row offset accidentally correct, so
    // it does not on its own prove the fast path checks the leading dims.
    suite.add("mismatched_inner_dim", problem(&[2, 3, 8], 2, &[2, 2, 5], |ix| (ix % 8) as i64));
    suite.add("mismatched_outer_dim", problem(&[2, 3, 8], 2, &[1, 3, 5], |ix| (ix % 8) as i64));
    suite.add("empty_rows", problem(&[2, 0, 8], 2, &[2, 0, 5], |_| 0));
    suite.add("empty_last_axis", problem(&[2, 3, 8], 2, &[2, 3, 0], |_| 0));

    Ok(suite)
}
