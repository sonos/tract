use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use std::ops::Range;
use tract_core::internal::*;
use tract_core::ops::cnn::*;
use tract_core::ops::nn::*;
use tract_ndarray::{prelude::*, *};

use crate::data_format;
use crate::tensor;

#[derive(Debug, Clone, Default)]
pub struct MaxPoolProblemParams {}

#[derive(Debug, Clone)]
pub struct MaxPoolProblem {
    pub data_format: DataFormat,
    pub padding: PaddingSpec,
    pub input: ArrayD<f32>,
    pub kernel_shape: TVec<usize>,
    pub strides: TVec<usize>,
    pub dilations: TVec<usize>,
}

impl Arbitrary for MaxPoolProblem {
    type Strategy = BoxedStrategy<MaxPoolProblem>;
    type Parameters = MaxPoolProblemParams;
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        // Drawn with everything else, a 2x2 unit-dilation NCHW 2D pool is about one case in a
        // thousand, so it gets a branch of its own: that layout has a dedicated loop whose
        // 8-lane body only runs on rows longer than 8.
        prop_oneof![
            problems(
                data_format().boxed(),
                1..4,
                (1usize..4).boxed(),
                (1usize..3).boxed(),
                (1usize..4).boxed(),
                0..6,
            ),
            problems(
                Just(DataFormat::NCHW).boxed(),
                2..3,
                Just(2usize).boxed(),
                Just(1usize).boxed(),
                prop_oneof![Just(1usize), 1usize..4].boxed(),
                0..16,
            ),
        ]
        .boxed()
    }
}

fn problems(
    data_format: BoxedStrategy<DataFormat>,
    georank: Range<usize>,
    kernel: BoxedStrategy<usize>,
    dilation: BoxedStrategy<usize>,
    stride: BoxedStrategy<usize>,
    margin: Range<usize>,
) -> BoxedStrategy<MaxPoolProblem> {
    (data_format, georank)
        .prop_flat_map(move |(df, georank)| {
            (
                Just(df),
                0usize..4, // Valid, SameUpper, SameLower, Explicit
                1usize..3, // n
                1usize..4, // c
                vec(kernel.clone(), georank..=georank),
                vec(dilation.clone(), georank..=georank),
                vec(stride.clone(), georank..=georank),
                vec(margin.clone(), georank..=georank),
                vec((0usize..8, 0usize..8), georank..=georank), // explicit pads, folded below
            )
        })
        .prop_flat_map(|(df, padding, n, c, kernel_shape, dilations, strides, margins, pads)| {
            let fields: Vec<usize> =
                kernel_shape.iter().zip(&dilations).map(|(k, d)| (k - 1) * d + 1).collect();
            // Sides at least as long as their kernel field, and explicit pads shorter than it,
            // leave no window entirely in the padding, where runtimes disagree on the answer
            // (f32::MIN on the CPU, -inf in the Metal kernel).
            let hw: Vec<usize> = fields.iter().zip(&margins).map(|(f, m)| f + m).collect();
            let padding = match padding {
                0 => PaddingSpec::Valid,
                1 => PaddingSpec::SameUpper,
                2 => PaddingSpec::SameLower,
                _ => {
                    let (before, after) =
                        pads.iter().zip(&fields).map(|((b, a), f)| (b % f, a % f)).unzip();
                    PaddingSpec::Explicit(before, after)
                }
            };
            let shape = df.from_n_c_hw(n, c, hw).unwrap();
            (
                Just(df),
                Just(padding),
                tensor(&shape.shape),
                Just(kernel_shape),
                Just(strides),
                Just(dilations),
            )
        })
        .prop_map(|(data_format, padding, input, kernel_shape, strides, dilations)| {
            MaxPoolProblem {
                data_format,
                padding,
                input,
                kernel_shape: kernel_shape.into(),
                strides: strides.into(),
                dilations: dilations.into(),
            }
        })
        .boxed()
}

impl MaxPoolProblem {
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let src = model.add_source("src", f32::fact(self.input.shape()))?;
        let c = *self.data_format.shape(self.input.shape())?.c();
        let pool_spec = PoolSpec::new(
            self.data_format,
            self.kernel_shape.clone(),
            self.padding.clone(),
            Some(self.dilations.clone()),
            Some(self.strides.clone()),
            c,
            c,
        );
        let output = model.wire_node("max_pool", MaxPool::new(pool_spec, None), &[src])?;
        model.select_output_outlets(&output)?;
        Ok(model)
    }

    /// Output length and padding before, per spatial axis, as ONNX defines them.
    fn geometry(&self) -> TractResult<TVec<(usize, usize)>> {
        let input_shape = self.data_format.shape(self.input.shape())?;
        tract_itertools::izip!(
            input_shape.hw_dims(),
            &self.kernel_shape,
            &self.strides,
            &self.dilations
        )
        .enumerate()
        .map(|(axis, (&i, &k, &s, &d))| {
            let field = (k - 1) * d + 1;
            Ok(match &self.padding {
                PaddingSpec::Valid => ((i - field) / s + 1, 0),
                PaddingSpec::Explicit(before, after) => {
                    ((i + before[axis] + after[axis] - field) / s + 1, before[axis])
                }
                PaddingSpec::SameUpper | PaddingSpec::SameLower => {
                    let o = i.div_ceil(s);
                    let total = ((o - 1) * s + field).saturating_sub(i);
                    let before = if self.padding == PaddingSpec::SameUpper {
                        total / 2
                    } else {
                        total - total / 2
                    };
                    (o, before)
                }
                other => bail!("no reference for {other:?}"),
            })
        })
        .collect()
    }

    fn reference(&self) -> TractResult<ArrayD<f32>> {
        let input_shape = self.data_format.shape(self.input.shape())?;
        let n = input_shape.n().copied().unwrap_or(1);
        let c = *input_shape.c();
        let geometry = self.geometry()?;
        let output_hw: TVec<usize> = geometry.iter().map(|g| g.0).collect();
        let output_shape = self.data_format.from_n_c_hw(n, c, &*output_hw)?;
        let mut output = ArrayD::<f32>::zeros(&*output_shape.shape);
        for n in 0..n {
            for c in 0..c {
                for out in indices(&*output_hw) {
                    let mut max: Option<f32> = None;
                    for tap in indices(&*self.kernel_shape) {
                        let pos: Option<TVec<usize>> = tract_itertools::izip!(
                            out.slice(),
                            tap.slice(),
                            &geometry,
                            &self.strides,
                            &self.dilations,
                            input_shape.hw_dims()
                        )
                        .map(|(o, t, (_, before), s, d, i)| {
                            (o * s + t * d).checked_sub(*before).filter(|p| p < i)
                        })
                        .collect();
                        if let Some(pos) = pos {
                            let v = self.input[&*self.data_format.from_n_c_hw(n, c, &*pos)?.shape];
                            max = Some(max.map_or(v, |m| m.max(v)));
                        }
                    }
                    let max = max
                        .with_context(|| format!("window {out:?} lies entirely in the padding"))?;
                    output[&*self.data_format.from_n_c_hw(n, c, out.slice())?.shape] = max;
                }
            }
        }
        Ok(output)
    }
}

impl Test for MaxPoolProblem {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference().context("Running reference")?.into_tensor();
        let mut model = self.tract().context("Generating model")?;
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let mut output = runtime.prepare(model)?.run(tvec![self.input.clone().into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

/// Distinct within any window, so a tap read from the wrong place changes the answer, and all
/// exact in f16.
fn nchw(
    shape: [usize; 4],
    kernel: [usize; 2],
    strides: [usize; 2],
    padding: PaddingSpec,
) -> MaxPoolProblem {
    let len = shape.iter().product::<usize>();
    let values = (0..len).map(|i| ((i * 7919) % 2039) as f32 - 1019.0).collect();
    MaxPoolProblem {
        data_format: DataFormat::NCHW,
        padding,
        input: ArrayD::from_shape_vec(IxDyn(&shape), values).unwrap(),
        kernel_shape: kernel.iter().copied().collect(),
        strides: strides.iter().copied().collect(),
        dilations: tvec!(1, 1),
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<MaxPoolProblem>("proptest", MaxPoolProblemParams::default());

    suite.add("nchw_2x2_s1_valid_8x8", nchw([1, 3, 8, 8], [2, 2], [1, 1], PaddingSpec::Valid));
    suite.add("nchw_2x2_s1_valid_17x19", nchw([1, 16, 17, 19], [2, 2], [1, 1], PaddingSpec::Valid));
    suite.add(
        "nchw_2x2_s1_same_upper_9x9",
        nchw([2, 4, 9, 9], [2, 2], [1, 1], PaddingSpec::SameUpper),
    );
    suite.add("nchw_2x2_s2_valid_16x16", nchw([1, 8, 16, 16], [2, 2], [2, 2], PaddingSpec::Valid));
    suite.add("nchw_3x3_s1_valid_11x13", nchw([1, 4, 11, 13], [3, 3], [1, 1], PaddingSpec::Valid));

    Ok(suite)
}
