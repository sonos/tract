use std::fmt;

use infra::Test;
use infra::TestResult;
use infra::TestSuite;
use proptest::collection::vec;
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::num_traits::Float;
use tract_core::num_traits::FromPrimitive;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::cast::Cast;
use tract_core::ops::math::{Add, Mul};
use tract_transformers::ops::rms_norm::RmsNorm;

use crate::tensor;

#[derive(Clone)]
pub struct RmsNormProblem<F>
where
    F: Datum + Float,
{
    input: Tensor,
    axis: usize,
    eps: f32,
    _phantom: PhantomData<F>,
}

impl<F> std::fmt::Debug for RmsNormProblem<F>
where
    F: Datum + Float,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Input:{:?} Axis:{:?} Epsilon:{:?}", self.input, self.axis, self.eps)
    }
}

impl<F> Arbitrary for RmsNormProblem<F>
where
    F: Datum + Float,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<RmsNormProblem<F>>;

    fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
        (0usize..3, 0usize..3)
            .prop_flat_map(|(left, right)| {
                let axis = left;
                let shape_len = usize::min(left + right, 4);
                let iter_ax_dim = 1usize..50;
                let other_dim = 1usize..5;
                (iter_ax_dim, vec(other_dim, shape_len..=shape_len), Just(axis))
            })
            .prop_flat_map(|(iter_dim, mut shape, axis)| {
                shape.insert(axis, iter_dim);
                let input = tensor::<F>(&shape);
                (input, Just(axis), 0f32..=1e6).prop_map(|(input, axis, eps)| Self {
                    input: input.into(),
                    axis,
                    eps: eps / 1e5,
                    _phantom: PhantomData,
                })
            })
            .boxed()
    }
}

impl<F> RmsNormProblem<F>
where
    F: Datum + Float + FromPrimitive,
{
    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let input = model.add_source("input", TypedFact::shape_and_dt_of(&self.input))?;

        let output = model.wire_node(
            "rms_norm",
            RmsNorm { axis: self.axis, eps: tensor0(self.eps).into_arc_tensor() },
            &[input],
        )?;
        model.select_output_outlets(&output)?;

        model = model.into_decluttered()?;
        Ok(model)
    }

    fn reference(&self) -> ArrayD<F> {
        let input = self.input.cast_to::<f32>().unwrap();

        let a = input.to_plain_array_view::<f32>().unwrap().to_owned();
        let mean_square = a.pow2().mean_axis(tract_ndarray::Axis(self.axis)).unwrap();

        let norm = mean_square
            .mapv(|ms| (ms + self.eps).sqrt())
            .insert_axis(tract_ndarray::Axis(self.axis));
        let broadcasted_norm = norm.broadcast(a.raw_dim()).unwrap().to_owned();

        (a / broadcasted_norm).mapv(|x| F::from(x).unwrap())
    }
}

impl<F> Test for RmsNormProblem<F>
where
    F: Datum + Float + FromPrimitive,
{
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference().into_tensor();
        let mut model = self.tract()?;

        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));

        let mut output = runtime.prepare(model)?.run(tvec!(self.input.clone().into()))?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

/// A pre-norm residual add and/or a learned-gamma scale multiply around a
/// plain `RmsNorm`, i.e. exactly the pattern `fuse_rms_norm_residual` /
/// `fuse_rms_norm_scale` target on GPU backends. Unlike `RmsNormProblem`,
/// this runs through the *real* backend transform pipeline (whatever
/// `Runtime` the harness is configured with), so on Metal/Cuda it exercises
/// the actual rewrite rules firing and the fused device kernel executing,
/// not just the kernel in isolation. `residual`/`gamma` being `None` means
/// that part of the pattern is absent from the graph, covering the
/// plain/scale-only/residual-only/both combinations.
///
/// `axis` is always the last axis: `gamma`'s reference broadcast (a 1-D
/// vector along that axis) only lines up trivially for a trailing axis.
#[derive(Clone)]
pub struct RmsNormFusionProblem<F>
where
    F: Datum + Float,
{
    input: Tensor,
    residual: Option<Tensor>,
    gamma: Option<Tensor>,
    eps: f32,
    _phantom: PhantomData<F>,
}

impl<F> std::fmt::Debug for RmsNormFusionProblem<F>
where
    F: Datum + Float,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Input:{:?} Residual:{:?} Gamma:{:?} Epsilon:{:?}",
            self.input, self.residual, self.gamma, self.eps
        )
    }
}

impl<F> Arbitrary for RmsNormFusionProblem<F>
where
    F: Datum + Float,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
        (vec(1usize..5, 0..3), 1usize..64)
            .prop_flat_map(|(mut leading, dim)| {
                leading.push(dim);
                let shape = leading;
                // Gamma is rank-padded to the input's rank (1s everywhere
                // but the last axis): a plain rank-1 vector would fail the
                // elementwise Mul's rank-match check against a higher-rank
                // input, same as any real model would need to shape it.
                let mut gamma_shape = vec![1; shape.len()];
                *gamma_shape.last_mut().unwrap() = dim;
                (
                    tensor::<F>(&shape),
                    proptest::option::of(tensor::<F>(&shape)),
                    proptest::option::of(tensor::<f32>(&gamma_shape)),
                    0f32..=1e6,
                )
            })
            .prop_map(|(input, residual, gamma, eps)| Self {
                input: input.into(),
                residual: residual.map(Into::into),
                gamma: gamma.map(Into::into),
                eps: eps / 1e5,
                _phantom: PhantomData,
            })
            .boxed()
    }
}

impl<F> RmsNormFusionProblem<F>
where
    F: Datum + Float + FromPrimitive,
{
    fn axis(&self) -> usize {
        self.input.rank() - 1
    }

    fn tract(&self) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let mut cur = model.add_source("input", TypedFact::shape_and_dt_of(&self.input))?;
        if let Some(residual) = &self.residual {
            let res = model.add_source("residual", TypedFact::shape_and_dt_of(residual))?;
            cur = model.wire_node("add", TypedBinOp(Box::new(Add), None), &[cur, res])?[0];
        }
        let mut out = model.wire_node(
            "rms_norm",
            RmsNorm { axis: self.axis(), eps: tensor0(self.eps).into_arc_tensor() },
            &[cur],
        )?[0];
        if let Some(gamma) = &self.gamma {
            // Gamma is always f32 (the realistic case: activations run in
            // the working dtype, learned norm weights stay f32). A raw
            // mixed-dtype `TypedBinOp(Mul)` isn't something tract-core's
            // generic binary op supports -- real models bridge the
            // boundary with explicit casts, which is exactly the
            // `fuse_scaled_rms_norm_in_cast`/`out_cast` target pattern, so
            // build it the same way here.
            if F::datum_type() != DatumType::F32 {
                out = model.wire_node("norm.cast", Cast { to: DatumType::F32 }, &[out])?[0];
            }
            let gamma = model.add_const("gamma", gamma.clone())?;
            out = model.wire_node("scale", TypedBinOp(Box::new(Mul), None), &[out, gamma])?[0];
            if F::datum_type() != DatumType::F32 {
                out = model.wire_node("scale.cast", Cast { to: F::datum_type() }, &[out])?[0];
            }
        }
        model.select_output_outlets(&[out])?;
        model = model.into_decluttered()?;
        Ok(model)
    }

    fn reference(&self) -> ArrayD<F> {
        let axis = self.axis();
        let mut a =
            self.input.cast_to::<f32>().unwrap().to_plain_array_view::<f32>().unwrap().to_owned();
        if let Some(residual) = &self.residual {
            let r =
                residual.cast_to::<f32>().unwrap().to_plain_array_view::<f32>().unwrap().to_owned();
            a += &r;
        }
        let mean_square = a.pow2().mean_axis(tract_ndarray::Axis(axis)).unwrap();
        let norm =
            mean_square.mapv(|ms| (ms + self.eps).sqrt()).insert_axis(tract_ndarray::Axis(axis));
        let broadcasted_norm = norm.broadcast(a.raw_dim()).unwrap().to_owned();
        let mut normed = a / broadcasted_norm;
        if let Some(gamma) = &self.gamma {
            let g =
                gamma.cast_to::<f32>().unwrap().to_plain_array_view::<f32>().unwrap().to_owned();
            let g = g.broadcast(normed.raw_dim()).unwrap().to_owned();
            normed *= &g;
        }
        normed.mapv(|x| F::from(x).unwrap())
    }
}

impl<F> Test for RmsNormFusionProblem<F>
where
    F: Datum + Float + FromPrimitive,
{
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.reference().into_tensor();
        let mut model = self.tract()?;

        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));

        let mut inputs = tvec!(self.input.clone().into());
        if let Some(residual) = &self.residual {
            inputs.push(residual.clone().into());
        }
        let mut output = runtime.prepare(model)?.run(inputs)?;
        let output = output.remove(0).into_tensor();
        output.close_enough(&reference, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();

    suite.add_arbitrary::<RmsNormProblem<f32>>("proptest_f32", ());
    suite.add_arbitrary::<RmsNormProblem<f16>>("proptest_f16", ());

    suite.add(
        "trivial_f32_0",
        RmsNormProblem::<f32> {
            input: tensor1(&[0f32]),
            axis: 0,
            eps: 0f32,
            _phantom: PhantomData,
        },
    );

    suite.add_arbitrary::<RmsNormFusionProblem<f32>>("proptest_fusion_f32", ());
    suite.add_arbitrary::<RmsNormFusionProblem<f16>>("proptest_fusion_f16", ());

    suite.add(
        "fusion_residual_and_scale_f32",
        RmsNormFusionProblem::<f32> {
            input: Tensor::from_shape(
                &[2, 8],
                &(0..16).map(|i| i as f32 / 3.0 - 2.0).collect::<Vec<_>>(),
            )?,
            residual: Some(Tensor::from_shape(
                &[2, 8],
                &(0..16).map(|i| i as f32 / 5.0 - 1.0).collect::<Vec<_>>(),
            )?),
            gamma: Some(Tensor::from_shape(
                &[1, 8],
                &(0..8).map(|i| 0.5 + i as f32 / 8.0).collect::<Vec<_>>(),
            )?),
            eps: 1e-4,
            _phantom: PhantomData,
        },
    );

    Ok(suite)
}
