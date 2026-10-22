use super::*;
use crate::conv_f32::{ConvProblem, ConvProblemParams};
use infra::*;
use tract_core::tract_data::half::f16;

#[derive(Debug, Clone)]
pub struct ConvProblemF16(pub ConvProblem);

impl ConvProblemF16 {
    fn tract(&self) -> TractResult<TypedModel> {
        let inner = &self.0;
        assert_eq!(inner.data.shape(), &*inner.shape_in.shape, "inconsistent shapes in test");
        let mut model = TypedModel::default();
        let wire = model.add_source("input", f16::fact(&inner.shape_in.shape))?;
        let ci = *inner.shape_in.c();
        let co = match inner.kernel_format {
            KernelFormat::OIHW => inner.kernel.shape()[0],
            KernelFormat::HWIO => inner.kernel.shape()[inner.kernel.ndim() - 1] * inner.group,
            KernelFormat::OHWI => inner.kernel.shape()[0] * inner.group,
        };
        let kernel_f16 = inner.kernel.mapv(f16::from_f32).into_arc_tensor();
        let kernel = model.add_const("kernel", kernel_f16)?;
        let bias_f16 = if let Some(bias) = &inner.bias {
            bias.mapv(f16::from_f32).into_arc_tensor()
        } else {
            rctensor0(f16::from_f32(0.0))
        };
        let bias = model.add_const("bias", bias_f16)?;
        let op = Conv::new(
            PoolSpec::new(
                inner.shape_in.fmt,
                inner.geo_ker().into(),
                inner.pad.clone(),
                Some(inner.dilations.clone()),
                Some(inner.strides.clone()),
                ci,
                co,
            ),
            inner.kernel_format,
            inner.group,
            None,
        );
        let wire = model.wire_node("conv", op, &[wire, kernel, bias])?[0];
        model.select_output_outlets(&[wire])?;
        Ok(model)
    }
}

impl Arbitrary for ConvProblemF16 {
    type Parameters = ConvProblemParams;
    type Strategy = BoxedStrategy<ConvProblemF16>;
    fn arbitrary_with(params: Self::Parameters) -> Self::Strategy {
        ConvProblem::arbitrary_with(params).prop_map(ConvProblemF16).boxed()
    }
}

impl Test for ConvProblemF16 {
    fn run_with_approx(
        &self,
        id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let reference = self.0.reference().into_tensor();
        let mut model = self.tract()?;
        model.declutter()?;
        model.properties.insert("tract-rt-test.id".to_string(), rctensor0(id.to_string()));
        let input_f16 = self.0.data.mapv(f16::from_f32).into_tensor();
        let mut output = runtime.prepare(model)?.run(tvec![input_f16.into_tvalue()])?;
        let output = output.remove(0).into_tensor();
        // Judge at f16 precision: the conv ran in f16, so compare against the reference
        // rounded to f16 (this selects the F16 tolerance row). Casting the output up to
        // f32 and comparing with f32 tolerances rejects legitimate ~1 f16-ULP differences.
        let reference_f16 = reference.cast_to::<f16>()?.into_owned();
        output.close_enough(&reference_f16, approx)
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<ConvProblemF16>("proptest", ConvProblemParams::default());

    // m, k and n all multiples of 4: the shape GPU backends route to their blocked matmul.
    let blocked = |bias: Option<ArrayD<f32>>| -> TractResult<ConvProblemF16> {
        Ok(ConvProblemF16(ConvProblem {
            shape_in: DataFormat::NCHW.from_n_c_hw(1, 4, [4, 4])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: ArrayD::from_shape_fn(vec![1, 4, 4, 4], |ix| (ix[1] + ix[3]) as f32 - 3.0),
            kernel: ArrayD::from_shape_fn(vec![4, 4, 1, 1], |ix| (ix[0] * 4 + ix[1]) as f32 - 7.0),
            bias,
            pad: PaddingSpec::Valid,
            strides: tvec!(1, 1),
            dilations: tvec!(1, 1),
        }))
    };
    suite.add("blocked_matmul", blocked(None)?);
    suite.add("blocked_matmul_bias", blocked(Some(arr1(&[1.0, -2.0, 3.0, -4.0]).into_dyn()))?);

    // A filter many times larger than the input: the shape GPU backends may run in NHWC.
    let filter_heavy = |strides: usize, bias: Option<ArrayD<f32>>| -> TractResult<ConvProblemF16> {
        Ok(ConvProblemF16(ConvProblem {
            shape_in: DataFormat::NCHW.from_n_c_hw(2, 24, [5, 6])?,
            kernel_format: KernelFormat::OIHW,
            group: 1,
            data: ArrayD::from_shape_fn(vec![2, 24, 5, 6], |ix| {
                ((ix[0] * 7 + ix[1] * 3 + ix[2] * 5 + ix[3]) % 9) as f32 * 0.25 - 1.0
            }),
            kernel: ArrayD::from_shape_fn(vec![64, 24, 3, 3], |ix| {
                ((ix[0] * 5 + ix[1] * 3 + ix[2] * 2 + ix[3]) % 7) as f32 * 0.125 - 0.375
            }),
            bias,
            pad: PaddingSpec::Explicit(tvec!(1, 1), tvec!(1, 1)),
            strides: tvec!(strides, strides),
            dilations: tvec!(1, 1),
        }))
    };
    suite.add("filter_heavy", filter_heavy(1, None)?);
    suite.add("filter_heavy_stride_2", filter_heavy(2, None)?);
    suite.add(
        "filter_heavy_bias",
        filter_heavy(1, Some(ArrayD::from_shape_fn(vec![64], |ix| ix[0] as f32 * 0.25 - 8.0)))?,
    );
    let same = || PaddingSpec::Explicit(tvec!(1, 1), tvec!(1, 1));
    let winograd = |n, hw, c, o, pad, bias| -> TractResult<ConvProblemF16> {
        Ok(ConvProblemF16(crate::conv_f32::winograd_sized(n, hw, c, o, pad, bias)?))
    };
    suite.add("winograd_same", winograd(1, [67, 70], 9, 250, same(), false)?);
    suite.add("winograd_valid", winograd(2, [50, 47], 250, 7, PaddingSpec::Valid, false)?);
    suite.add("winograd_bias", winograd(1, [64, 64], 17, 240, same(), true)?);

    Ok(suite)
}
