use analyser::rules::prelude::*;
use ops::prelude::*;

use super::FixedParamsConv;
use dim::DimLike;
use ops::nn::DataFormat;
use ops::nn::PaddingSpec;

use insideout::InsideOut;

#[derive(Debug, Clone, new, Default)]
pub struct Conv {
    pub(super) data_fmt: DataFormat,
    pub(super) kernel_is_hwio: bool, // default is oihw (onnx)
    pub(super) dilations: Option<Vec<usize>>,
    kernel_shape: Option<Vec<usize>>,
    pub(super) padding: PaddingSpec,
    pub(super) strides: Option<Vec<usize>>,
}

impl Conv {
    pub(super) fn axis_kernel_spatial(&self) -> usize {
        if self.kernel_is_hwio {
            0
        } else {
            2
        }
    }

    fn output_shape<D: DimLike>(&self, ishape: &[D], kshape: &[D]) -> Vec<D> {
        let mut result = ishape.to_vec();
        let ishape = self.data_fmt.shape(ishape);
        let spatial_rank = ishape.hw_rank();
        let ones = vec![1; spatial_rank];
        let kernel_spatial_shape = &kshape[2 * (!self.kernel_is_hwio as usize)..][..spatial_rank];
        let computed = self.padding.compute(
            ishape.hw_dims(),
            kernel_spatial_shape,
            self.dilations.as_ref().unwrap_or(&ones),
            self.strides.as_ref().unwrap_or(&ones),
        );
        let channels_out = kshape[(self.kernel_is_hwio as usize)*(kshape.len() - 1)];
        result[ishape.c_axis()] = channels_out;
        result[ishape.hw_axes()].copy_from_slice(&computed.output);
        result
    }
}

impl Op for Conv {
    fn name(&self) -> &str {
        "Conv"
    }

    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let (input, kernel, bias) = if inputs.len() == 2 {
            let (input, kernel) = args_2!(inputs);
            (input, kernel, None)
        } else {
            let (input, kernel, bias) = args_3!(inputs);
            (input, kernel, Some(bias))
        };
        let convoler = FixedParamsConv::new(
            &self,
            input.shape(),
            kernel.to_array_view::<f32>()?,
            bias.as_ref()
                .map(|b| b.to_array_view::<f32>())
                .inside_out()?,
        )?;
        let output = convoler.convolve(&input.to_array_view::<f32>()?)?;
        Ok(tvec!(output.into()))
    }
}

impl InferenceRulesOp for Conv {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        if let Some(kshape) = &self.kernel_shape {
            s.equals(&inputs[1].rank, kshape.len() as i64 + 2)?;
            for (ix, dim) in kshape.iter().enumerate() {
                s.equals(
                    &inputs[1].shape[ix + self.axis_kernel_spatial()],
                    TDim::from(*dim as i64),
                )?;
            }
        }
        s.equals(&outputs.len, 1)?;
        s.equals_all(wrap![
            &outputs[0].datum_type,
            &inputs[0].datum_type,
            &inputs[1].datum_type
        ])?;
        s.given(&inputs.len, move |s, len| {
            if len == 3 {
                s.equals(&inputs[2].rank, 1)?;
                s.equals(&outputs[0].datum_type, &inputs[2].datum_type)?;
                s.given(&inputs[1].rank, move |s, krank| {
                    let filter_o = if self.kernel_is_hwio {
                        &inputs[1].shape[krank as usize - 1]
                    } else {
                        &inputs[1].shape[0] // oihw
                    };
                    s.equals(&inputs[2].shape[0], filter_o)
                })?
            }
            Ok(())
        })?;
        s.given_2(
            &inputs[0].rank,
            &inputs[1].rank,
            move |s, irank, krank| {
                let input_c = if self.data_fmt == DataFormat::NHWC {
                    &inputs[0].shape[irank as usize - 1]
                } else {
                    &inputs[0].shape[1]
                };
                let filter_i = if self.kernel_is_hwio {
                    &inputs[1].shape[krank as usize - 2]
                } else {
                    &inputs[1].shape[1]
                };
                s.equals(input_c, filter_i)
            }
        )?;
        s.given_2(
            &inputs[0].shape,
            &inputs[1].shape,
            move |s, ishape, kshape| {
                let oshape = self.output_shape(&*ishape, &*kshape);
                s.equals(&outputs[0].shape, oshape)
            }
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::prelude::*;
    use ops::nn::DataFormat::NHWC;

    #[test]
    fn test_infer_with_known_kshape() {
        let mut op = Conv::default();
        op.strides = Some(vec![2, 2]);
        op.kernel_shape = Some(vec![3, 3]);
        let facts = op
            .infer_facts(
                tvec!(
                    TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 7, 5)),
                    TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 3)),
                ),
                tvec!(TensorFact::default()),
            ).unwrap();
        assert_eq!(
            facts.1,
            tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 2)))
        );
    }

    #[test]
    fn test_infer_channels() {
        let op = Conv::default();
        let facts = op
            .infer_facts(
                tvec!(
                    TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1, 1)),
                    TensorFact::dt_shape(DatumType::F32, shapefact!(3, 2, 1, 1)),
                ),
                tvec!(TensorFact::default()),
            ).unwrap();
        assert_eq!(
            facts.1,
            tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 3, 1, 1)))
        );
    }

    #[test]
    fn test_infer_onxx_strides_no_padding() {
        let mut op = Conv::default();
        op.strides = Some(vec![2, 2]);
        let facts = op
            .infer_facts(
                tvec!(
                    TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 7, 5)),
                    TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 3)),
                ),
                tvec!(TensorFact::default()),
            ).unwrap();
        assert_eq!(
            facts.1,
            tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 2)))
        );
    }

    #[test]
    fn test_infer_nhwc() {
        let op = Conv::new(NHWC, true, None, None, PaddingSpec::SameUpper, None);
        let facts = op
            .infer_facts(
                tvec!(
                    ArrayD::<f32>::zeros(vec![1, 2, 2, 2]).into(),
                    ArrayD::<f32>::zeros(vec![2, 2, 2, 1]).into()
                ),
                tvec!(TensorFact::default()),
            ).unwrap();
        assert_eq!(
            facts.1,
            tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 2, 1)))
        );
    }

    #[test]
    fn test_eval_nhwc_1() {
        let op = Conv::new(NHWC, true, None, None, PaddingSpec::SameUpper, None);
        let res = op
            .eval(tvec!(
                ArrayD::<f32>::zeros(vec![1, 2, 2, 2]).into(),
                ArrayD::<f32>::zeros(vec![2, 2, 2, 1]).into()
            )).unwrap();
        assert_eq!(
            res,
            tvec!(Tensor::from(ArrayD::<f32>::zeros(vec!(1, 2, 2, 1))).into())
        );
    }

    #[test]
    fn test_eval_nhwc_2() {
        let op = Conv::new(NHWC, true, None, None, PaddingSpec::SameUpper, None);
        let i: Tensor = Tensor::from(arr4(&[[[[0.0f32, 0.0], [1.0, 0.0]]]]));
        let k: Tensor = Tensor::from(arr4(&[[[[0.0f32], [0.0]], [[1.0], [0.0]]]]));
        let e: Tensor = Tensor::from(arr4(&[[[[1.0f32], [0.0]]]]));
        let res = op.eval(tvec!(i.into(), k.into())).unwrap();
        assert_eq!(res, tvec!(e.into()));
    }

    #[test]
    fn test_eval_nhwc() {
        let op = Conv::new(NHWC, true, None, None, PaddingSpec::SameUpper, None);
        let result = op
            .eval(tvec!(
                arr4(&[[[[2.0f32]]], [[[0.0f32]]]]).into(),
                arr4(&[[[[1.0f32]]]]).into()
            )).unwrap();
        assert_eq!(result, tvec!(arr4(&[[[[2.0f32]]], [[[0.0f32]]]]).into()));
    }
}
