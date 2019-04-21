use crate::internal::*;

use super::ConvUnary;
use crate::dim::DimLike;
use crate::ops::nn::conv::KernelFormat;
use crate::ops::nn::DataFormat;
use crate::ops::nn::PaddingSpec;

#[derive(Debug, Clone, new)]
pub struct Conv {
    pub(super) data_format: DataFormat,
    pub(super) kernel_fmt: KernelFormat,
    pub(super) dilations: Option<TVec<usize>>,
    kernel_shape: Option<TVec<usize>>,
    pub(super) padding: PaddingSpec,
    pub(super) strides: Option<TVec<usize>>,
    pub(super) group: usize,
}

impl ::std::default::Default for Conv {
    fn default() -> Conv {
        Conv {
            data_format: DataFormat::default(),
            kernel_fmt: KernelFormat::default(),
            dilations: None,
            kernel_shape: None,
            padding: PaddingSpec::default(),
            strides: None,
            group: 1,
        }
    }
}

impl Conv {
    fn output_shape<D: DimLike>(
        &self,
        ishape: &[D],
        kshape: &[usize],
    ) -> TVec<D> {
        let mut result: TVec<D> = ishape.into();
        let ishape = self.data_format.shape(ishape);
        let spatial_rank = ishape.hw_rank();
        let ones = tvec![1; spatial_rank];
        let kernel_spatial_shape = &kshape[self.kernel_fmt.h_axis()..][..spatial_rank];
        let computed = self.padding.compute(
            ishape.hw_dims(),
            kernel_spatial_shape,
            self.dilations.as_ref().unwrap_or(&ones),
            self.strides.as_ref().unwrap_or(&ones),
        );
        let channels_out = match self.kernel_fmt {
            KernelFormat::OIHW => kshape[0],
            KernelFormat::HWIO => kshape[kshape.len() - 1] * self.group,
        };
        result[ishape.c_axis()] = channels_out.into();
        for (ix, d) in computed.iter().enumerate() {
            result[ishape.h_axis() + ix] = d.output;
        }
        result
    }

    pub fn to_unary(&self, inputs: &[&TypedTensorInfo]) -> TractResult<Option<ConvUnary>> {
        let input = inputs[0];
        let kernel = inputs[1];
        if inputs.len() == 2 {
            if let Some(kvalue) = kernel.konst.clone() {
                let ishape: TVec<TDim> = input.shape.iter().collect();
                let reduced = ConvUnary::new(
                    &self,
                    &ishape,
                    &self.output_shape(&*ishape, kvalue.shape()),
                    kvalue.to_tensor(),
                    None,
                    self.group,
                )?;
                return Ok(Some(reduced));
            }
        } else {
            let bias = inputs[2];
            if let (Some(kvalue), Some(bias)) = (kernel.konst.clone(), bias.konst.clone()) {
                let ishape: TVec<TDim> = input.shape.iter().collect();
                let reduced = ConvUnary::new(
                    &self,
                    &ishape,
                    &self.output_shape(&ishape, kvalue.shape()),
                    kvalue.to_tensor(),
                    Some(bias.to_tensor()),
                    self.group,
                )?;
                return Ok(Some(reduced));
            }
        }
        Ok(None)
    }
}

impl Op for Conv {
    fn name(&self) -> Cow<str> {
        "Conv".into()
    }

    fn cost(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let unary = self.to_unary(&*inputs)?.ok_or_else(|| format!("Can not unarize conv: {:?}", self))?;
        unary.cost(&[inputs[0]])
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if let Some(op) = self.to_unary(&*inputs)? {
            return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
        } else {
            Ok(None)
        }
    }
}

impl StatelessOp for Conv {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (input, kernel, bias) = if inputs.len() == 2 {
            let (input, kernel) = args_2!(inputs);
            (input, kernel, None)
        } else {
            let (input, kernel, bias) = args_3!(inputs);
            (input, kernel, Some(bias.to_tensor()))
        };
        let ishape: TVec<TDim> = input.shape().iter().map(|i| i.to_dim()).collect();
        let reduced = ConvUnary::new(
            &self,
            &ishape,
            &self.output_shape(&*ishape, &kernel.shape()),
            kernel.to_tensor(),
            bias,
            self.group,
        )?;
        reduced.eval(tvec!(input))
    }
}

impl InferenceRulesOp for Conv {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        if let Some(kshape) = &self.kernel_shape {
            s.equals(&inputs[1].rank, kshape.len() as i32 + 2)?;
            for (ix, dim) in kshape.iter().enumerate() {
                s.equals(&inputs[1].shape[ix + self.kernel_fmt.h_axis()], TDim::from(*dim as i32))?;
            }
        }
        s.equals(&inputs[0].rank, &inputs[1].rank)?;
        s.equals(&outputs[0].rank, &inputs[1].rank)?;
        check_output_arity(&outputs, 1)?;
        s.equals_all(wrap![&outputs[0].datum_type, &inputs[0].datum_type, &inputs[1].datum_type])?;
        if inputs.len() == 3 {
            s.equals(&inputs[2].rank, 1)?;
            s.equals(&outputs[0].datum_type, &inputs[2].datum_type)?;
            s.given(&inputs[1].rank, move |s, krank| {
                let filter_o = match self.kernel_fmt {
                    KernelFormat::OIHW => &inputs[1].shape[0],
                    KernelFormat::HWIO => &inputs[1].shape[krank as usize - 1],
                };
                s.equals(&inputs[2].shape[0], filter_o)
            })?
        }
        s.given_2(&inputs[0].rank, &inputs[1].rank, move |s, irank, krank| {
            let input_c = if self.data_format == DataFormat::NHWC {
                &inputs[0].shape[irank as usize - 1]
            } else {
                &inputs[0].shape[1]
            };
            let filter_i = match self.kernel_fmt {
                KernelFormat::OIHW => &inputs[1].shape[1],
                KernelFormat::HWIO => &inputs[1].shape[krank as usize - 2],
            };
            s.equals(input_c.bex(), self.group as i32 * filter_i.bex())
        })?;
        s.given_2(&inputs[0].shape, &inputs[1].shape, move |s, ishape, kshape| {
            if kshape.iter().all(|d| d.to_integer().is_ok()) {
                let kshape:TVec<usize> = kshape.iter().map(|d| d.to_integer().unwrap() as _).collect();
                let oshape = self.output_shape(&*ishape, &*kshape);
                s.equals(&outputs[0].shape, oshape)?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ops::nn::conv::KernelFormat::HWIO;
    use crate::ops::nn::DataFormat::NHWC;
    use ndarray::*;

    #[test]
    fn test_infer_with_known_kshape() {
        let mut op = Conv::default();
        op.strides = Some(tvec![2, 2]);
        op.kernel_shape = Some(tvec![3, 3]);
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 7, 5));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 3));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact)).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 2))));
    }

    #[test]
    fn test_infer_channels() {
        let op = Conv::default(); // NCHW - OIHW
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1, 1));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(3, 2, 1, 1));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact)).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 3, 1, 1))));
    }

    #[test]
    fn test_infer_onxx_strides_no_padding() {
        let mut op = Conv::default();
        op.strides = Some(tvec![2, 2]);
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 7, 5));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 3));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact)).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 2))));
    }

    #[test]
    fn test_infer_nhwc_1() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 2, 2));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(2, 2, 2, 1));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact)).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 2, 1))));
    }

    #[test]
    fn test_eval_nhwc_1() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let res = op
            .eval(tvec!(
                ArrayD::<f32>::zeros(vec![1, 2, 2, 2]).into(),
                ArrayD::<f32>::zeros(vec![2, 2, 2, 1]).into()
            ))
            .unwrap();
        assert_close!(res[0], Tensor::from(ArrayD::<f32>::zeros(vec!(1, 2, 2, 1))).into());
    }

    #[test]
    fn test_infer_nhwc_2() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 2, 2));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(2, 1, 2, 1));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact)).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 2, 1))));
    }

    #[test]
    fn test_eval_nhwc_2() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let i: Tensor = Tensor::from(arr4(&[[[[0.0f32, 0.0], [1.0, 0.0]]]]));
        let k: Tensor = Tensor::from(arr4(&[[[[0.0f32], [0.0]], [[1.0], [0.0]]]]));
        let e: Tensor = Tensor::from(arr4(&[[[[1.0f32], [0.0]]]]));
        let res = op.eval(tvec!(i.into(), k.into())).unwrap();
        assert_eq!(res, tvec!(e.into()));
    }

    #[test]
    fn test_eval_nhwc_3() {
        //        ::setup_test_logger();
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::Valid, None, 1);
        let i: Tensor =
            Tensor::from(arr4(&[[[[0.0f32, 1.0], [2.0, 3.0]], [[10.0, 11.0], [12.0, 13.0]]]]));
        let k: Tensor = Tensor::from(arr4(&[[[[1.0f32, 0.0], [0.0, 1.0]]]]));
        let res = op.eval(tvec!(i.clone().into(), k.into())).unwrap();
        assert_eq!(res, tvec!(i.into()));
    }

    #[test]
    fn test_eval_nhwc_batch() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let result = op
            .eval(tvec!(arr4(&[[[[2.0f32]]], [[[0.0f32]]]]).into(), arr4(&[[[[1.0f32]]]]).into()))
            .unwrap();
        assert_eq!(result, tvec!(arr4(&[[[[2.0f32]]], [[[0.0f32]]]]).into()));
    }

    #[test]
    fn test_infer_ntc_simple() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 1));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact)).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1))));
    }

    #[test]
    fn test_eval_ntc_simple() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let result = op
            .eval(tvec!(arr3(&[[[2.0f32], [0.0f32]]]).into(), arr3(&[[[1.0f32]]]).into()))
            .unwrap();
        assert_eq!(result, tvec!(arr3(&[[[2.0f32], [0.0f32]]]).into()));
    }

    #[test]
    fn test_infer_ntc_batch() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(2, 1, 1));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 1));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact)).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(2, 1, 1))));
    }

    #[test]
    fn test_eval_ntc_batch() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let result = op
            .eval(tvec!(arr3(&[[[2.0f32]], [[0.0f32]]]).into(), arr3(&[[[1.0f32]]]).into()))
            .unwrap();
        assert_eq!(result, tvec!(arr3(&[[[2.0f32]], [[0.0f32]]]).into()));
    }

    #[test]
    fn test_infer_ntc_channel() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 2));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact)).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 1))));
    }

    #[test]
    fn test_eval_ntc_channel() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let result = op
            .eval(tvec!(arr3(&[[[2.0f32, 0.0f32]]]).into(), arr3(&[[[1.0f32], [0.0f32]]]).into()))
            .unwrap();
        assert_eq!(result, tvec!(arr3(&[[[2.0f32]]]).into()));
    }
}
