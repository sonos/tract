use crate::internal::*;

use super::ConvUnary;
use crate::dim::DimLike;
use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::PaddingSpec;
use crate::ops::nn::DataFormat;
use std::borrow::Borrow;

#[derive(Debug, Clone, new)]
pub struct Conv {
    pub data_format: DataFormat,
    pub kernel_fmt: KernelFormat,
    pub dilations: Option<TVec<usize>>,
    pub kernel_shape: Option<TVec<usize>>,
    pub padding: PaddingSpec,
    pub strides: Option<TVec<usize>>,
    pub group: usize,
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
    pub fn output_shape<D: DimLike>(&self, ishape: &[D], kshape: &[usize]) -> TVec<D> {
        debug_assert_eq!(ishape.len(), kshape.len(), "Input and kernel should have the same rank");
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
            result[ishape.h_axis() + ix] = d.output.clone();
        }
        result
    }

    pub fn to_unary(
        &self,
        inputs: &[impl Borrow<TypedTensorInfo>],
    ) -> TractResult<Option<ConvUnary>> {
        let input = &inputs[0];
        let kernel = &inputs[1];
        let input_shape = self.data_format.shape(input.borrow().shape.iter().collect::<TVec<_>>());
        let kshape = kernel.borrow().shape.iter().collect::<TVec<_>>();
        let channels_in = match self.kernel_fmt {
            KernelFormat::OIHW => kshape[1].clone() * self.group,
            KernelFormat::HWIO => kshape[kshape.len() - 2].clone(),
        };
        if input_shape.c_dim() != &channels_in {
            bail!("Input has {} channels, kernel expects {}", input_shape.c_dim(), channels_in)
        }
        if let Some(kvalue) = kernel.borrow().konst.clone() {
            let ishape: TVec<TDim> = input.borrow().shape.iter().collect();
            let reduced = ConvUnary::new(
                &self,
                &ishape,
                &self.output_shape(&*ishape, kvalue.shape()),
                kvalue.into_tensor(),
                self.group,
            )?;
            return Ok(Some(reduced));
        }
        Ok(None)
    }

    pub fn add_bias_t<T: FloatLike + std::ops::AddAssign>(
        &self,
        conv_result: &mut Tensor,
        bias: &Tensor,
    ) -> TractResult<()> {
        let mut conv = conv_result.to_array_view_mut::<T>()?;
        let shape = self.data_format.shape(conv.shape());
        let bias = bias.to_array_view::<T>()?;
        let mut reshaped = vec![1; conv.ndim()];
        reshaped[shape.c_axis()] = bias.len();
        conv += &bias.into_shape(reshaped)?;
        Ok(())
    }
}

impl Op for Conv {
    fn name(&self) -> Cow<str> {
        "Conv".into()
    }

    fn cost(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let unary =
            self.to_unary(&*inputs)?.ok_or_else(|| format!("Can not unarize conv: {:?}", self))?;
        unary.cost(&[inputs[0]])
    }

    op_as_typed_op!();
}

impl StatelessOp for Conv {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let inputs_info: TVec<TypedTensorInfo> =
            inputs.iter().map(|t| TypedTensorInfo::from(&**t)).collect();
        let unary = self.to_unary(&*inputs_info)?.unwrap();
        let mut result = unary.eval(tvec!(inputs[0].clone()))?;
        if let Some(bias) = inputs.get(2) {
            let mut result = result.remove(0).into_tensor();
            dispatch_floatlike!(Self::add_bias_t(bias.datum_type())(self, &mut result, bias))?;
            Ok(tvec!(result.into_arc_tensor()))
        } else {
            Ok(result)
        }
    }
}

impl InferenceRulesOp for Conv {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        if inputs.len() < 2 {
            bail!("Wrong number of inputs. Expected 2 or more, got {}", inputs.len());
        }
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
                let kshape: TVec<usize> =
                    kshape.iter().map(|d| d.to_integer().unwrap() as _).collect();
                let oshape = self.output_shape(&*ishape, &*kshape);
                s.equals(&outputs[0].shape, oshape)?;
            }
            Ok(())
        })
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for Conv {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        if inputs[1].shape.iter().all(|d| d.to_integer().is_ok()) {
            let kshape: TVec<usize> =
                inputs[1].shape.iter().map(|d| d.to_integer().unwrap() as _).collect();
            let oshape = self.output_shape(&*inputs[0].shape.to_tvec(), &*kshape);
            Ok(tvec!(TypedTensorInfo::dt_shape(inputs[0].datum_type, &*oshape)?))
        } else {
            bail!("Streaming on kernel is not typeable")
        }
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if let Some(op) = self.to_unary(&*inputs)? {
            let mut patch = TypedModelPatch::default();
            patch.tap_model(model, node.inputs[0])?;
            let mut output: OutletId =
                patch.chain(&*node.name, op, tvec!(node.outputs[0].fact.clone()))?.into();
            if let Some(bias) = node.inputs.get(2) {
                let mut tap = patch.tap_model(model, *bias)?;
                if self.data_format == DataFormat::NCHW {
                    let data_rank = node.outputs[0].fact.shape.rank();
                    let add_dims = crate::ops::array::AddDims::new((1..data_rank - 1).collect());
                    tap = patch.wire_node(
                        format!("{}-reshaped-bias", node.name),
                        add_dims,
                        [tap].as_ref(),
                    )?[0];
                }
                output = patch.wire_node(
                    format!("{}-add-bias", node.name),
                    crate::ops::math::add::bin(),
                    [output, tap].as_ref(),
                )?[0];
            }
            patch.shunt_outside(OutletId::new(node.id, 0), output)?;
            return Ok(Some(patch));
        } else {
            Ok(None)
        }
    }

}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ops::cnn::conv::KernelFormat::HWIO;
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
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 2))));
    }

    #[test]
    fn test_infer_channels() {
        let mut op = Conv::default(); // NCHW - OIHW
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1, 1));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(3, 2, 1, 1));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 3, 1, 1))));
    }

    #[test]
    fn test_infer_onxx_strides_no_padding() {
        let mut op = Conv::default();
        op.strides = Some(tvec![2, 2]);
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 7, 5));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 3));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 2))));
    }

    #[test]
    fn test_infer_nhwc_1() {
        let mut op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 2, 2));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(2, 2, 2, 1));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 2, 1))));
    }

    #[test]
    fn test_eval_nhwc_1() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let res = op
            .eval(tvec!(
                ArrayD::<f32>::zeros(vec![1, 2, 2, 2]).into_arc_tensor(),
                ArrayD::<f32>::zeros(vec![2, 2, 2, 1]).into_arc_tensor()
            ))
            .unwrap();
        assert_close!(res[0], Tensor::from(ArrayD::<f32>::zeros(vec!(1, 2, 2, 1))).into());
    }

    #[test]
    fn test_infer_nhwc_2() {
        let mut op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 2, 2));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(2, 1, 2, 1));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 2, 1))));
    }

    #[test]
    fn test_eval_nhwc_2() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let i = rctensor4(&[[[[0.0f32, 0.0], [1.0, 0.0]]]]);
        let k = rctensor4(&[[[[0.0f32], [0.0]], [[1.0], [0.0]]]]);
        let e = rctensor4(&[[[[1.0f32], [0.0]]]]);
        let res = op.eval(tvec!(i, k)).unwrap();
        assert_eq!(res, tvec!(e.into()));
    }

    #[test]
    fn test_eval_nhwc_3() {
        //        ::setup_test_logger();
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::Valid, None, 1);
        let i = rctensor4(&[[[[0.0f32, 1.0], [2.0, 3.0]], [[10.0, 11.0], [12.0, 13.0]]]]);
        let k = rctensor4(&[[[[1.0f32, 0.0], [0.0, 1.0]]]]);
        let res = op.eval(tvec!(i.clone(), k)).unwrap();
        assert_eq!(res, tvec!(i));
    }

    #[test]
    fn test_eval_nhwc_batch() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let result = op
            .eval(tvec!(rctensor4(&[[[[2.0f32]]], [[[0.0f32]]]]), rctensor4(&[[[[1.0f32]]]])))
            .unwrap();
        assert_eq!(result, tvec!(rctensor4(&[[[[2.0f32]]], [[[0.0f32]]]])));
    }

    #[test]
    fn test_infer_ntc_simple() {
        let mut op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 1));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1))));
    }

    #[test]
    fn test_eval_ntc_simple() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let result =
            op.eval(tvec!(rctensor3(&[[[2.0f32], [0.0f32]]]), rctensor3(&[[[1.0f32]]]))).unwrap();
        assert_eq!(result, tvec!(rctensor3(&[[[2.0f32], [0.0f32]]])));
    }

    #[test]
    fn test_infer_ntc_batch() {
        let mut op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(2, 1, 1));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 1));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(2, 1, 1))));
    }

    #[test]
    fn test_eval_ntc_batch() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let result =
            op.eval(tvec!(rctensor3(&[[[2.0f32]], [[0.0f32]]]), rctensor3(&[[[1.0f32]]]))).unwrap();
        assert_eq!(result, tvec!(rctensor3(&[[[2.0f32]], [[0.0f32]]])));
    }

    #[test]
    fn test_infer_ntc_channel() {
        let mut op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let ifact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 2));
        let kfact = TensorFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1));
        let ofact = TensorFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(TensorFact::dt_shape(DatumType::F32, shapefact!(1, 1, 1))));
    }

    #[test]
    fn test_eval_ntc_channel() {
        let op = Conv::new(NHWC, HWIO, None, None, PaddingSpec::SameUpper, None, 1);
        let result = op
            .eval(tvec!(rctensor3(&[[[2.0f32, 0.0f32]]]), rctensor3(&[[[1.0f32], [0.0f32]]])))
            .unwrap();
        assert_eq!(result, tvec!(rctensor3(&[[[2.0f32]]])));
    }
}
