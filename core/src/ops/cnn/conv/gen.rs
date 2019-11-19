use crate::internal::*;

use super::ConvUnary;
use crate::dim::DimLike;
use crate::ops::cnn::conv::KernelFormat;
use crate::ops::cnn::PaddingSpec;
use crate::ops::nn::DataFormat;
use crate::ops::math::mat_mul::QParams;
use std::borrow::Borrow;

#[derive(Debug, Clone, Default)]
pub struct Conv {
    pub data_format: DataFormat,
    pub kernel_fmt: KernelFormat,
    pub dilations: Option<TVec<usize>>,
    pub kernel_shape: Option<TVec<usize>>,
    pub padding: PaddingSpec,
    pub strides: Option<TVec<usize>>,
    pub group: Option<usize>,

    pub bias_input: Option<usize>,
    pub x_zero_point_input: Option<usize>,
    pub k_zero_point_input: Option<usize>,

    pub override_output_datum_type: Option<DatumType>,
}

impl Conv {
    pub fn nhwc(self) -> Conv {
        Conv { data_format: DataFormat::NHWC, ..self }
    }

    pub fn hwio(self) -> Conv {
        Conv { kernel_fmt: KernelFormat::HWIO, ..self }
    }

    pub fn padding(self, padding: PaddingSpec) -> Conv {
        Conv { padding, ..self }
    }

    pub fn dilations(self, dilations: TVec<usize>) -> Conv {
        Conv { dilations: Some(dilations), ..self }
    }

    pub fn group(self, group: usize) -> Conv {
        Conv { group: Some(group), ..self }
    }

    pub fn strides(self, strides: TVec<usize>) -> Conv {
        Conv { strides: Some(strides), ..self }
    }

    pub fn kernel_shape(self, kernel_shape: TVec<usize>) -> Conv {
        Conv { kernel_shape: Some(kernel_shape), ..self }
    }

    pub fn bias_input(self, input: usize) -> Conv {
        Conv { bias_input: Some(input), ..self }
    }

    pub fn x_zero_point_input(self, input: usize) -> Conv {
        Conv { x_zero_point_input: Some(input), ..self }
    }

    pub fn k_zero_point_input(self, input: usize) -> Conv {
        Conv { k_zero_point_input: Some(input), ..self }
    }

    pub fn override_output_datum_type(self, override_output_datum_type: DatumType) -> Conv {
        Conv { override_output_datum_type: Some(override_output_datum_type), ..self }
    }

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
            KernelFormat::HWIO => kshape[kshape.len() - 1] * self.group.unwrap_or(1),
        };
        result[ishape.c_axis()] = channels_out.into();
        for (ix, d) in computed.iter().enumerate() {
            result[ishape.h_axis() + ix] = d.output.clone();
        }
        result
    }

    pub fn to_unary(&self, inputs: &[impl Borrow<TypedFact>]) -> TractResult<Option<ConvUnary>> {
        let input = &inputs[0];
        let kernel = &inputs[1];
        let input_shape = self.data_format.shape(input.borrow().shape.iter().collect::<TVec<_>>());
        let kshape = kernel.borrow().shape.iter().collect::<TVec<_>>();
        let channels_in = match self.kernel_fmt {
            KernelFormat::OIHW => kshape[1].clone() * self.group.unwrap_or(1),
            KernelFormat::HWIO => kshape[kshape.len() - 2].clone(),
        };
        if input_shape.c_dim() != &channels_in {
            bail!("Input has {} channels, kernel expects {}", input_shape.c_dim(), channels_in)
        }
        if let Some(kvalue) = kernel.borrow().konst.clone() {
            let mut qp = None;
            let dt = self.override_output_datum_type.unwrap_or(input.borrow().datum_type);
            if let Some(slot) = self.x_zero_point_input {
                if let Some(ref value) = inputs[slot].borrow().konst {
                    qp.get_or_insert(QParams::new(dt)).set_zero_point_b(value);
                } else {
                    bail!("Input zero point must be const")
                }
            }
            if let Some(slot) = self.k_zero_point_input {
                if let Some(ref value) = inputs[slot].borrow().konst {
                    qp.get_or_insert(QParams::new(dt)).set_zero_point_a(value);
                } else {
                    bail!("Kernel zero point must be const")
                }
            }
            let reduced = ConvUnary::new(&self, kvalue, self.group.unwrap_or(1), qp)?;
            return Ok(Some(reduced));
        }
        Ok(None)
    }

    pub fn add_bias_t<T: Datum + std::ops::AddAssign>(
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

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Conv {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let inputs_info: TVec<TypedFact> = inputs.iter().map(|t| TypedFact::from(&**t)).collect();
        let unary = self.to_unary(&*inputs_info)?.unwrap();
        let mut result = unary.eval(tvec!(inputs[0].clone()))?;
        if let Some(bias_input) = self.bias_input {
            let mut result = result.remove(0).into_tensor();
            let bias = &inputs[bias_input];
            dispatch_numbers!(Self::add_bias_t(bias.datum_type())(
                self,
                &mut result,
                &bias
            ))?;
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
        s.equals(&inputs[0].datum_type, &inputs[1].datum_type)?;
        if let Some(dt) = self.override_output_datum_type {
            s.equals(&outputs[0].datum_type, dt)?;
        } else {
            s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        }
        if let Some(bias) = self.bias_input {
            s.equals(&inputs[bias].rank, 1)?;
            s.equals(&inputs[0].datum_type, &inputs[bias].datum_type)?;
            s.given(&inputs[1].rank, move |s, krank| {
                let filter_o = match self.kernel_fmt {
                    KernelFormat::OIHW => &inputs[1].shape[0],
                    KernelFormat::HWIO => &inputs[1].shape[krank as usize - 1],
                };
                s.equals(&inputs[bias].shape[0], filter_o)
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
            s.equals(input_c.bex(), self.group.unwrap_or(1) as i32 * filter_i.bex())
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
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[1].shape.iter().all(|d| d.to_integer().is_ok()) {
            let kshape: TVec<usize> =
                inputs[1].shape.iter().map(|d| d.to_integer().unwrap() as _).collect();
            let oshape = self.output_shape(&*inputs[0].shape.to_tvec(), &*kshape);
            Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*oshape)?))
        } else {
            bail!("Streaming on kernel is not typeable")
        }
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let unary =
            self.to_unary(&*inputs)?.ok_or_else(|| format!("Can not unarize conv: {:?}", self))?;
        unary.cost(&[inputs[0]])
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if let Some(op) = self.to_unary(&*inputs)? {
            let mut patch = TypedModelPatch::default();
            let tap = patch.tap_model(model, node.inputs[0])?;
            let mut output: OutletId = patch.wire_node(&*node.name, op, &[tap])?[0];
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

    typed_op_as_op!();
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::setup_test_logger;
    use ndarray::*;

    #[test]
    fn test_infer_with_known_kshape() {
        let mut op = Conv::default().strides(tvec![2, 2]).kernel_shape(tvec![3, 3]);
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 1, 7, 5));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 3));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 2))));
    }

    #[test]
    fn test_infer_channels() {
        let mut op = Conv::default(); // NCHW - OIHW
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1, 1));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefact!(3, 2, 1, 1));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 3, 1, 1))));
    }

    #[test]
    fn test_infer_onxx_strides_no_padding() {
        let mut op = Conv::default().strides(tvec![2, 2]);
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 1, 7, 5));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 3));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 1, 3, 2))));
    }

    #[test]
    fn test_infer_nhwc_1() {
        let mut op = Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper);
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 2, 2, 2));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefact!(2, 2, 2, 1));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 2, 2, 1))));
    }

    #[test]
    fn test_eval_nhwc_1() {
        setup_test_logger();
        let op = Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper);
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
        setup_test_logger();
        let mut op = Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper);
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 1, 2, 2));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefact!(2, 1, 2, 1));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 1, 2, 1))));
    }

    #[test]
    fn test_eval_nhwc_2() {
        setup_test_logger();
        let op = Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper);
        let i = rctensor4(&[[[[0.0f32, 0.0], [1.0, 0.0]]]]);
        let k = rctensor4(&[[[[0.0f32], [0.0]], [[1.0], [0.0]]]]);
        let e = rctensor4(&[[[[1.0f32], [0.0]]]]);
        let res = op.eval(tvec!(i, k)).unwrap();
        assert_eq!(res, tvec!(e.into()));
    }

    #[test]
    fn test_eval_nhwc_3() {
        setup_test_logger();
        let op = Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper);
        let i = rctensor4(&[[[[0.0f32, 1.0], [2.0, 3.0]], [[10.0, 11.0], [12.0, 13.0]]]]);
        let k = rctensor4(&[[[[1.0f32, 0.0], [0.0, 1.0]]]]);
        let res = op.eval(tvec!(i.clone(), k)).unwrap();
        assert_eq!(res, tvec!(i));
    }

    #[test]
    fn test_eval_nhwc_batch() {
        setup_test_logger();
        let op = Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper);
        let result = op
            .eval(tvec!(rctensor4(&[[[[2.0f32]]], [[[0.0f32]]]]), rctensor4(&[[[[1.0f32]]]])))
            .unwrap();
        assert_eq!(result, tvec!(rctensor4(&[[[[2.0f32]]], [[[0.0f32]]]])));
    }

    #[test]
    fn test_infer_ntc_simple() {
        let mut op = Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper);
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 1, 1));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1))));
    }

    #[test]
    fn test_eval_ntc_simple() {
        let op = Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper);
        let result =
            op.eval(tvec!(rctensor3(&[[[2.0f32], [0.0f32]]]), rctensor3(&[[[1.0f32]]]))).unwrap();
        assert_eq!(result, tvec!(rctensor3(&[[[2.0f32], [0.0f32]]])));
    }

    #[test]
    fn test_infer_ntc_batch() {
        let mut op = Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper);
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefact!(2, 1, 1));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 1, 1));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(InferenceFact::dt_shape(DatumType::F32, shapefact!(2, 1, 1))));
    }

    #[test]
    fn test_eval_ntc_batch() {
        let op = Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper);
        let result =
            op.eval(tvec!(rctensor3(&[[[2.0f32]], [[0.0f32]]]), rctensor3(&[[[1.0f32]]]))).unwrap();
        assert_eq!(result, tvec!(rctensor3(&[[[2.0f32]], [[0.0f32]]])));
    }

    #[test]
    fn test_infer_ntc_channel() {
        let mut op = Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper);
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 1, 2));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 2, 1));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(InferenceFact::dt_shape(DatumType::F32, shapefact!(1, 1, 1))));
    }

    #[test]
    fn test_eval_ntc_channel() {
        let op = Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper);
        let result = op
            .eval(tvec!(rctensor3(&[[[2.0f32, 0.0f32]]]), rctensor3(&[[[1.0f32], [0.0f32]]])))
            .unwrap();
        assert_eq!(result, tvec!(rctensor3(&[[[2.0f32]]])));
    }
}
