use crate::infer::*;
use crate::internal::*;

use std::borrow::Borrow;
use tract_core::dim::DimLike;
use tract_core::ops::cnn::conv::ConvUnary;
use tract_core::ops::cnn::conv::KernelFormat;
use tract_core::ops::cnn::{PaddingSpec, PoolSpec};
use tract_core::ops::nn::DataFormat;
use tract_core::ops::quant::QParams;

#[derive(Debug, Clone, Default, Hash)]
pub struct Conv {
    pub data_format: DataFormat,
    pub kernel_fmt: KernelFormat,
    pub dilations: Option<TVec<usize>>,
    pub kernel_shape: Option<TVec<usize>>,
    pub padding: PaddingSpec,
    pub strides: Option<TVec<usize>>,
    pub group: Option<usize>,

    pub x_scale_input: Option<usize>,
    pub x_zero_point_input: Option<usize>,
    pub k_input: Option<usize>,
    pub k_scale_input: Option<usize>,
    pub k_zero_point_input: Option<usize>,

    pub y_scale_input: Option<usize>,
    pub y_zero_point_input: Option<usize>,

    pub bias_input: Option<usize>,

    pub override_output_datum_type: Option<DatumType>,
}

tract_data::impl_dyn_hash!(Conv);

impl Conv {
    pub fn hwc(self) -> Conv {
        Conv { data_format: DataFormat::HWC, ..self }
    }

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

    pub fn output_shape<D: DimLike>(&self, ishape: &[D], kshape: &[usize]) -> TractResult<TVec<D>> {
        debug_assert_eq!(
            ishape.len()
                + (self.data_format == DataFormat::HWC || self.data_format == DataFormat::CHW)
                    as usize,
            kshape.len(),
            "Input and kernel ranks are inconsistent"
        );
        let mut result: TVec<D> = ishape.into();
        let ishape = self.data_format.shape(ishape)?;
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
        Ok(result)
    }

    pub fn to_unary(&self, inputs: &[&TypedFact]) -> TractResult<Option<ConvUnary>> {
        let input = &inputs[0].borrow();
        let kernel = &inputs[self.k_input.unwrap_or(1)].borrow();
        let input_shape = self.data_format.shape(input.shape.iter().collect::<TVec<_>>())?;
        let kshape = kernel.shape.iter().collect::<TVec<_>>();
        let channels_in = match self.kernel_fmt {
            KernelFormat::OIHW => kshape[1].clone() * self.group.unwrap_or(1),
            KernelFormat::HWIO => kshape[kshape.len() - 2].clone(),
        };
        if input_shape.c_dim() != &channels_in {
            bail!("Input has {} channels, kernel expects {}", input_shape.c_dim(), channels_in)
        }
        if let Some(kvalue) = kernel.konst.clone() {
            let mut qp = None;
            let dt = self.override_output_datum_type.unwrap_or(input.datum_type);
            let mut scale = 1.0;
            if let Some(slot) = self.x_scale_input {
                if let Some(ref value) = inputs[slot].borrow().konst {
                    scale *= value.to_scalar::<f32>()?;
                } else {
                    bail!("Input scale must be const")
                }
            }
            if let Some(slot) = self.k_scale_input {
                if let Some(ref value) = inputs[slot].borrow().konst {
                    scale *= value.to_scalar::<f32>()?;
                } else {
                    bail!("Filter scale must be const")
                }
            }
            if let Some(slot) = self.y_scale_input {
                if let Some(ref value) = inputs[slot].borrow().konst {
                    scale /= value.to_scalar::<f32>()?;
                } else {
                    bail!("Output scale must be const")
                }
            }
            if scale != 1.0 {
                qp.get_or_insert(QParams::new(dt)).set_scale_factor(scale);
            }
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
            if let Some(slot) = self.y_zero_point_input {
                if let Some(ref value) = inputs[slot].borrow().konst {
                    qp.get_or_insert(QParams::new(dt)).set_zero_point_c(value);
                } else {
                    bail!("Output zero point must be const")
                }
            }
            let bias = if let Some(slot) = self.bias_input {
                if let Some(ref value) = inputs[slot].borrow().konst {
                    Some(value.clone())
                } else {
                    bail!("Bias must be const")
                }
            } else {
                None
            };
            let spatial_rank = kernel.rank() - 2;
            let kshape = kvalue.shape();
            let group = self.group.unwrap_or(1);
            let output_channels = match self.kernel_fmt {
                KernelFormat::OIHW => kshape[0],
                KernelFormat::HWIO => kshape[kshape.len() - 1] * group,
            };
            let pool_spec = PoolSpec {
                data_format: self.data_format,
                padding: self.padding.clone(),
                strides: self.strides.clone(),
                dilations: self.dilations.clone(),
                kernel_shape: kshape[self.kernel_fmt.h_axis()..][..spatial_rank].into(),
                output_channel_override: Some(output_channels),
            };
            let reduced = ConvUnary::new(pool_spec, self.kernel_fmt, kvalue, group, bias, qp);
            return Ok(Some(reduced));
        } else {
            bail!("Kernel should be a const, found {:?}", kernel)
        }
    }
}

impl Expansion for Conv {
    fn name(&self) -> Cow<str> {
        "Conv".into()
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        if inputs.len() < 2 {
            bail!("Wrong number of inputs. Expected 2 or more, got {}", inputs.len());
        }
        let has_n = self.data_format == DataFormat::NHWC || self.data_format == DataFormat::NCHW;
        let k_input = &inputs[self.k_input.unwrap_or(1)];
        if let Some(kshape) = &self.kernel_shape {
            s.equals(&k_input.rank, kshape.len() as i64 + 2)?;
            for (ix, dim) in kshape.iter().enumerate() {
                s.equals(&k_input.shape[ix + self.kernel_fmt.h_axis()], TDim::from(*dim as i64))?;
            }
        }
        s.equals(&inputs[0].rank, k_input.rank.bex() + (has_n as usize as i64 - 1))?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &k_input.datum_type)?;
        if let Some(dt) = self.override_output_datum_type {
            s.equals(&outputs[0].datum_type, dt)?;
        } else {
            s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        }
        if let Some(bias) = self.bias_input {
            // bias datum type is ill-defined. no check
            s.equals(&inputs[bias].rank, 1)?;
            s.given(&k_input.rank, move |s, krank| {
                let filter_o = match self.kernel_fmt {
                    KernelFormat::OIHW => &k_input.shape[0],
                    KernelFormat::HWIO => &k_input.shape[krank as usize - 1],
                };
                s.equals(&inputs[bias].shape[0], filter_o)
            })?
        }
        s.given_2(&inputs[0].rank, &k_input.rank, move |s, irank, krank| {
            let input_c =
                if self.data_format == DataFormat::NHWC || self.data_format == DataFormat::HWC {
                    &inputs[0].shape[irank as usize - 1]
                } else {
                    &inputs[0].shape[1]
                };
            let filter_i = match self.kernel_fmt {
                KernelFormat::OIHW => &k_input.shape[1],
                KernelFormat::HWIO => &k_input.shape[krank as usize - 2],
            };
            s.equals(input_c.bex(), self.group.unwrap_or(1) as i64 * filter_i.bex())
        })?;
        s.given_2(&inputs[0].shape, &k_input.shape, move |s, ishape, kshape| {
            if let Some(kshape) =
                kshape.iter().map(|d| d.to_usize().ok()).collect::<Option<TVec<_>>>()
            {
                let oshape = self.output_shape(&*ishape, &*kshape)?;
                s.equals(&outputs[0].shape, oshape)?;
            }
            Ok(())
        })
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let unary = {
            let facts: TVec<&TypedFact> =
                inputs.iter().map(|t| target.outlet_fact(*t)).collect::<TractResult<_>>()?;
            self.to_unary(&*facts)?.with_context(|| {
                format!("Can not make {} into a typed op. (inputs facts: {:?})", prefix, facts)
            })?
        };
        target.wire_node(&*prefix, unary, &[inputs[0]])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::setup_test_logger;
    use tract_ndarray::*;

    #[test]
    fn test_infer_with_known_kshape() {
        let mut op = expand(Conv::default().strides(tvec![2, 2]).kernel_shape(tvec![3, 3]));
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 1, 7, 5));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 1, 3, 3));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(
            facts.1,
            tvec!(InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 1, 3, 2)))
        );
    }

    #[test]
    fn test_infer_channels() {
        let mut op = expand(Conv::default()); // NCHW - OIHW
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 2, 1, 1));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(3, 2, 1, 1));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(
            facts.1,
            tvec!(InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 3, 1, 1)))
        );
    }

    #[test]
    fn test_infer_onnx_strides_no_padding() {
        let mut op = expand(Conv::default().strides(tvec![2, 2]));
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 1, 7, 5));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 1, 3, 3));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(
            facts.1,
            tvec!(InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 1, 3, 2)))
        );
    }

    #[test]
    fn test_infer_nhwc_1() {
        let mut op = expand(Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper));
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 2, 2, 2));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(2, 2, 2, 1));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(
            facts.1,
            tvec!(InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 2, 2, 1)))
        );
    }

    #[test]
    fn test_eval_nhwc_1() {
        setup_test_logger();
        let op = expand(Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper));
        let res = op
            .eval(tvec!(
                ArrayD::<f32>::zeros(vec![1, 2, 2, 2]).into_arc_tensor(),
                ArrayD::<f32>::zeros(vec![2, 2, 2, 1]).into_arc_tensor()
            ))
            .unwrap();
        Tensor::from(ArrayD::<f32>::zeros(vec![1, 2, 2, 1])).close_enough(&res[0], false).unwrap()
    }

    #[test]
    fn test_infer_nhwc_2() {
        setup_test_logger();
        let mut op = expand(Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper));
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 1, 2, 2));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(2, 1, 2, 1));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(
            facts.1,
            tvec!(InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 1, 2, 1)))
        );
    }

    #[test]
    fn test_eval_nhwc_2() {
        setup_test_logger();
        let op = expand(Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper));
        let i = rctensor4(&[[[[0.0f32, 0.0], [1.0, 0.0]]]]);
        let k = rctensor4(&[[[[0.0f32], [0.0]], [[1.0], [0.0]]]]);
        let e = rctensor4(&[[[[1.0f32], [0.0]]]]);
        let res = op.eval(tvec!(i, k)).unwrap();
        assert_eq!(res, tvec!(e.into()));
    }

    #[test]
    fn test_eval_nhwc_3() {
        setup_test_logger();
        let op = expand(Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper));
        let i = rctensor4(&[[[[0.0f32, 1.0], [2.0, 3.0]], [[10.0, 11.0], [12.0, 13.0]]]]);
        let k = rctensor4(&[[[[1.0f32, 0.0], [0.0, 1.0]]]]);
        let res = op.eval(tvec!(i.clone(), k)).unwrap();
        assert_eq!(res, tvec!(i));
    }

    #[test]
    fn test_eval_nhwc_batch() {
        setup_test_logger();
        let op = expand(Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper));
        let result = op
            .eval(tvec!(rctensor4(&[[[[2.0f32]]], [[[0.0f32]]]]), rctensor4(&[[[[1.0f32]]]])))
            .unwrap();
        assert_eq!(result, tvec!(rctensor4(&[[[[2.0f32]]], [[[0.0f32]]]])));
    }

    #[test]
    fn test_infer_ntc_simple() {
        let mut op = expand(Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper));
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 2, 1));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 1, 1));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 2, 1))));
    }

    #[test]
    fn test_eval_ntc_simple() {
        let op = expand(Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper));
        let result =
            op.eval(tvec!(rctensor3(&[[[2.0f32], [0.0f32]]]), rctensor3(&[[[1.0f32]]]))).unwrap();
        assert_eq!(result, tvec!(rctensor3(&[[[2.0f32], [0.0f32]]])));
    }

    #[test]
    fn test_infer_ntc_batch() {
        let mut op = expand(Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper));
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(2, 1, 1));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 1, 1));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(InferenceFact::dt_shape(DatumType::F32, shapefactoid!(2, 1, 1))));
    }

    #[test]
    fn test_eval_ntc_batch() {
        let op = expand(Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper));
        let result =
            op.eval(tvec!(rctensor3(&[[[2.0f32]], [[0.0f32]]]), rctensor3(&[[[1.0f32]]]))).unwrap();
        assert_eq!(result, tvec!(rctensor3(&[[[2.0f32]], [[0.0f32]]])));
    }

    #[test]
    fn test_infer_ntc_channel() {
        let mut op = expand(Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper));
        let ifact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 1, 2));
        let kfact = InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 2, 1));
        let ofact = InferenceFact::default();
        let facts = op.infer_facts(tvec!(&ifact, &kfact), tvec!(&ofact), tvec!()).unwrap();
        assert_eq!(facts.1, tvec!(InferenceFact::dt_shape(DatumType::F32, shapefactoid!(1, 1, 1))));
    }

    #[test]
    fn test_eval_ntc_channel() {
        let op = expand(Conv::default().nhwc().hwio().padding(PaddingSpec::SameUpper));
        let result = op
            .eval(tvec!(rctensor3(&[[[2.0f32, 0.0f32]]]), rctensor3(&[[[1.0f32], [0.0f32]]])))
            .unwrap();
        assert_eq!(result, tvec!(rctensor3(&[[[2.0f32]]])));
    }
}
