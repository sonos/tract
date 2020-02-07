use crate::internal::*;
use crate::infer::*;

use crate::ops::cnn::{PaddingSpec, Patch, PatchSpec};
use crate::ops::nn::{DataFormat, DataShape};

#[derive(Debug, Clone, new, Default)]
pub struct PoolSpec {
    pub data_format: DataFormat,
    pub kernel_shape: TVec<usize>,
    pub padding: PaddingSpec,
    pub dilations: Option<TVec<usize>>,
    pub strides: Option<TVec<usize>>,
    pub output_channel_override: Option<usize>,
}

impl PoolSpec {
    pub fn info(&self) -> Vec<String> {
        vec![
            format!("Data format: {:?}", self.data_format),
            format!(
                "Kernel shape:{:?} (strides:{:?}, padding:{:?}, dilations:{:?})",
                self.kernel_shape, self.strides, self.padding, self.dilations,
            ),
        ]
    }

    pub fn dilation(&self, geo_axis: usize) -> usize {
        self.dilations.as_ref().map(|d| d[geo_axis]).unwrap_or(1)
    }

    pub fn stride(&self, geo_axis: usize) -> usize {
        self.strides.as_ref().map(|s| s[geo_axis]).unwrap_or(1)
    }

    pub fn compute_geo(&self, input_full_shape: &[usize]) -> (DataShape, Patch, DataShape) {
        let input_shape = self.data_format.shape(input_full_shape.into());
        let output_inner_stride = match self.data_format {
            DataFormat::NCHW|DataFormat::CHW => 1,
            DataFormat::NHWC|DataFormat::HWC => self.output_channel_override.clone().unwrap_or(*input_shape.c()),
        };
        let mut spec = PatchSpec::for_full_shape(self.data_format, input_full_shape)
            .with_output_inner_stride(output_inner_stride)
            .with_kernel_shape(self.kernel_shape.clone())
            .with_padding(self.padding.clone());
        if let Some(strides) = self.strides.clone() {
            spec = spec.with_strides(strides);
        }
        if let Some(dilations) = self.dilations.clone() {
            spec = spec.with_dilations(dilations);
        }
        let patch = spec.into_patch();
        let output_shape = input_shape.fmt.from_n_c_hw(
            *input_shape.n().unwrap_or(&1),
            self.output_channel_override.unwrap_or(*input_shape.c()),
            &*patch.output_shape,
        );
        (input_shape, patch, output_shape)
    }

    pub fn rules_for_shape<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.given(&inputs[0].shape, move |s, ishape| {
            let ishape = self.data_format.shape(ishape);
            let ones = tvec![1; ishape.hw_rank()];
            let computed = self.padding.compute(
                ishape.hw_dims(),
                &*self.kernel_shape,
                self.dilations.as_ref().unwrap_or(&ones),
                self.strides.as_ref().unwrap_or(&ones),
            );
            for o in 0..outputs.len() {
                for (ix, d) in computed.iter().enumerate() {
                    s.equals(&outputs[o].shape[ix + ishape.h_axis()], &d.output)?;
                }
                if ishape.n_axis().is_some() {
                    s.equals(&outputs[o].shape[ishape.n_axis().unwrap()], ishape.n_dim().unwrap())?;
                }
                if let Some(c) = self.output_channel_override {
                    s.equals(&outputs[o].shape[ishape.c_axis()], c.to_dim())?;
                } else {
                    s.equals(&outputs[o].shape[ishape.c_axis()], ishape.c_dim())?;
                }
            }
            Ok(())
        })
    }

    pub fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let ishape = self.data_format.shape(inputs[0].shape.to_tvec());
        let ones = tvec![1; ishape.hw_rank()];
        let computed = self.padding.compute(
            ishape.hw_dims(),
            &*self.kernel_shape,
            self.dilations.as_ref().unwrap_or(&ones),
            self.strides.as_ref().unwrap_or(&ones),
        );
        let spatial_dims = computed.into_iter().map(|d| d.output).collect::<TVec<TDim>>();
        let oshape = self.data_format.from_n_c_hw(
            ishape.n().cloned().unwrap_or(1.to_dim()),
            self.output_channel_override.map(|i| i.to_dim()).unwrap_or(ishape.c().clone()),
            spatial_dims,
        );
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*oshape.shape)?))
    }

    pub fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        op: &dyn PulsedOp,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let fact = target.outlet_fact(input)?.clone();
        let input_shape = self.data_format.shape(&*fact.shape);
        if Some(fact.axis) == input_shape.n_axis() {
            target.wire_node(&*node.name, dyn_clone::clone_box(op), &[input])
        } else if fact.axis == input_shape.c_axis() {
            bail!("Can not pulsify cnn pooling ops along the input channel axis");
        } else {
            let geo_axis = fact.axis - input_shape.h_axis();
            let stride = self.strides.as_ref().and_then(|v| v.get(geo_axis).cloned()).unwrap_or(1);
            let pulse = fact.pulse();
            if fact.pulse() % stride != 0 {
                bail!("Pulsificaton requires pulse to be a stride multiple")
            }
            let dilation = self.dilations.as_ref().map(|d| d[geo_axis]).unwrap_or(1);
            let kernel_len = (self.kernel_shape[geo_axis] - 1) * dilation;
            let overlap = (kernel_len + 1).saturating_sub(stride);
            let misalignment = fact.delay % pulse;
            let mut wire = input;

            if overlap > 0 || misalignment > 0 {
                let align_to = (overlap + fact.delay).div_ceil(stride) * stride;
                let delay = align_to - overlap - fact.delay;
                wire = target.wire_node(
                    format!("{}/Delay", node.name),
                    crate::pulse::delay::Delay::new(&fact, delay, overlap),
                    &[wire],
                )?[0];
            }
            target.wire_node(&*node.name, dyn_clone::clone_box(op), &[wire])
        }
    }

    pub fn dispose_n_axis(&self) -> PoolSpec {
        PoolSpec {
            data_format: self.data_format.dispose_n_axis(),
            .. self.clone()
        }
    }

    pub fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let ishape = self.data_format.shape(&inputs[0].shape);
        let ones = tvec![1; ishape.hw_rank()];
        let computed = self.padding.compute(
            ishape.hw_dims(),
            &*self.kernel_shape,
            self.dilations.as_ref().unwrap_or(&ones),
            self.strides.as_ref().unwrap_or(&ones),
        );
        let spatial_dims = computed.into_iter().map(|d| d.output).collect::<TVec<usize>>();
        let oshape = self.data_format.from_n_c_hw(
            ishape.n().cloned().unwrap_or(1),
            self.output_channel_override.unwrap_or(*ishape.c()),
            spatial_dims,
        );
        let mut fact = inputs[0].clone();
        let input_shape = self.data_format.shape(&*fact.shape);
        let geo_axis = fact.axis - input_shape.h_axis();
        let dilation = self.dilations.as_ref().map(|d| d[geo_axis]).unwrap_or(1);
        let kernel_len = (self.kernel_shape[geo_axis] - 1) * dilation;
        let stride = self.strides.as_ref().and_then(|v| v.get(geo_axis).cloned()).unwrap_or(1);
        fact.delay /= stride;
        fact.dim = (fact.dim.clone() - kernel_len.to_dim()).div_ceil(stride.to_dim());
        fact.shape = oshape.shape;
        Ok(tvec!(fact))
    }
}
