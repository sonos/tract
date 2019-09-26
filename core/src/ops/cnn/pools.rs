use crate::internal::*;

use crate::ops::cnn::{PaddingSpec, Patch, PatchSpec};
use crate::ops::nn::{DataFormat, DataShape};

#[derive(Debug, Clone, new, Default)]
pub struct PoolSpec {
    data_format: DataFormat,
    kernel_shape: TVec<usize>,
    padding: PaddingSpec,
    strides: Option<TVec<usize>>,
}

impl PoolSpec {
    pub fn info(&self) -> Vec<String> {
        vec![
            format!("Data format: {:?}", self.data_format),
            format!(
                "Kernel shape:{:?} (strides:{:?}, padding:{:?})",
                self.kernel_shape, self.strides, self.padding,
            ),
        ]
    }

    pub fn compute_geo(&self, input_full_shape: &[usize]) -> (DataShape, Patch, DataShape) {
        let input_shape = self.data_format.shape(input_full_shape.into());
        let mut spec = PatchSpec::for_full_shape(self.data_format, input_full_shape)
            .with_output_inner_stride(*input_shape.w_stride())
            .with_kernel_shape(self.kernel_shape.clone())
            .with_padding(self.padding.clone());
        if let Some(strides) = self.strides.clone() {
            spec = spec.with_strides(strides);
        }
        let patch = spec.into_patch();
        let output_shape =
            input_shape.fmt.from_n_c_hw(*input_shape.n(), *input_shape.c(), &*patch.output_shape);
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
                &ones,
                self.strides.as_ref().unwrap_or(&ones),
            );
            for o in 0..outputs.len() {
                for (ix, d) in computed.iter().enumerate() {
                    s.equals(&outputs[o].shape[ix + ishape.h_axis()], &d.output)?;
                }
                s.equals(&outputs[o].shape[ishape.n_axis()], ishape.n_dim())?;
                s.equals(&outputs[o].shape[ishape.c_axis()], ishape.c_dim())?;
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
            &ones,
            self.strides.as_ref().unwrap_or(&ones),
        );
        let spatial_dims = computed.into_iter().map(|d| d.output).collect::<TVec<TDim>>();
        let oshape =
            self.data_format.from_n_c_hw(ishape.n().clone(), ishape.c().clone(), spatial_dims);
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
        if fact.axis == input_shape.n_axis() {
            target.wire_node(&*node.name, objekt::clone_box(op), &[input])
        } else if fact.axis == input_shape.c_axis() {
            bail!("Can not pulsify cnn pooling ops along the input channel axis");
        } else {
            let geo_axis = fact.axis - input_shape.h_axis();
            let stride = self.strides.as_ref().and_then(|v| v.get(geo_axis).cloned()).unwrap_or(1);
            let pulse = fact.pulse();
            if fact.pulse() % stride != 0 {
                bail!("Pulsificaton requires pulse to be a stride multiple")
            }
            let dilation = 1;
            let kernel_len = (self.kernel_shape[geo_axis] - 1) * dilation;
            let overlap = (kernel_len + 1).saturating_sub(stride);
            let misalignment = fact.delay % pulse;
            let mut wire = input;

            if overlap > 0 || misalignment > 0 {
                let align_to = (overlap + fact.delay).div_ceil(pulse) * pulse;
                let delay = align_to - overlap - fact.delay;
                wire = target.wire_node(
                    format!("{}/Delay", node.name),
                    crate::pulse::delay::Delay::new(&fact, delay, overlap),
                    &[wire],
                )?[0];
            }
            target.wire_node(&*node.name, objekt::clone_box(op), &[wire])
        }
    }

    pub fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let ishape = self.data_format.shape(&inputs[0].shape);
        let ones = tvec![1; ishape.hw_rank()];
        let computed = self.padding.compute(
            ishape.hw_dims(),
            &*self.kernel_shape,
            &ones,
            self.strides.as_ref().unwrap_or(&ones),
        );
        let spatial_dims = computed.into_iter().map(|d| d.output).collect::<TVec<usize>>();
        let oshape =
            self.data_format.from_n_c_hw(ishape.n().clone(), ishape.c().clone(), spatial_dims);
        let mut fact = inputs[0].clone();
        let input_shape = self.data_format.shape(&*fact.shape);
        let geo_axis = fact.axis - input_shape.h_axis();
        let dilation = 1;
        let kernel_len = (self.kernel_shape[geo_axis] - 1) * dilation;
        let stride = self.strides.as_ref().and_then(|v| v.get(geo_axis).cloned()).unwrap_or(1);
        fact.delay /= stride;
        fact.dim = (fact.dim.clone() - kernel_len.to_dim()).div_ceil(stride.to_dim());
        fact.shape = oshape.shape;
        Ok(tvec!(fact))
    }
}
