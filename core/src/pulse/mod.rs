use crate::datum::TryInto;
use crate::ops::prelude::*;
use crate::ops::source::Source;
use std::fmt;

pub mod delay;

#[derive(Clone, PartialEq)]
pub struct PulsedTensorFact {
    pub dt: DatumType,
    pub shape: TVec<usize>,
    pub axis: usize,
    pub dim: TDim,
    pub delay: usize,
}

impl fmt::Debug for PulsedTensorFact {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use itertools::Itertools;
        write!(
            fmt,
            "Pulse:{}x{:?}{{ax:{},d:{},dim:{:?}}}",
            self.shape.iter().join("x"),
            self.dt,
            self.axis,
            self.delay,
            self.dim
        )
    }
}

impl TensorInfo for PulsedTensorFact {
    fn to_tensor_fact(&self) -> TensorFact {
        TensorFact::dt_shape(self.dt, &self.shape)
    }
}

impl TryInto<TypedTensorInfo> for PulsedTensorFact {
    fn try_into(&self) -> TractResult<TypedTensorInfo> {
        Ok(TypedTensorInfo { shape: (&self.shape).into(), datum_type: self.dt, konst: None })
    }
}

impl PulsedTensorFact {
    pub fn from_tensor_fact_pulse(
        tf: &NormalizedTensorInfo,
        pulse: usize,
    ) -> TractResult<PulsedTensorFact> {
        let dt = tf.datum_type;
        let stream = tf.shape.stream_info.ok_or("Can not pulse a tensor with no streaming dim")?;
        let shape =
            tf.shape.iter().map(|d| d.to_integer().map(|d| d as usize).unwrap_or(pulse)).collect();
        Ok(PulsedTensorFact { dt, shape, axis: stream.axis, dim: stream.len, delay: 0 })
    }

    pub fn pulse(&self) -> usize {
        self.shape[self.axis]
    }

    pub fn to_pulse_fact(&self) -> NormalizedTensorInfo {
        NormalizedTensorInfo { datum_type: self.dt, shape: ShapeInfo::from(&*self.shape) }
    }

    pub fn streaming_shape(&self) -> Vec<TDim> {
        self.shape
            .iter()
            .enumerate()
            .map(|(ix, &d)| if ix == self.axis { self.dim } else { d.to_dim() })
            .collect()
    }

    pub fn to_streaming_fact(&self) -> NormalizedTensorInfo {
        let mut info = self.to_pulse_fact();
        info.shape.stream_info = Some(StreamInfo { axis: self.axis, len: self.dim });
        info
    }
}

pub type PulsedModel = Model<PulsedTensorFact>;

impl PulsedModel {
    pub fn new(source: &NormalizedModel, pulse: usize) -> TractResult<PulsedModel> {
        Ok(PulsedModel::new_with_mapping(source, pulse)?.0)
    }

    pub fn new_with_mapping(
        source: &NormalizedModel,
        pulse: usize,
    ) -> TractResult<(PulsedModel, HashMap<OutletId, OutletId>)> {
        let mut target = PulsedModel::default();
        let mut mapping = HashMap::new();
        for old_id in source.eval_order()? {
            trace!(
                "Pulsify node {} {} ({})",
                old_id,
                source.node(old_id).name,
                source.node(old_id).op().name()
            );
            if source.node(old_id).op_as::<Source>().is_some() {
                let node = source.node(old_id);
                let pulsed_fact =
                    PulsedTensorFact::from_tensor_fact_pulse(&node.outputs[0].fact, pulse)?;
                let id = target.add_source(node.name.clone(), pulsed_fact)?;
                mapping.insert(OutletId::new(old_id, 0), OutletId::new(id, 0));
            } else {
                let node = &source.nodes()[old_id];
                let outlets = node.op().pulsify(&source, &node, &mut target, &mapping)?;
                for (ix, outlet) in outlets.into_iter().enumerate() {
                    mapping.insert(OutletId::new(node.id, ix), outlet);
                }
            }
            trace!("Target is now {}", target.nodes().len());
        }
        // maintaining order of i/o interface
        target.inputs = source.inputs()?.iter().map(|i| mapping[&i]).collect();
        target.outputs = source.outputs()?.iter().map(|o| mapping[&o]).collect();
        Ok((target, mapping))
    }

    pub fn into_typed(self) -> TractResult<TypedModel> {
        crate::model::compact::compact(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use ndarray::*;
    use proptest::proptest;
    use proptest::test_runner::TestCaseResult;
    use proptest::*;

    #[test]
    fn test_source_must_stream() {
        let mut model = Model::default();
        let _a =
            model.add_source("a", TensorFact::dt_shape(DatumType::F32, vec![1, 2, 3])).unwrap();
        assert!(
            PulsedModel::new(&model.into_typed().unwrap().into_normalized().unwrap(), 4).is_err()
        );

        let mut model = Model::default();
        let _a = model
            .add_source(
                "a",
                TensorFact::dt_shape(DatumType::F32, vec![1.to_dim(), TDim::s(), 3.to_dim()]),
            )
            .unwrap();
        let pulse =
            PulsedModel::new(&model.into_typed().unwrap().into_normalized().unwrap(), 4).unwrap();
        assert_eq!(
            pulse.fact(OutletId::new(0, 0)).unwrap().to_tensor_fact(),
            TensorFact::dt_shape(DatumType::F32, vec!(1, 4, 3))
        );
    }

    #[test]
    fn test_immediate() {
        let mut model = Model::default();
        let _a = model
            .add_source(
                "a",
                TensorFact::dt_shape(DatumType::F32, vec![TDim::s(), 2.to_dim(), 3.to_dim()]),
            )
            .unwrap();

        let pulse = PulsedModel::new(&model.into_normalized().unwrap(), 4).unwrap();

        assert_eq!(
            pulse.input_fact().unwrap().to_tensor_fact(),
            TensorFact::dt_shape(DatumType::F32, vec!(4, 2, 3))
        );
        assert_eq!(
            pulse.output_fact().unwrap().to_tensor_fact(),
            TensorFact::dt_shape(DatumType::F32, vec!(4, 2, 3))
        );
    }

    fn proptest_regular_against_pulse(
        model: InferenceModel,
        pulse: usize,
        input_array: ArrayD<f32>,
        axis: usize,
    ) -> TestCaseResult {
        let input = Tensor::from(input_array.clone());
        let plan = SimplePlan::new(&model).unwrap();
        let outputs = plan.run(tvec!(input.clone())).unwrap();
        let model = model.into_normalized().unwrap();

        let pulsed = PulsedModel::new(&model, pulse).unwrap();
        let output_fact = pulsed.output_fact().unwrap().clone();

        let output_stream_axis = output_fact.axis;
        let delay = output_fact.delay;
        let mut initial_output_shape = output_fact.shape.clone();
        initial_output_shape[output_stream_axis] = 0;

        let pulsed_plan = crate::plan::SimplePlan::new(pulsed).unwrap();
        let mut state = crate::plan::SimpleState::new(&pulsed_plan).unwrap();

        let mut got: ArrayD<f32> = ArrayD::zeros(&*initial_output_shape);
        let mut output_len = None;

        let mut written = 0;
        loop {
            let to_write_in_chunk = pulse.min(input_array.shape()[axis].saturating_sub(written));
            let mut chunk: ArrayD<f32> = input_array
                .slice_axis(Axis(axis), (written..written + to_write_in_chunk).into())
                .to_owned();
            written += to_write_in_chunk;
            if to_write_in_chunk < pulse {
                let mut filler_shape = input_array.shape().to_vec();
                filler_shape[axis] = pulse - to_write_in_chunk;
                chunk = stack(
                    Axis(axis),
                    &[chunk.view(), ArrayD::from_elem(filler_shape, std::f32::NAN).view()],
                )
                .unwrap();
                output_len = output_fact.dim.eval(written as _);
                state.session_state.known_stream_len = Some(written)
            }
            let mut outputs = state.run(tvec!(Tensor::from(chunk.to_owned()).into())).unwrap();
            got = stack(
                Axis(output_stream_axis),
                &[got.view(), outputs.remove(0).to_array_view::<f32>().unwrap()],
            )
            .unwrap();
            if let Some(output_len) = output_len {
                if got.shape()[output_stream_axis] >= output_len as usize + delay {
                    break;
                }
            }
        }

        let pulsed_output = got.slice_axis(
            Axis(output_stream_axis),
            (output_fact.delay..output_fact.delay + output_len.unwrap() as usize).into(),
        );

        prop_assert_eq!(pulsed_output, outputs[0].to_array_view::<f32>().unwrap());
        Ok(())
    }

    proptest! {
        #[test]
        fn proptest_crop(pulse in 1i32..3, input_len in 0i32..10, begin in 0i32..3, end in 0i32..3) {
            use crate::ops::array::Slice;
            let input_len = input_len + begin + end;
            let mut model = Model::default();
            let _ = model
                .add_source("a", TensorFact::dt_shape(f32::datum_type(), shapefact!(S)))
                .unwrap();
            model.chain_default("slice", Slice::new(vec![(begin as usize, end as usize)])).unwrap();

            let input = Array1::range(1.0f32, input_len as f32 + 1.0, 1.0);
            proptest_regular_against_pulse(model, pulse as _, input.into_dyn(), 0)?;
        }

        #[test]
        fn proptest_pad(pulse in 1i32..3, input_len in 0i32..10, begin in 0i32..3, end in 0i32..3) {
            use crate::ops::array::{ Pad, PadMode };
            let mut model = Model::default();
            let _ = model
                .add_source("a", TensorFact::dt_shape(f32::datum_type(), shapefact!(S)))
                .unwrap();
            model.chain_default("pad", Pad::new(vec![(begin as _, end as _)], PadMode::Constant(-1.0))).unwrap();

            let input = Array1::range(1.0f32, input_len as f32 + 1.0, 1.0);
            proptest_regular_against_pulse(model, pulse as _, input.into_dyn(), 0)?;
        }

    }

    #[test]
    fn test_simple_conv() {
        use crate::ops::nn::*;

        let mut model = Model::default();
        let ker = model.add_const("kernel", arr3(&[[[0.5f32, 1.0, -0.1]]]).into()).unwrap();
        let _ = model
            .add_source("a", TensorFact::dt_shape(f32::datum_type(), shapefact!(1, 1, S))) // NCT
            .unwrap();

        let conv = model.chain_default("conv", Conv::default()).unwrap();
        model.add_edge(OutletId::new(ker, 0), InletId::new(conv, 1)).unwrap();

        let input = arr3(&[[[1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]]]);
        proptest_regular_against_pulse(model, 4, input.into_dyn(), 2).unwrap();
    }

    #[test]
    fn test_pad_after_1() {
        use crate::ops::array::{Pad, PadMode};
        let mut model = Model::default();
        let _ =
            model.add_source("a", TensorFact::dt_shape(f32::datum_type(), shapefact!(S))).unwrap();
        model.chain_default("pad", Pad::new(vec![(0, 1)], PadMode::Constant(-1.0))).unwrap();

        let input = arr1(&[]);
        proptest_regular_against_pulse(model, 1, input.into_dyn(), 0).unwrap();
    }

    #[test]
    fn test_pad_before_1() {
        use crate::ops::array::{Pad, PadMode};
        let mut model = Model::default();
        let _ =
            model.add_source("a", TensorFact::dt_shape(f32::datum_type(), shapefact!(S))).unwrap();
        model.chain_default("pad", Pad::new(vec![(1, 0)], PadMode::Constant(-1.0))).unwrap();

        let input = arr1(&[1.0]);
        proptest_regular_against_pulse(model, 1, input.into_dyn(), 0).unwrap();
    }

    #[test]
    fn test_pad_before_2() {
        use crate::ops::array::{Pad, PadMode};
        let mut model = Model::default();
        let _ =
            model.add_source("a", TensorFact::dt_shape(f32::datum_type(), shapefact!(S))).unwrap();
        model.chain_default("pad", Pad::new(vec![(1, 0)], PadMode::Constant(-1.0))).unwrap();

        let input = arr1(&[1.0, 2.0]);
        proptest_regular_against_pulse(model, 2, input.into_dyn(), 0).unwrap();
    }

}
