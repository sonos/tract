use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use tract_hir::internal::*;
use tract_hir::ops::array::Pad;
use tract_hir::ops::cast::cast;
use tract_hir::ops::math::div;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("DFT", dft);
    reg.insert("STFT", stft);
    reg.insert("MelWeightMatrix", mel_weight_matrix);
    reg.insert("BlackmanWindow", window);
    reg.insert("HammingWindow", window);
    reg.insert("HannWindow", window);
}

fn dft(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    let inverse = node.get_attr_opt("inverse")?.unwrap_or(0i64) != 0;
    let onesided = node.get_attr_opt("onesided")?.unwrap_or(0) != 0;
    if node.input.len() > 1 {
        bail!("length input is not implemented")
    }
    Ok((expand(Dft { axis, inverse, onesided, has_length_input: node.input.len() == 2 }), vec![]))
}

fn stft(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let onesided = node.get_attr_opt("onesided")?.unwrap_or(1) != 0;
    let mut options = crate::model::optional_inputs(node).skip(2);
    Ok((
        expand(Stft {
            onesided,
            optional_window_input: options.next().unwrap(),
            optional_frame_length_input: options.next().unwrap(),
        }),
        vec![],
    ))
}

fn mel_weight_matrix(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let datum_type = node.get_attr_opt("output_datatype")?.unwrap_or(DatumType::F32);
    Ok((expand(MelWeightMatrix { datum_type }), vec![]))
}

fn window(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let datum_type = node.get_attr_opt("output_datatype")?.unwrap_or(DatumType::F32);
    let periodic = node.get_attr_opt("periodic")?.unwrap_or(1i64) == 1i64;
    let window = match &*node.op_type {
        "BlackmanWindow" => StftWindowType::Blackman,
        "HammingWindow" => StftWindowType::Hamming,
        "HannWindow" => StftWindowType::Hann,
        _ => unreachable!(),
    };
    Ok((expand(StftWindow { datum_type, periodic, window }), vec![]))
}

#[derive(Clone, Debug, Hash)]
struct Dft {
    axis: usize,
    inverse: bool,
    onesided: bool,
    has_length_input: bool,
}



impl Expansion for Dft {
    fn name(&self) -> Cow<str> {
        "DFT".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1 + self.has_length_input as usize)?;
        check_output_arity(outputs, 1)?;

        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        if self.has_length_input {
            s.equals(&inputs[1].rank, 0)?;
        }
        s.given(&inputs[0].rank, |s, rank| {
            for ax in 0..rank as usize - 1 {
                if ax != self.axis {
                    s.equals(&inputs[0].shape[ax], &outputs[0].shape[ax])?;
                }
            }
            s.equals(&outputs[0].shape[rank as usize - 1], 2.to_dim())?;
            Ok(())
        })?;
        if self.has_length_input {
            s.given(&inputs[1].value[0], |s, len| {
                s.equals(len.to_dim(), &outputs[0].shape[self.axis])
            })?;
        } else {
            s.equals(&inputs[0].shape[self.axis], &outputs[0].shape[self.axis])?;
        }
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let fact = model.outlet_fact(inputs[0])?.clone();
        let mut wire: TVec<OutletId> = inputs.into();
        if fact.shape.last() == Some(&1.to_dim()) {
            let mut pads = vec![(0, 0); fact.rank() - 1];
            pads.push((0, 1));
            wire = model.wire_node(
                format!("{prefix}.add_imaginary"),
                Pad { mode: tract_hir::ops::array::PadMode::Constant(rctensor0(0f32)), pads },
                &wire,
            )?;
        };
        wire = model.wire_node(
            format!("{prefix}.fft"),
            tract_core::ops::fft::Fft { axis: self.axis, inverse: self.inverse },
            &wire,
        )?;
        if self.inverse {
            let len = model.add_const(
                format!("{prefix}.len"),
                tensor0(fact.shape[self.axis].clone()).broadcast_into_rank(fact.rank())?,
            )?;
            let casted =
                model.wire_node(format!("{prefix}.cast"), cast(fact.datum_type), &[len])?;
            wire = model.wire_node(format!("{prefix}.norm"), div(), &[wire[0], casted[0]])?;
        }
        if self.onesided {
            let frame = fact.shape[self.axis].clone() / 2 + 1;
            wire = model.wire_node(
                format!("{prefix}.onesided"),
                tract_core::ops::array::Slice::new(2, 0, frame),
                &wire,
            )?;
        }
        Ok(wire)
    }
}

#[derive(Clone, Debug, Hash)]
struct Stft {
    onesided: bool,
    optional_window_input: Option<usize>,
    optional_frame_length_input: Option<usize>,
}



impl Expansion for Stft {
    fn name(&self) -> Cow<str> {
        "STFT".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(
            inputs,
            2 + self.optional_window_input.is_some() as usize
                + self.optional_frame_length_input.is_some() as usize,
        )?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, 3)?;
        s.equals(&outputs[0].rank, 4)?;
        s.equals(&outputs[0].shape[0], &inputs[0].shape[0])?;
        s.equals(&outputs[0].shape[3], 2.to_dim())?;
        let mut frame_len = None;
        let mut frame_len_2 = None; // ugly ! but exps are not clonable
        if let Some(l) = self.optional_frame_length_input {
            s.equals(&inputs[l].datum_type, i64::datum_type())?;
            s.equals(&inputs[l].rank, 0)?;
            frame_len = Some(inputs[l].value[0].bex().to_dim());
            frame_len_2 = Some(inputs[l].value[0].bex().to_dim());
        }
        if let Some(w) = self.optional_window_input {
            s.equals(&inputs[w].datum_type, &inputs[0].datum_type)?;
            s.equals(&inputs[w].rank, 1)?;
            frame_len = Some(inputs[w].shape[0].bex());
            frame_len_2 = Some(inputs[w].shape[0].bex());
        }
        if let (Some(w), Some(l)) = (self.optional_window_input, self.optional_frame_length_input) {
            s.equals(inputs[l].value[0].bex().to_dim(), &inputs[w].shape[0])?;
        }
        if let Some(frame_len) = frame_len {
            s.given_3(
                &inputs[0].shape[1],
                frame_len,
                &inputs[1].value[0],
                |s, signal, frame, stride| {
                    let frames = (signal - frame) / stride + 1;
                    s.equals(&outputs[0].shape[1], frames)?;
                    Ok(())
                },
            )?;
        }
        if let Some(frame_len) = frame_len_2 {
            s.given(frame_len, |s, frame_len| {
                let fst_len = if self.onesided { frame_len / 2 + 1 } else { frame_len };
                s.equals(&outputs[0].shape[2], fst_len)
            })?;
        }
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let fact = model.outlet_fact(inputs[0])?.clone();
        let mut wire: TVec<OutletId> = tvec!(inputs[0]);
        let (frame, window) = if let Some(w) = self.optional_window_input {
            let window = model
                .outlet_fact(inputs[w])?
                .konst
                .clone()
                .context("STFT expects a constant window")?;
            (window.len(), Some(window))
        } else if let Some(fl) = self.optional_frame_length_input {
            let frame = model
                .outlet_fact(inputs[fl])?
                .konst
                .as_ref()
                .context("STFT expects a constant frame length")?
                .cast_to_scalar::<i64>()? as usize;
            (frame, None)
        } else {
            bail!("Need window or frame len")
        };
        let stride = model
            .outlet_fact(inputs[1])?
            .konst
            .as_ref()
            .context("STFT expects a constant frame_step")?
            .cast_to_scalar::<i64>()? as usize;
        if fact.shape.last() == Some(&1.to_dim()) {
            let mut pads = vec![(0, 0); fact.rank() - 1];
            pads.push((0, 1));
            wire = model.wire_node(
                format!("{prefix}.add_imaginary"),
                Pad { mode: tract_hir::ops::array::PadMode::Constant(rctensor0(0f32)), pads },
                &wire,
            )?;
        };
        wire = model.wire_node(
            format!("{prefix}.fft"),
            tract_core::ops::fft::Stft { axis: 1, frame, window, stride },
            &wire,
        )?;
        if self.onesided {
            wire = model.wire_node(
                format!("{prefix}.onesided"),
                tract_core::ops::array::Slice::new(2, 0, frame / 2 + 1),
                &wire,
            )?;
        }
        Ok(wire)
    }
}

#[derive(Clone, Debug, Hash)]
pub struct MelWeightMatrix {
    datum_type: DatumType,
}



impl Expansion for MelWeightMatrix {
    fn name(&self) -> Cow<str> {
        "MelWeightMatrix".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 5)?;
        check_output_arity(outputs, 1)?;
        for input in inputs {
            s.equals(&input.rank, 0)?;
        }
        s.equals(&outputs[0].datum_type, self.datum_type)?;
        s.equals(&outputs[0].rank, 2)?;
        s.given(&inputs[1].value[0], |s, dft_length| {
            s.equals(&outputs[0].shape[0], (dft_length / 2 + 1).to_dim())
        })?;
        s.given(&inputs[0].value[0], |s, num_mel_bins| {
            s.equals(&outputs[0].shape[1], num_mel_bins.to_dim())
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let (
            Some(num_mel_bins),
            Some(dft_length),
            Some(sample_rate),
            Some(lower_edge_hertz),
            Some(upper_edge_hertz),
            ) = (
                model.outlet_fact(inputs[0])?.konst.as_ref(),
                model.outlet_fact(inputs[1])?.konst.as_ref(),
                model.outlet_fact(inputs[2])?.konst.as_ref(),
                model.outlet_fact(inputs[3])?.konst.as_ref(),
                model.outlet_fact(inputs[4])?.konst.as_ref(),
                ) else {
                bail!("Expect all inputs to be constants")
            };
        let num_mel_bins = num_mel_bins.cast_to_scalar::<i64>()? as usize;
        let dft_length = dft_length.cast_to_scalar::<i64>()? as usize;
        let sample_rate = sample_rate.cast_to_scalar::<i64>()? as usize;
        let lower_edge_hertz = lower_edge_hertz.cast_to_scalar::<f32>()?;
        let upper_edge_hertz = upper_edge_hertz.cast_to_scalar::<f32>()?;

        let num_spectrogram_bins = dft_length / 2 + 1;
        let low_frequency_mel = 2595. * (1. + lower_edge_hertz / 700.).log10();
        let high_frequency_mel = 2595. * (1. + upper_edge_hertz / 700.).log10();
        let mel_step = (high_frequency_mel - low_frequency_mel) / (num_mel_bins + 2) as f32;

        let frequency_bins: Vec<usize> = (0..num_mel_bins + 2)
            .map(|ix| {
                let freq = ix as f32 * mel_step + low_frequency_mel;
                let freq = 700. * (10f32.powf(freq / 2596.) - 1.);
                let freq = ((dft_length + 1) as f32 * freq) / sample_rate as f32;
                freq as usize
            })
            .collect();

        let mut output = Tensor::zero::<f32>(&[num_spectrogram_bins, num_mel_bins])?;
        let mut view = output.to_array_view_mut::<f32>()?.into_dimensionality()?;
        for i in 0..num_mel_bins {
            let lower = frequency_bins[i];
            let center = frequency_bins[i + 1];
            let higher = frequency_bins[i + 2];
            if center == lower {
                view[(center, i)] = 1.;
            } else {
                for j in lower..center + 1 {
                    view[(j, i)] = (j - lower) as f32 / (center - lower) as f32;
                }
            }
            if higher > center {
                for j in center..higher {
                    view[(j, i)] = (higher - j) as f32 / (higher - center) as f32;
                }
            }
        }
        let wire = model.add_const(prefix, output.cast_to_dt(self.datum_type)?.into_owned())?;
        Ok(tvec!(wire))
    }
}

#[derive(Copy, Clone, Debug, Hash)]
enum StftWindowType {
    Blackman,
    Hamming,
    Hann,
}

impl StftWindowType {
    fn generate(&self, size: usize, periodic: bool) -> TractResult<Tensor> {
        use std::f32::consts::PI;
        let divisor = ((size - 1 + periodic as usize) as f32).recip();
        let mut output = Tensor::zero::<f32>(&[size])?;
        match self {
            Self::Blackman => {
                output.as_slice_mut::<f32>()?.iter_mut().enumerate().for_each(|(ix, y)| {
                    *y = 0.42 - 0.5 * (2. * PI * ix as f32 * divisor).cos()
                        + 0.08 * (4. * PI * ix as f32 * divisor).cos()
                })
            }
            Self::Hamming => {
                output.as_slice_mut::<f32>()?.iter_mut().enumerate().for_each(|(ix, y)| {
                    *y = (25. / 46.) - (21. / 46.) * (2. * PI * ix as f32 * divisor).cos()
                })
            }
            Self::Hann => output
                .as_slice_mut::<f32>()?
                .iter_mut()
                .enumerate()
                .for_each(|(ix, y)| *y = 0.5 - 0.5 * (2. * PI * ix as f32 * divisor).cos()),
        }
        Ok(output)
    }
}

#[derive(Clone, Debug, Hash)]
pub struct StftWindow {
    datum_type: DatumType,
    periodic: bool,
    window: StftWindowType,
}



impl Expansion for StftWindow {
    fn name(&self) -> Cow<str> {
        format!("StftWindow<{:?}>", self.window).into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].rank, 0)?;
        s.equals(&outputs[0].datum_type, self.datum_type)?;
        s.equals(&outputs[0].rank, 1)?;
        s.given(&inputs[0].value[0], |s, length| s.equals(&outputs[0].shape[0], length.to_dim()))
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let len = model
            .outlet_fact(inputs[0])?
            .konst
            .as_ref()
            .context("Expect constant input size")?
            .cast_to_scalar::<i64>()? as usize;
        let window =
            self.window.generate(len, self.periodic)?.cast_to_dt(self.datum_type)?.into_owned();
        let wire = model.add_const(prefix, window)?;
        Ok(tvec!(wire))
    }
}
