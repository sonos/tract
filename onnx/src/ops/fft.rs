use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::NodeProto;
use tract_hir::internal::*;
use tract_hir::ops::array::Pad;
use tract_hir::ops::cast::cast;
use tract_hir::ops::math::div;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("DFT", dft);
    reg.insert("STFT", stft);
}

pub fn dft(
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

pub fn stft(
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

#[derive(Clone, Debug, Hash)]
struct Dft {
    axis: usize,
    inverse: bool,
    onesided: bool,
    has_length_input: bool,
}

impl_dyn_hash!(Dft);

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
                format!("{}.add_imaginary", prefix),
                Pad { mode: tract_hir::ops::array::PadMode::Constant(rctensor0(0f32)), pads },
                &wire,
            )?;
        };
        wire = model.wire_node(
            format!("{}.pair_to_cplx", prefix),
            tract_core::ops::math::InnerDimToComplex,
            &wire,
        )?;
        wire = model.wire_node(
            format!("{}.fft", prefix),
            tract_core::ops::fft::Fft { axis: self.axis, inverse: self.inverse },
            &wire,
        )?;
        wire = model.wire_node(
            format!("{}.to_pair", prefix),
            tract_core::ops::math::ComplexToInnerDim,
            &wire,
        )?;
        if self.inverse {
            let len = model.add_const(
                format!("{}.len", prefix),
                tensor0(fact.shape[self.axis].clone()).broadcast_into_rank(fact.rank())?,
            )?;
            let casted =
                model.wire_node(format!("{}.cast", prefix), cast(fact.datum_type), &[len])?;
            wire = model.wire_node(format!("{}.norm", prefix), div(), &[wire[0], casted[0]])?;
        }
        if self.onesided {
            let frame = fact.shape[self.axis].clone() / 2 + 1;
            wire = model.wire_node(
                format!("{}.onesided", prefix),
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

impl_dyn_hash!(Stft);

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
        dbg!(self);
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
        dbg!(&frame_len);
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
                format!("{}.add_imaginary", prefix),
                Pad { mode: tract_hir::ops::array::PadMode::Constant(rctensor0(0f32)), pads },
                &wire,
            )?;
        };
        wire = model.wire_node(
            format!("{}.pair_to_cplx", prefix),
            tract_core::ops::math::InnerDimToComplex,
            &wire,
        )?;
        wire = model.wire_node(
            format!("{}.fft", prefix),
            tract_core::ops::fft::Stft { axis: 1, frame, window, stride },
            &wire,
        )?;
        if self.onesided {
            wire = model.wire_node(
                format!("{}.onesided", prefix),
                tract_core::ops::array::Slice::new(2, 0, frame / 2 + 1),
                &wire,
            )?;
        }
        wire = model.wire_node(
            format!("{}.to_pair", prefix),
            tract_core::ops::math::ComplexToInnerDim,
            &wire,
        )?;
        Ok(wire)
    }
}
