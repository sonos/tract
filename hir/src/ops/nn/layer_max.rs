use tract_core::ops::nn::Softmax;

use crate::infer::*;
use crate::internal::*;

// TODO tricky to re-express in "core" because of the multiple hot point... do
// we need one more reduce ?
#[derive(Debug, Clone, new, Default, Hash)]
pub struct LayerHardmax {
    axis: isize,
    coerce_to_2d: bool,
}

impl Expansion for LayerHardmax {
    fn name(&self) -> Cow<str> {
        "LayerHardmax".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    fn wire(
        &self,
        name: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        use tract_core::ops::{array, change_axes, nn};
        let input = inputs[0];
        let input_fact = target.outlet_fact(input)?.clone();
        let input_dt = input_fact.datum_type;
        let rank = input_fact.rank();
        let axis = if self.axis < 0 { rank as isize + self.axis } else { self.axis } as usize;
        let suffix_dim: TDim = input_fact.shape[axis..].iter().product();
        let dim = if self.coerce_to_2d {
            suffix_dim.to_usize()
        } else {
            input_fact.shape[axis].to_usize()
        }
        .context("Assumes known dimension on working axes suffix.")?;
        let off = tensor0(0f32).cast_to_dt(input_dt)?.into_owned().into_arc_tensor();
        let on = tensor0(1f32).cast_to_dt(input_dt)?.into_owned().into_arc_tensor();
        let mut wires = inputs.into();
        if self.coerce_to_2d {
            wires = target.wire_node(
                format!("{name}.reshaped"),
                AxisOp::Reshape(axis, input_fact.shape[axis..].into(), tvec!(suffix_dim.clone())),
                &[input],
            )?;
        }
        wires = target.wire_node(
            format!("{name}.argmax"),
            nn::Reduce::new(tvec!(axis), nn::Reducer::ArgMax(false)),
            &wires,
        )?;
        wires =
            target.wire_node(format!("{name}.rm_axis"), change_axes::AxisOp::Rm(axis), &wires)?;
        wires = target.wire_node(
            format!("{name}.hardmax"),
            array::OneHot { axis, dim, off, on },
            &wires,
        )?;
        if self.coerce_to_2d {
            wires = target.wire_node(
                format!("{name}.hardmax_reshaped"),
                AxisOp::Reshape(axis, tvec!(suffix_dim), input_fact.shape[axis..].into()),
                &wires,
            )?;
        }
        Ok(wires)
    }
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct LayerLogSoftmax {
    pub axis: isize,
    pub coerce_to_2d: bool,
}

impl Expansion for LayerLogSoftmax {
    fn name(&self) -> Cow<str> {
        "LayerLogSoftmax".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    fn wire(
        &self,
        name: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let softmax = LayerSoftmax { axis: self.axis, coerce_to_2d: self.coerce_to_2d }
            .wire(name, target, inputs)?;
        target.wire_node(format!("{name}.logsoftmax"), tract_core::ops::math::ln(), &softmax)
    }
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct LayerSoftmax {
    axis: isize,
    coerce_to_2d: bool,
}

impl Expansion for LayerSoftmax {
    fn name(&self) -> Cow<str> {
        "LayerSoftmax".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    fn wire(
        &self,
        name: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input = inputs[0];
        let rank = target.outlet_fact(input)?.rank();
        let dt = target.outlet_fact(input)?.datum_type;
        let axis = if self.axis < 0 { rank as isize + self.axis } else { self.axis } as usize;
        let axes =
            if self.coerce_to_2d { (axis..rank).collect::<TVec<usize>>() } else { tvec!(axis) };
        let quant_output_dt = if dt.is_float() { None } else { Some(dt) };
        target.wire_node(name, Softmax { axes, quant_output_dt, ..Softmax::default() }, inputs)
    }
}

fn rules<'r, 'p: 'r, 's: 'r>(
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    check_output_arity(outputs, 1)?;
    s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
    s.equals(&outputs[0].rank, &inputs[0].rank)?;
    s.equals(&outputs[0].shape, &inputs[0].shape)?;
    Ok(())
}
