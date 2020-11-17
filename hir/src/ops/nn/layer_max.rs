use crate::infer::*;
use crate::internal::*;

// TODO tricky to re-express in "core" because of the multiple hot point... do
// we need one more reduce ?
#[derive(Debug, Clone, new, Default, Hash)]
pub struct LayerHardmax {
    axis: isize,
}

impl_dyn_hash!(LayerHardmax);

impl Expansion for LayerHardmax {
    fn name(&self) -> Cow<str> {
        "LayerHardmax".into()
    }
    op_hir!();

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
        let suffix_dim: TDim = input_fact.shape[axis..].iter().maybe_product()?;
        let dim = suffix_dim
            .to_usize()
            .context("OneHot assumes known dimension on working axes suffix.")?;
        let off = tensor0(0f32).cast_to_dt(input_dt)?.into_owned().into_arc_tensor();
        let on = tensor0(1f32).cast_to_dt(input_dt)?.into_owned().into_arc_tensor();
        let mut wires = target.wire_node(
            format!("{}.reshaped", name),
            AxisOp::Reshape(axis, input_fact.shape[axis..].into(), tvec!(suffix_dim.clone())),
            &[input],
        )?;
        wires = target.wire_node(
            format!("{}.argmax", name),
            nn::Reduce::new(tvec!(axis), nn::Reducer::ArgMax(false)),
            &wires,
        )?;
        wires =
            target.wire_node(format!("{}.rm_axis", name), change_axes::AxisOp::Rm(axis), &wires)?;
        wires = target.wire_node(
            format!("{}.hardmax", name),
            array::OneHot { axis, dim, off, on },
            &wires,
        )?;
        target.wire_node(
            format!("{}.hardmax_reshaped", name),
            AxisOp::Reshape(axis, tvec!(suffix_dim), input_fact.shape[axis..].into()),
            &wires,
        )
    }
}


#[derive(Debug, Clone, new, Default, Hash)]
pub struct LayerLogSoftmax {
    axis: isize,
}

impl_dyn_hash!(LayerLogSoftmax);

impl Expansion for LayerLogSoftmax {
    fn name(&self) -> Cow<str> {
        "LayerLogSoftmax".into()
    }

    op_hir!();

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
        let softmax = LayerSoftmax { axis: self.axis }.wire(name, target, inputs)?;
        target.wire_node(format!("{}.logsoftmax", name), tract_core::ops::math::ln(), &softmax)
    }
}

#[derive(Debug, Clone, new, Default, Hash)]
pub struct LayerSoftmax {
    axis: isize,
}

impl_dyn_hash!(LayerSoftmax);

impl Expansion for LayerSoftmax {
    fn name(&self) -> Cow<str> {
        "LayerSoftmax".into()
    }
    op_hir!();

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
        use tract_core::ops::{math, nn};
        let input = inputs[0];
        let rank = target.outlet_fact(input)?.rank();
        let axis = if self.axis < 0 { rank as isize + self.axis } else { self.axis } as usize;
        let reducing_axes = (axis..rank).collect::<TVec<usize>>();
        let maxes = target.wire_node(
            format!("{}.max", name),
            nn::Reduce::new(reducing_axes.clone(), nn::Reducer::Max),
            &[input],
        )?[0];
        let normed = target.wire_node(
            format!("{}.normed", name),
            math::sub::bin_typed(),
            &[input, maxes],
        )?[0];
        let exp =
            target.wire_node(format!("{}.exp", name), tract_core::ops::math::exp(), &[normed])?[0];
        let sum = target.wire_node(
            format!("{}.sum", name),
            nn::Reduce::new(reducing_axes, nn::Reducer::Sum),
            &[exp],
        )?[0];
        target.wire_node(
            format!("{}.softmax", name),
            tract_core::ops::math::div::bin_typed(),
            &[exp, sum],
        )
    }
}

fn rules<'r, 'p: 'r, 's: 'r>(
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    check_output_arity(&outputs, 1)?;
    s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
    s.equals(&outputs[0].rank, &inputs[0].rank)?;
    s.equals(&outputs[0].shape, &inputs[0].shape)?;
    Ok(())
}
