use crate::tensor::{DeviceTensorExt, IntoDevice};
use tract_core::internal::*;
use tract_core::ops::array::{Pad, PadMode};

/// Constant padding via two `copy_nd`s: broadcast the pad value across the whole
/// output, then drop the input into the interior. No dedicated kernel. Reflect/
/// Edge modes are left on the host (see [`GpuPad::from_core`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuPad {
    pub pads: Vec<(usize, usize)>,
    pub value: Arc<Tensor>,
}

impl GpuPad {
    /// Build from a core `Pad`, or `None` when the mode isn't `Constant`.
    pub fn from_core(op: &Pad) -> Option<Self> {
        let PadMode::Constant(value) = &op.mode else { return None };
        Some(Self { pads: op.pads.clone(), value: value.clone() })
    }

    fn output_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        input.iter().zip(&self.pads).map(|(d, (a, b))| d.clone() + *a + *b).collect()
    }
}

impl Op for GpuPad {
    fn name(&self) -> StaticName {
        "GpuPad".into()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuPad {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input_value = args_1!(inputs);
        let input = input_value.to_device_tensor()?;
        let dt = input.datum_type();
        let out_shape = self.output_shape(input.shape());

        let output =
            crate::session_handler::make_tensor_for_node(session, node_id, dt, &out_shape)?;

        let ctx = crate::device::get_context()?;

        // Fill the whole output with the pad value, broadcast from a scalar.
        let value = self.value.cast_to_dt(dt)?.into_owned().into_device()?;
        let zero_strides = vec![0isize; out_shape.len()];
        ctx.copy_nd(&value, 0, &zero_strides, &output, 0, &out_shape, output.strides())?;

        // Place the input at the interior offset.
        if input.len() != 0 {
            let interior: usize = self
                .pads
                .iter()
                .enumerate()
                .map(|(axis, (before, _))| before * output.strides()[axis] as usize)
                .sum();
            ctx.copy_nd(
                input,
                0,
                input.strides(),
                &output,
                interior * dt.size_of(),
                input.shape(),
                output.strides(),
            )?;
        }
        Ok(tvec![output.into_tensor().into_tvalue()])
    }
}

impl TypedOp for GpuPad {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            Ok(tvec!(facts[0].datum_type.fact(self.output_shape(&facts[0].shape.to_tvec()))))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
