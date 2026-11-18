use crate::tensor::{DeviceTensorExt, IntoDevice};
use tract_core::internal::*;
use tract_core::ops::array::{Pad, PadMode};

/// Constant and edge padding, as `copy_nd`s with no dedicated kernel.
///
/// Constant mode broadcasts the pad value across the whole output, then drops the input into
/// the interior. Edge mode splits the output into, per padded axis, a before strip, the
/// interior and an after strip, and fills each region from the input with a zero stride on the
/// axes where it lies in a strip, anchored on that axis' first or last element. Reflect mode
/// is left on the host (see [`GpuPad::from_core`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuPad {
    pub pads: Vec<(usize, usize)>,
    pub mode: PadMode,
}

impl GpuPad {
    /// Build from a core `Pad`, or `None` in reflect mode.
    pub fn from_core(op: &Pad) -> Option<Self> {
        match op.mode {
            PadMode::Constant(_) | PadMode::Edge => {
                Some(Self { pads: op.pads.clone(), mode: op.mode.clone() })
            }
            PadMode::Reflect => None,
        }
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
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input_value = args_1!(inputs);
        let input = input_value.to_device_tensor()?;
        let dt = input.datum_type();
        let out_shape = self.output_shape(input.shape());

        let output = crate::turn_handler::make_tensor_for_node(ctx, dt, &out_shape)?;
        let ctx = crate::device::get_context()?;

        if let PadMode::Constant(value) = &self.mode {
            let value = value.cast_to_dt(dt)?.into_owned().into_device()?;
            let zero_strides = vec![0isize; out_shape.len()];
            ctx.copy_nd(&value, 0, &zero_strides, &output, 0, &out_shape, output.strides())?;
        }
        if input.len() == 0 {
            return Ok(tvec![output.into_tensor().into_tvalue()]);
        }
        let padded: Vec<usize> = if self.mode == PadMode::Edge {
            (0..self.pads.len()).filter(|&ax| self.pads[ax] != (0, 0)).collect()
        } else {
            vec![]
        };
        let regions = 3usize.pow(padded.len() as u32);
        for region in 0..regions {
            let mut shape: TVec<usize> = input.shape().into();
            let mut src_strides: TVec<isize> = input.strides().into();
            let mut src_offset = 0isize;
            let mut dst_offset = 0isize;
            let mut code = region;
            for (axis, (before, after)) in self.pads.iter().enumerate() {
                let dim = input.shape()[axis];
                let side = if padded.contains(&axis) {
                    let side = code % 3;
                    code /= 3;
                    side
                } else {
                    1
                };
                match side {
                    0 => {
                        shape[axis] = *before;
                        src_strides[axis] = 0;
                    }
                    1 => dst_offset += *before as isize * output.strides()[axis],
                    _ => {
                        shape[axis] = *after;
                        src_offset += (dim - 1) as isize * input.strides()[axis];
                        src_strides[axis] = 0;
                        dst_offset += (before + dim) as isize * output.strides()[axis];
                    }
                }
            }
            if shape.contains(&0) {
                continue;
            }
            ctx.copy_nd(
                input,
                src_offset as usize * dt.size_of(),
                &src_strides,
                &output,
                dst_offset as usize * dt.size_of(),
                &shape,
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
