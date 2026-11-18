use crate::tensor::DeviceTensorExt;
use tract_core::internal::*;
use tract_core::ops::array::Tile;

/// Tiling as one `copy_nd` with no dedicated kernel: each repeated axis gets an outer axis of
/// its own, read with a zero stride, so the output reads as `[m0, d0, m1, d1, ...]`. Extent-one
/// axes are left out, as the copy takes at most six.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuTile {
    pub multipliers: TVec<usize>,
}

impl GpuTile {
    /// Build from a core `Tile` whose multipliers are known, or `None`.
    pub fn from_core(op: &Tile) -> Option<Self> {
        let multipliers: TVec<usize> =
            op.multipliers.iter().map(|m| m.to_usize().ok()).collect::<Option<_>>()?;
        let repeated = multipliers.iter().filter(|m| **m > 1).count();
        (multipliers.len() + repeated <= 6).then_some(Self { multipliers })
    }
}

impl Op for GpuTile {
    fn name(&self) -> StaticName {
        "GpuTile".into()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuTile {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input_value = args_1!(inputs);
        let input = input_value.to_device_tensor()?;
        let dt = input.datum_type();
        let out_shape: TVec<usize> =
            input.shape().iter().zip(&self.multipliers).map(|(d, m)| d * m).collect();
        let output = crate::turn_handler::make_tensor_for_node(ctx, dt, &out_shape)?;
        if output.len() == 0 {
            return Ok(tvec![output.into_tensor().into_tvalue()]);
        }
        let mut shape = tvec![];
        let mut in_strides = tvec![];
        let mut out_strides = tvec![];
        for (axis, &m) in self.multipliers.iter().enumerate() {
            let dim = input.shape()[axis];
            let out_stride = output.strides()[axis];
            if m > 1 {
                shape.push(m);
                in_strides.push(0);
                out_strides.push(dim as isize * out_stride);
            }
            if dim > 1 {
                shape.push(dim);
                in_strides.push(input.strides()[axis]);
                out_strides.push(out_stride);
            }
        }
        if shape.is_empty() {
            shape.push(1);
            in_strides.push(1);
            out_strides.push(1);
        }
        crate::device::get_context()?.copy_nd(
            input,
            0,
            &in_strides,
            &output,
            0,
            &shape,
            &out_strides,
        )?;
        Ok(tvec![output.into_tensor().into_tvalue()])
    }
}

impl TypedOp for GpuTile {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let shape: TVec<TDim> =
                facts[0].shape.iter().zip(&self.multipliers).map(|(d, m)| d.clone() * *m).collect();
            Ok(tvec!(facts[0].datum_type.fact(shape)))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
