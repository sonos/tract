use crate::tensor::{DeviceTensor, DeviceTensorExt};
use derive_new::new;
use tract_core::broadcast::multi_broadcast;
use tract_core::internal::*;

static IFF_MAX_RANK: usize = 5;

/// Dispatch function for the iff (select) kernel.
/// Args: cond, then, else tensors with pre-computed broadcast strides,
/// output tensor, output shape and strides. All strides are padded to IFF_MAX_RANK.
pub type DispatchIffFn = fn(
    cond: &DeviceTensor,
    then_value: &DeviceTensor,
    else_value: &DeviceTensor,
    cond_strides: &[isize],
    then_strides: &[isize],
    else_strides: &[isize],
    output: &DeviceTensor,
    output_shape: &[usize],
    output_strides: &[isize],
) -> TractResult<()>;

#[derive(Clone, new)]
pub struct GpuIff {
    pub backend_name: &'static str,
    pub dispatch: DispatchIffFn,
}

impl std::fmt::Debug for GpuIff {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}Iff", self.backend_name)
    }
}

impl PartialEq for GpuIff {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
    }
}

impl Eq for GpuIff {}

impl std::hash::Hash for GpuIff {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
    }
}

impl Op for GpuIff {
    fn name(&self) -> StaticName {
        format!("{}Iff", self.backend_name).into()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuIff {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (cond_val, then_val, else_val) = args_3!(inputs);

        let cond = cond_val.to_device_tensor()?;
        let then_t = then_val.to_device_tensor()?;
        let else_t = else_val.to_device_tensor()?;
        ensure!(cond.rank() == then_t.rank());
        ensure!(cond.rank() == else_t.rank());
        ensure!(then_t.datum_type() == else_t.datum_type());

        let out_shape = multi_broadcast(&[cond.shape(), then_t.shape(), else_t.shape()])
            .context("No broadcasting solution found")?;
        let out_dt = then_t.datum_type();
        let output =
            crate::session_handler::make_tensor_for_node(session, node_id, out_dt, &out_shape)?;

        if output.len() > 0 {
            let rank = cond.rank();
            ensure!(rank <= IFF_MAX_RANK);
            let rank_pad = IFF_MAX_RANK - rank;

            let mut padded_cond_strides = [0isize; IFF_MAX_RANK];
            let mut padded_then_strides = [0isize; IFF_MAX_RANK];
            let mut padded_else_strides = [0isize; IFF_MAX_RANK];
            let mut padded_out_shape = [1usize; IFF_MAX_RANK];
            let mut padded_out_strides = [0isize; IFF_MAX_RANK];

            for axis in 0..rank {
                padded_out_shape[rank_pad + axis] = output.shape()[axis];
                padded_out_strides[rank_pad + axis] = output.strides()[axis];
                padded_cond_strides[rank_pad + axis] = if cond.shape()[axis] < output.shape()[axis]
                {
                    0
                } else {
                    cond.strides()[axis]
                };
                padded_then_strides[rank_pad + axis] =
                    if then_t.shape()[axis] < output.shape()[axis] {
                        0
                    } else {
                        then_t.strides()[axis]
                    };
                padded_else_strides[rank_pad + axis] =
                    if else_t.shape()[axis] < output.shape()[axis] {
                        0
                    } else {
                        else_t.strides()[axis]
                    };
            }

            (self.dispatch)(
                cond,
                then_t,
                else_t,
                &padded_cond_strides,
                &padded_then_strides,
                &padded_else_strides,
                &output,
                &padded_out_shape,
                &padded_out_strides,
            )
            .with_context(|| "Error while dispatching eval for Iff")?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuIff {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |inputs| {
            let out_shape =
                multi_broadcast(&[&*inputs[0].shape, &*inputs[1].shape, &*inputs[2].shape])
                    .context("No broadcasting solution found")?;
            let out_dt = inputs[1].datum_type;
            Ok(tvec!(out_dt.fact(out_shape)))
        })
    }

    as_op!();
}
