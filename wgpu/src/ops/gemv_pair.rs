use tract_core::internal::*;
use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt};

use crate::kernels::gemv_pair::{GemvPairShape, wgpu_gemv_pair_dispatch};
use crate::kernels::matmul::{mkn, output_shape};
use crate::kernels::shaders::ChainStep;

/// Two single-row matrix products with an activation between them, as one
/// kernel: the shape a squeeze-excitation gate takes once its pooling is done.
/// Inputs past the three operands belong to the activation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuGemvPair {
    pub first: PrefixMatMul,
    pub second: PrefixMatMul,
    /// How the hidden vector is shaped where the second product reads it.
    pub hidden: TVec<usize>,
    pub act: Vec<ChainStep>,
}

/// The stride along each matmul axis, as the transpose flags name them.
fn axes_strides(t: &DeviceTensor, transposed: bool) -> (usize, usize) {
    let rank = t.rank();
    let (row, col) = (t.strides()[rank - 2], t.strides()[rank - 1]);
    if transposed { (col as usize, row as usize) } else { (row as usize, col as usize) }
}

impl Op for WgpuGemvPair {
    fn name(&self) -> StaticName {
        "WgpuGemvPair".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("act: {:?}", self.act)])
    }

    op_as_typed_op!();
}

impl EvalOp for WgpuGemvPair {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let inputs =
            inputs.iter().map(|it| it.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;
        let (x, w1, w2) = (inputs[0], inputs[1], inputs[2]);
        let shape = output_shape(
            &self.hidden,
            w2.shape(),
            self.second.transpose_a,
            self.second.transpose_b,
            self.second.transpose_c,
        )?;
        let output = tract_gpu::turn_handler::make_tensor_for_node(ctx, x.datum_type(), &shape)?;
        if output.len() > 0 {
            let (_, k, r) =
                mkn(x.shape(), w1.shape(), self.first.transpose_a, self.first.transpose_b)?;
            let (_, _, n) =
                mkn(&self.hidden, w2.shape(), self.second.transpose_a, self.second.transpose_b)?;
            wgpu_gemv_pair_dispatch(
                GemvPairShape {
                    k,
                    r,
                    n,
                    x_s: axes_strides(x, self.first.transpose_a).1,
                    w1: axes_strides(w1, self.first.transpose_b),
                    w2: axes_strides(w2, self.second.transpose_b),
                    out_s: axes_strides(&output, self.second.transpose_c).1,
                },
                &self.act,
                x,
                w1,
                w2,
                &inputs[3..],
                &output,
            )?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for WgpuGemvPair {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let hidden = TypedFact::dt_shape(facts[0].datum_type, &*self.hidden);
            self.second.output_facts(&[&hidden, facts[2]])
        })
        .with_context(|| "Error while computing facts for WgpuGemvPair")
    }
}
