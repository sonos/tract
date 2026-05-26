use crate::model::{ParsingContext, optional_outputs};
use crate::pb::NodeProto;
use tract_core::ops::cast::cast;
use tract_core::ops::math::{add, mul, rsqrt};
use tract_core::ops::nn::{Reduce, Reducer};
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;

// com.microsoft SkipSimplifiedLayerNormalization:
//   input_skip_bias_sum = input + skip   (ORT >= 1.19 does NOT apply the optional bias here)
//   output              = RMSNorm(input_skip_bias_sum, last axis) * gamma
// Outputs: output(0), mean(1, unsupported), inv_std_var(2, opt), input_skip_bias_sum(3, opt).
pub fn skip_simplified_layer_normalization(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let epsilon = node.get_attr_opt("epsilon")?.unwrap_or(1e-5f32);
    let mut oo = optional_outputs(node).skip(1);
    let mean_out = oo.next().unwrap();
    let invstd_out = oo.next().unwrap();
    let sum_out = oo.next().unwrap();
    ensure!(mean_out.is_none(), "SkipSimplifiedLayerNormalization: mean output is unsupported");
    Ok((
        expand(SkipSimplifiedLayerNorm {
            epsilon,
            invstd: invstd_out.is_some(),
            sum: sum_out.is_some(),
        }),
        vec![],
    ))
}

#[derive(Debug, Clone)]
struct SkipSimplifiedLayerNorm {
    epsilon: f32,
    invstd: bool,
    sum: bool,
}

impl Expansion for SkipSimplifiedLayerNorm {
    fn name(&self) -> StaticName {
        "SkipSimplifiedLayerNorm".into()
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1 + self.invstd as usize + self.sum as usize)
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        // (input, skip, gamma[, bias]); ORT does not apply the optional bias for the simplified
        // variant, so a 4th input is accepted but ignored.
        ensure!(
            inputs.len() == 3 || inputs.len() == 4,
            "SkipSimplifiedLayerNormalization expects 3 or 4 inputs, got {}",
            inputs.len()
        );
        check_output_arity(outputs, self.nboutputs()?)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        if self.sum {
            // input_skip_bias_sum has the same type/shape as the input; it is the last output.
            let si = 1 + self.invstd as usize;
            s.equals(&inputs[0].datum_type, &outputs[si].datum_type)?;
            s.equals(&inputs[0].shape, &outputs[si].shape)?;
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
        let rank = fact.rank();
        let dt = fact.datum_type;
        let stash = DatumType::F32;

        // input + skip. NB: ORT (>= 1.19) does not apply the optional bias input for the simplified
        // (RMSNorm) variant -- verified empirically -- so we match that and ignore it. RMSNorm is
        // biasless in practice (Llama/Phi).
        let sum = wire_with_rank_broadcast(
            format!("{prefix}.skip"),
            model,
            add(),
            &[inputs[0], inputs[1]],
        )?[0];

        // RMSNorm over the last axis (in f32), then scale by gamma.
        let x_cast = model.wire_node(format!("{prefix}.cast_x"), cast(stash), &[sum])?[0];
        let mean_sq = model.wire_node(
            format!("{prefix}.mean_sq"),
            Reduce { axes: tvec![rank - 1], reducer: Reducer::MeanOfSquares },
            &[x_cast],
        )?[0];
        let eps = model.add_const(
            format!("{prefix}.eps"),
            tensor0(self.epsilon).cast_to_dt(stash)?.into_owned(),
        )?;
        let mean_sq_eps =
            wire_with_rank_broadcast(format!("{prefix}.add_eps"), model, add(), &[mean_sq, eps])?
                [0];
        let inv_rms = model.wire_node(format!("{prefix}.rsqrt"), rsqrt(), &[mean_sq_eps])?[0];
        let normalized =
            wire_with_rank_broadcast(format!("{prefix}.norm"), model, mul(), &[x_cast, inv_rms])?
                [0];
        let normalized_cast =
            model.wire_node(format!("{prefix}.cast_out"), cast(dt), &[normalized])?[0];
        let output = wire_with_rank_broadcast(
            format!("{prefix}.scaled"),
            model,
            mul(),
            &[normalized_cast, inputs[2]],
        )?[0];

        let mut outputs = tvec!(output);
        if self.invstd {
            outputs.push(inv_rms);
        }
        if self.sum {
            outputs.push(sum);
        }
        Ok(outputs)
    }
}
