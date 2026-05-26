use crate::model::{ParsingContext, optional_outputs};
use crate::pb::NodeProto;
use tract_core::ops::cast::cast;
use tract_core::ops::math::{add, div, mul, rsqrt, square, sub};
use tract_core::ops::nn::{Reduce, Reducer};
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;

// com.microsoft SkipLayerNormalization:
//   input_skip_bias_sum = input + skip (+ bias)
//   output = LayerNorm(input_skip_bias_sum, last axis) * gamma (+ beta)
// Inputs:  input(0), skip(1), gamma(2), beta(3, opt), bias(4, opt)
// Outputs: output(0), mean(1, opt), inv_std_var(2, opt), input_skip_bias_sum(3, opt)
pub fn skip_layer_normalization(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let epsilon = node.get_attr_opt("epsilon")?.unwrap_or(1e-5f32);
    let have_beta = node.input.len() >= 4 && !node.input[3].is_empty();
    let have_bias = node.input.len() >= 5 && !node.input[4].is_empty();
    let mut oo = optional_outputs(node).skip(1);
    let mean = oo.next().unwrap().is_some();
    let invstd = oo.next().unwrap().is_some();
    let sum = oo.next().unwrap().is_some();
    Ok((expand(SkipLayerNorm { epsilon, have_beta, have_bias, mean, invstd, sum }), vec![]))
}

#[derive(Debug, Clone)]
struct SkipLayerNorm {
    epsilon: f32,
    have_beta: bool,
    have_bias: bool,
    mean: bool,
    invstd: bool,
    sum: bool,
}

impl Expansion for SkipLayerNorm {
    fn name(&self) -> StaticName {
        "SkipLayerNorm".into()
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(1 + self.mean as usize + self.invstd as usize + self.sum as usize)
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 3 + self.have_beta as usize + self.have_bias as usize)?;
        check_output_arity(outputs, self.nboutputs()?)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        if self.sum {
            let si = 1 + self.mean as usize + self.invstd as usize;
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
        let axes: TVec<usize> = tvec![rank - 1];

        // input_skip_bias_sum = input + skip (+ bias)
        let mut sum = wire_with_rank_broadcast(
            format!("{prefix}.skip"),
            model,
            add(),
            &[inputs[0], inputs[1]],
        )?[0];
        if self.have_bias {
            sum = wire_with_rank_broadcast(
                format!("{prefix}.bias"),
                model,
                add(),
                &[sum, inputs[4]],
            )?[0];
        }

        // LayerNorm over the last axis, computed in f32.
        let x = model.wire_node(format!("{prefix}.cast_x"), cast(stash), &[sum])?[0];
        // mean / var via Sum / count (the core Reducer has no Mean variant).
        let count: TDim = fact.shape[rank - 1].clone();
        let count = model.add_const(format!("{prefix}.count"), tensor0(count))?;
        let count = model.wire_node(format!("{prefix}.count_f32"), cast(stash), &[count])?[0];
        let sum_x = model.wire_node(
            format!("{prefix}.sum_x"),
            Reduce { axes: axes.clone(), reducer: Reducer::Sum },
            &[x],
        )?[0];
        let mean =
            wire_with_rank_broadcast(format!("{prefix}.mean"), model, div(), &[sum_x, count])?[0];
        let d = wire_with_rank_broadcast(format!("{prefix}.d"), model, sub(), &[x, mean])?[0];
        let dd = model.wire_node(format!("{prefix}.dd"), square(), &[d])?[0];
        let sum_dd = model.wire_node(
            format!("{prefix}.sum_dd"),
            Reduce { axes, reducer: Reducer::Sum },
            &[dd],
        )?[0];
        let var =
            wire_with_rank_broadcast(format!("{prefix}.var"), model, div(), &[sum_dd, count])?[0];
        let eps = model.add_const(
            format!("{prefix}.eps"),
            tensor0(self.epsilon).cast_to_dt(stash)?.into_owned(),
        )?;
        let var_eps =
            wire_with_rank_broadcast(format!("{prefix}.var_eps"), model, add(), &[var, eps])?[0];
        let inv_std = model.wire_node(format!("{prefix}.rsqrt"), rsqrt(), &[var_eps])?[0];
        let normalized =
            wire_with_rank_broadcast(format!("{prefix}.norm"), model, mul(), &[d, inv_std])?[0];
        let normalized = model.wire_node(format!("{prefix}.cast_out"), cast(dt), &[normalized])?[0];

        // scale by gamma (+ beta)
        let mut output = wire_with_rank_broadcast(
            format!("{prefix}.scaled"),
            model,
            mul(),
            &[normalized, inputs[2]],
        )?[0];
        if self.have_beta {
            output = wire_with_rank_broadcast(
                format!("{prefix}.beta"),
                model,
                add(),
                &[output, inputs[3]],
            )?[0];
        }

        let mut outputs = tvec!(output);
        if self.mean {
            outputs.push(mean);
        }
        if self.invstd {
            outputs.push(inv_std);
        }
        if self.sum {
            outputs.push(sum);
        }
        Ok(outputs)
    }
}
