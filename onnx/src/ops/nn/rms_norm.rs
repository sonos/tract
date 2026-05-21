use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::cast::cast;
use tract_core::ops::math::{add, mul, rsqrt};
use tract_core::ops::nn::{Reduce, Reducer};
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;

pub fn rms_normalization(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt::<isize>("axis")?.unwrap_or(-1);
    let epsilon = node.get_attr_opt("epsilon")?.unwrap_or(1e-5f32);
    let have_bias = node.input.len() >= 3 && !node.input[2].is_empty();
    Ok((expand(RmsNormalization { axis, epsilon, have_bias }), vec![]))
}

#[derive(Debug, Clone)]
struct RmsNormalization {
    axis: isize,
    epsilon: f32,
    have_bias: bool,
}

impl Expansion for RmsNormalization {
    fn name(&self) -> StaticName {
        "RmsNormalization".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2 + self.have_bias as usize)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let x_fact = model.outlet_fact(inputs[0])?.clone();
        let rank = x_fact.rank();
        let axis =
            if self.axis < 0 { (self.axis + rank as isize) as usize } else { self.axis as usize };
        let dt = x_fact.datum_type;
        let stash_dt = DatumType::F32;

        let axes: TVec<usize> = (axis..rank).collect();

        let x_cast = model.wire_node(format!("{prefix}.cast_x"), cast(stash_dt), &[inputs[0]])?[0];
        let mean_sq = model.wire_node(
            format!("{prefix}.mean_sq"),
            Reduce { axes, reducer: Reducer::MeanOfSquares },
            &[x_cast],
        )?[0];
        let eps = model.add_const(
            format!("{prefix}.eps"),
            tensor0(self.epsilon).cast_to_dt(stash_dt)?.into_owned(),
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
        let scaled = wire_with_rank_broadcast(
            format!("{prefix}.scaled"),
            model,
            mul(),
            &[normalized_cast, inputs[1]],
        )?[0];
        if self.have_bias {
            wire_with_rank_broadcast(prefix, model, add(), &[scaled, inputs[2]])
        } else {
            Ok(tvec![scaled])
        }
    }
}
