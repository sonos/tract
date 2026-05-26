use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;
use tract_hir::ops::math::{add, div, sqrt, square, sub};
use tract_hir::ops::nn::{Reduce, Reducer};

// ONNX MeanVarianceNormalization is defined as a function:
//   mean   = ReduceMean(X, axes)
//   var    = ReduceMean(X^2, axes) - mean^2
//   Y      = (X - mean) / (Sqrt(var) + 1e-9)
// Note epsilon (1e-9) is added *outside* the square root, and is fixed (not an attribute).
const EPSILON: f32 = 1e-9;

pub fn mean_variance_normalization(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axes = node.get_attr_opt_vec("axes")?.unwrap_or_else(|| vec![0, 2, 3]);
    Ok((expand(MeanVarianceNorm { axes }), vec![]))
}

#[derive(Debug, Clone, new)]
struct MeanVarianceNorm {
    axes: Vec<i64>,
}

impl Expansion for MeanVarianceNorm {
    fn name(&self) -> StaticName {
        "MeanVarianceNorm".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
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
        let input_fact = model.outlet_fact(inputs[0])?.clone();
        let rank = input_fact.rank() as i64;
        let dt = input_fact.datum_type;
        let axes: Vec<i64> = self.axes.iter().map(|&a| if a < 0 { a + rank } else { a }).collect();

        let mean = Reduce::new(Some(axes.clone()), true, Reducer::Mean).wire(
            &format!("{prefix}.mean"),
            model,
            &inputs[0..1],
        )?;
        let x_sq = model.wire_node(format!("{prefix}.sq"), square(), &inputs[0..1])?;
        let ex_sq = Reduce::new(Some(axes), true, Reducer::Mean).wire(
            &format!("{prefix}.ex_sq"),
            model,
            &x_sq,
        )?;
        let mean_sq = model.wire_node(format!("{prefix}.mean_sq"), square(), &mean)?;
        let var = model.wire_node(format!("{prefix}.var"), sub(), &[ex_sq[0], mean_sq[0]])?;
        let std = model.wire_node(format!("{prefix}.std"), sqrt(), &var)?;
        let eps = model
            .add_const(format!("{prefix}.eps"), tensor0(EPSILON).cast_to_dt(dt)?.into_owned())?;
        let std_eps =
            wire_with_rank_broadcast(format!("{prefix}.std_eps"), model, add(), &[std[0], eps])?;
        let centered = wire_with_rank_broadcast(
            format!("{prefix}.centered"),
            model,
            sub(),
            &[inputs[0], mean[0]],
        )?;
        wire_with_rank_broadcast(prefix, model, div(), &[centered[0], std_eps[0]])
    }
}
