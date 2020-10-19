use tract_hir::internal::*;
use tract_hir::ops::nn;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

#[derive(Debug, Clone, new, Hash)]
pub struct Reduce {
    t: DatumType,
    t_idx: DatumType,
    keep_dims: bool,
    reducer: nn::Reducer,
}

tract_data::impl_dyn_hash!(Reduce);

pub fn max(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    reduce(pb, nn::Reducer::Max)
}

pub fn mean(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    reduce(pb, nn::Reducer::Mean)
}

pub fn min(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    reduce(pb, nn::Reducer::Min)
}

pub fn prod(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    reduce(pb, nn::Reducer::Prod)
}

pub fn sum(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    reduce(pb, nn::Reducer::Sum)
}

pub fn reduce(pb: &NodeDef, op: nn::Reducer) -> TractResult<Box<dyn InferenceOp>> {
    let t = pb.get_attr_datum_type("T")?;
    let t_idx = pb.get_attr_datum_type("Tidx")?;
    let keep_dims = pb.get_attr_bool("keep_dims")?;
    Ok(Box::new(Reduce::new(t, t_idx, keep_dims, op)))
}

impl Op for Reduce {
    fn name(&self) -> Cow<str> {
        format!("{:?}", self.reducer).into()
    }

    op_tf!();
    not_a_typed_op!();
}

impl EvalOp for Reduce {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, axes) = args_2!(inputs);
        let axes: Vec<i64> = axes.cast_to::<i64>()?.as_slice::<i64>()?.to_vec();
        let op = nn::Reduce::new(Some(axes), self.keep_dims, self.reducer);
        expand(op).eval(tvec!(input))
    }
}

impl InferenceRulesOp for Reduce {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        if self.keep_dims {
            s.equals(&inputs[0].rank, &outputs[0].rank)?;
        } else {
            s.given(&inputs[1].rank, move |s, rank| {
                if rank == 1 {
                    s.equals(
                        inputs[0].rank.bex().to_dim(),
                        inputs[1].shape[0].bex() + outputs[0].rank.bex().to_dim(),
                    )
                } else {
                    s.equals(
                        inputs[0].rank.bex().to_dim(),
                        outputs[0].rank.bex().to_dim() + 1.to_dim(),
                    )
                }
            })?;
        }
        s.given_3(
            &inputs[0].rank,
            &outputs[0].rank,
            &inputs[1].value,
            move |s, irank, orank, axes| {
                let axes: TVec<usize> = axes
                    .cast_to::<i64>()?
                    .as_slice::<i64>()?
                    .iter()
                    .map(|&ax| if ax > 0 { ax } else { ax + irank } as usize)
                    .collect();
                let mut od = 0;
                for id in 0..(irank as usize) {
                    if axes.contains(&id) {
                        if self.keep_dims {
                            s.equals(&outputs[0].shape[od], 1.to_dim())?;
                            od += 1;
                        }
                    } else {
                        if od < orank as usize {
                            s.equals(&outputs[0].shape[od], &inputs[0].shape[id])?;
                            od += 1;
                        }
                    }
                }
                Ok(())
            },
        )?;
        Ok(())
    }

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        if let Some(ref axes) = target.outlet_fact(mapping[&node.inputs[1]])?.konst {
            let axes: Vec<i64> = axes.cast_to::<i64>()?.as_slice::<i64>()?.to_vec();
            let op = nn::Reduce::new(Some(axes), self.keep_dims, self.reducer);
            op.wire(&node.name, target, &[mapping[&node.inputs[0]]])
        } else {
            bail!("Nees axes to be const")
        }
    }

    as_op!();
}
