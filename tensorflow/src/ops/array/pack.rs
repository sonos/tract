use tract_hir::internal::*;
use tract_hir::ops::array::TypedConcat;
use tract_hir::ops::binary::wire_cast;

use crate::model::ParsingContext;
use crate::tfpb::tensorflow::NodeDef;

pub fn pack(_ctx: &ParsingContext, pb: &NodeDef) -> TractResult<Box<dyn InferenceOp>> {
    let n = pb.input.len();
    let axis = pb.get_attr_int("axis")?;

    Ok(expand(Pack::new(n, axis)))
}

#[derive(Debug, Clone, new, Hash)]
pub struct Pack {
    n: usize, // The number of inputs
    axis: usize,
}



impl Expansion for Pack {
    fn name(&self) -> Cow<str> {
        "Pack".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        let axis = self.axis;
        check_input_arity(inputs, self.n)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].rank, inputs[0].rank.bex() + 1)?;
        s.equals_all((0..self.n).map(|i| inputs[i].rank.bex()).collect())?;
        s.given_all((0..self.n).map(move |i| &inputs[i].datum_type), move |s, dts| {
            if let Some(dt) = DatumType::super_type_for(dts) {
                s.equals(&outputs[0].datum_type, dt)?;
            }
            Ok(())
        })?;
        s.given(&inputs[0].rank, move |s, r| {
            for d in 0..r as usize {
                s.equals_all((0..self.n).map(|i| inputs[i].shape[d].bex()).collect())?;
            }
            Ok(())
        })?;
        s.given(&inputs[0].rank, move |s, r| {
            for d in 0..axis {
                s.equals(&outputs[0].shape[d], &inputs[0].shape[d])?;
            }
            if r > 0 {
                for d in axis..r as usize {
                    s.equals(&outputs[0].shape[d + 1], &inputs[0].shape[d])?
                }
            }
            Ok(())
        })?;
        s.equals(&outputs[0].shape[axis], self.n.to_dim())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let dt = inputs
            .iter()
            .map(|&i| Ok(model.outlet_fact(i)?.datum_type))
            .collect::<TractResult<TVec<DatumType>>>()?;
        let dt = DatumType::super_type_for(dt.iter()).context("No supertype")?;
        let wires = wire_cast(prefix, model, inputs, dt)?;
        let inputs: TVec<OutletId> = wires
            .iter()
            .enumerate()
            .map(|(ix, &o)| {
                Ok(model.wire_node(
                    format!("{prefix}.add_dims-{ix}"),
                    AxisOp::Add(self.axis),
                    &[o],
                )?[0])
            })
            .collect::<TractResult<TVec<OutletId>>>()?;
        model.wire_node(prefix, TypedConcat::new(self.axis), &inputs)
    }
}
