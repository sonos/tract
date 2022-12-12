use super::OptimizerSession;
use super::TypedPass;
use crate::internal::*;
use crate::model::*;
use std::collections::HashSet;
use std::fmt::Debug;

use crate::ops::change_axes::*;

#[derive(Clone, Default)]
pub struct ChangeAxes(HashSet<crate::ops::change_axes::AxisChange>);

impl Debug for ChangeAxes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ChangeAxes")
    }
}

impl TypedPass for ChangeAxes {
    fn reset(&mut self) -> TractResult<()> {
        self.0.clear();
        Ok(())
    }
    fn next(
        &mut self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut interfaces = model.output_outlets()?.to_vec();
        interfaces.extend(model.input_outlets()?.iter());
        for n in model.eval_order()? {
            for suggestion in model.node(n).op.suggested_axis_changes()? {
                let outlet = suggestion.0.as_outlet(model.node(n));
                let change = AxisChange { outlet, op: suggestion.1 };
                if self.0.insert(change.clone()) {
                    if let Some((patch, _)) = change_axes(model, &change, &interfaces, &[])
                        .with_context(|| {
                            format!("Making patch for {:?} from {}", change, model.node(n))
                        })?
                    {
                        return Ok(Some(patch));
                    }
                }
            }
            /*
            for (slot, fact) in model.node(n).outputs.iter().enumerate() {
                for (ix, dim) in fact.fact.shape.iter().enumerate() {
                    if dim.is_one() {
                        let change =
                            AxisChange { outlet: OutletId::new(n, slot), op: AxisOp::Rm(ix) };
                        if self.0.insert(change.clone()) {
                            if let Some((patch, _)) = change_axes(model, &change, &interfaces, &[])
                                .with_context(|| {
                                    format!("Making patch for {:?} from {}", change, model.node(n))
                                })?
                            {
                                return Ok(Some(patch));
                            }
                        }
                    }
                }
            }
            */
        }
        Ok(None)
    }
}
