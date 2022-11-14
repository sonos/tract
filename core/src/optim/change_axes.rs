use super::OptimizerSession;
use super::TypedPass;
use crate::internal::*;
use crate::model::*;
use std::collections::HashSet;
use std::fmt::Debug;

use crate::ops::change_axes::*;

#[derive(Clone, Default)]
pub struct ChangeAxes(HashSet<(usize, (InOut, AxisOp))>);

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
                if self.0.insert((n, suggestion.clone())) {
                    let outlet = suggestion.0.as_outlet(model.node(n));
                    let change = AxisChange { outlet, op: suggestion.1.clone() };
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
        Ok(None)
    }
}
