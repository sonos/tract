use super::TypedPass;
use crate::model::*;
use crate::TractResult;
use crate::internal::*;

use crate::ops::change_axes::*;

#[derive(Debug)]
pub struct ChangeAxes;

impl TypedPass for ChangeAxes {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool> {
        let mut suggestions = vec![];
        for n in model.eval_order()? {
            let node = model.node(n);
            for suggestion in node.op.suggested_axis_changes()? {
                let outlet = suggestion.0.as_outlet(&node);
                suggestions.push(AxisChange { outlet, op: suggestion.1 })
            }
        }
        let mut interfaces = model.output_outlets()?.to_vec();
        interfaces.extend(model.input_outlets()?.iter());
        for suggestion in suggestions.into_iter() {
            if change_axes(model, &suggestion, &interfaces, &[])
                .chain_err(|| format!("Applying {:?}", suggestion))?
                .is_some()
            {
                return Ok(true);
            }
        }
        Ok(false)
    }
}
