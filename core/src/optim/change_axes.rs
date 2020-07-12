use super::TypedPass;
use crate::internal::*;
use crate::model::*;
use crate::TractResult;

use crate::ops::change_axes::*;

#[derive(Debug)]
pub struct ChangeAxes;

impl TypedPass for ChangeAxes {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool> {
        let mut interfaces = model.output_outlets()?.to_vec();
        interfaces.extend(model.input_outlets()?.iter());
        let mut done_something = false;
        'top: loop {
            for n in model.eval_order()? {
                for suggestion in model.node(n).op.suggested_axis_changes()? {
                    let outlet = suggestion.0.as_outlet(&model.node(n));
                    let change = AxisChange { outlet, op: suggestion.1 };
                    if change_axes(
                        model,
                        &change,
                        &interfaces,
                        &[],
                    )
                    .chain_err(|| format!("Applying {:?}", change))?
                    .is_some()
                    {
                        done_something = true;
                        continue 'top;
                    }
                }
            }
            return Ok(done_something)
        }
    }
}
