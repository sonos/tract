use super::TypedPass;
use crate::internal::*;
use crate::model::*;
use crate::TractResult;

use crate::ops::change_axes::*;

#[derive(Clone, Debug)]
pub struct ChangeAxes;

impl TypedPass for ChangeAxes {
    fn reset(&mut self) -> TractResult<()> {
        Ok(())
    }
    fn next(&mut self, model: &TypedModel) -> TractResult<Option<TypedModelPatch>> {
        let mut interfaces = model.output_outlets()?.to_vec();
        interfaces.extend(model.input_outlets()?.iter());
        for n in model.eval_order()? {
            for suggestion in model.node(n).op.suggested_axis_changes()? {
                let outlet = suggestion.0.as_outlet(&model.node(n));
                let change = AxisChange { outlet, op: suggestion.1 };
                if let Some((patch, _)) = change_axes(model, &change, &interfaces, &[])
                    .with_context(|| {
                        format!("Making patch for {:?} from {}", change, model.node(n))
                    })?
                {
                    return Ok(Some(patch));
                }
            }
        }
        Ok(None)
    }
}
