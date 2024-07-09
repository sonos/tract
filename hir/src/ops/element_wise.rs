use tract_core::ops::cast::wire_cast;

use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone)]
pub struct ElementWiseOp(pub Box<dyn ElementWiseMiniOp>);

impl Expansion for ElementWiseOp {
    fn name(&self) -> Cow<str> {
        self.0.name().into()
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let operating_datum_type =
            self.0.operating_datum_type(target.outlet_fact(inputs[0])?.datum_type);
        let wires = wire_cast(prefix, target, inputs, operating_datum_type)?;
        target.wire_node(
            prefix,
            tract_core::ops::element_wise::ElementWiseOp(self.0.clone(), None),
            &wires,
        )
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.given(&inputs[0].datum_type, move |s, dt| {
            let dt = self.0.operating_datum_type(dt);
            if let Some(dt) = self.0.output_type(dt) {
                s.equals(&outputs[0].datum_type, dt)
            } else {
                s.equals(&outputs[0].datum_type, dt)
            }
        })?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }
}

pub trait ElementWiseIntoHir {
    fn into_hir(self) -> Box<dyn InferenceOp>;
}

impl ElementWiseIntoHir for tract_core::ops::element_wise::ElementWiseOp {
    fn into_hir(self) -> Box<dyn InferenceOp> {
        expand(ElementWiseOp(self.0))
    }
}
