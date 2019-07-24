use super::codegen::Codegen;
use crate::internal::*;

use super::*;

#[derive(Debug, Clone, Default)]
pub struct Typed {
    pub body: TypedModel,
    decluttered: bool,
    pub hidden_state_len: usize,
    pub input_mapping: Vec<InputMapping<TDim>>,
    pub(super) scan_output_axes: Vec<usize>,
    pub(super) scan_output_len_hint: Vec<Option<TDim>>,
}

impl Typed {
    pub fn to_codegen_op(&self) -> TractResult<Codegen> {
        let plan = SimplePlan::new(self.body.clone().into_optimized()?)?;
        let input_mapping = self
            .input_mapping
            .iter()
            .map(|im| {
                Ok(match im {
                    InputMapping::Scan { axis, slot, chunk } => InputMapping::Scan {
                        axis: *axis,
                        slot: *slot,
                        chunk: chunk.to_integer()? as usize,
                    },
                    InputMapping::Full { slot } => InputMapping::Full { slot: *slot },
                    InputMapping::State { initializer } => {
                        InputMapping::State { initializer: initializer.clone() }
                    }
                })
            })
            .collect::<TractResult<_>>()?;
        Ok(Codegen::new(
            Arc::new(plan),
            self.hidden_state_len,
            input_mapping,
            self.scan_output_axes.clone(),
            self.scan_output_len_hint.clone(),
        ))
    }

    pub fn new(
        body: TypedModel,
        hidden_state_len: usize,
        input_mapping: Vec<InputMapping<TDim>>,
        scan_output_axes: Vec<usize>,
        scan_output_len_hint: Vec<Option<TDim>>,
    ) -> Typed {
        Typed {
            body,
            decluttered: false,
            hidden_state_len,
            input_mapping,
            scan_output_axes,
            scan_output_len_hint,
        }
    }
}

impl Op for Typed {
    fn name(&self) -> Cow<str> {
        "Scan::Typed".into()
    }

    fn nested_models(&self) -> Vec<(Cow<str>, &Model)> {
        vec![("loop".into(), &self.body)]
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if !self.decluttered {
            let mut new = self.clone();
            new.body = self.body.clone().declutter()?;
            new.decluttered = true;
            return Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, new)?));
        }
        Ok(None)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(Some(TypedModelPatch::replace_single_op(
            &model,
            node,
            &node.inputs,
            self.to_codegen_op()?,
        )?))
    }
}

impl StatelessOp for Typed {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        self.to_codegen_op()?.eval(inputs)
    }
}
