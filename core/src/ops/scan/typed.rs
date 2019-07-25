use super::codegen::Codegen;

use super::*;

#[derive(Debug, Clone, Default)]
pub struct Typed {
    pub body: TypedModel,
    decluttered: bool,
    pub input_mapping: Vec<InputMapping<TDim>>,
    pub output_mapping: Vec<OutputMapping<TDim>>,
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

        let output_mapping = self
            .output_mapping
            .iter()
            .map(|im| {
                Ok(match im {
                    OutputMapping::Scan { axis, slot, chunk } => OutputMapping::Scan {
                        axis: *axis,
                        slot: *slot,
                        chunk: chunk.to_integer()? as usize,
                    },
                    OutputMapping::State { slot } => OutputMapping::State { slot: *slot },
                })
            })
            .collect::<TractResult<_>>()?;

        Ok(Codegen::new(
            Arc::new(plan),
            input_mapping,
            output_mapping,
            self.scan_output_len_hint.clone(),
        ))
    }

    pub fn new(
        body: TypedModel,
        input_mapping: Vec<InputMapping<TDim>>,
        output_mapping: Vec<OutputMapping<TDim>>,
        scan_output_len_hint: Vec<Option<TDim>>,
    ) -> Typed {
        Typed { body, decluttered: false, input_mapping, output_mapping, scan_output_len_hint }
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
