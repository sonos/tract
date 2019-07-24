use super::codegen::Codegen;
use crate::internal::*;

#[derive(Debug, Clone, new, Default)]
pub struct Typed {
    pub body: TypedModel,
    decluttered: bool,
    pub(super) closure_inputs: usize,
    pub(super) scan_input_axes: Vec<usize>,
    pub(super) scan_output_axes: Vec<usize>,
    pub(super) scan_output_len_hint: Vec<Option<TDim>>,
}

impl Typed {
    pub fn to_codegen_op(&self) -> TractResult<Codegen> {
        let plan = SimplePlan::new(self.body.clone().into_optimized()?)?;
        Ok(Codegen::new(
            Arc::new(plan),
            self.closure_inputs,
            self.scan_input_axes.clone(),
            self.scan_output_axes.clone(),
            self.scan_output_len_hint.clone(),
        ))
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
