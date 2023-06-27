use std::any::{Any, TypeId};

use crate::internal::*;

#[derive(Default)]
pub struct Rewriter<Ctx> {
    rules: HashMap<
        TypeId,
        Box<dyn Fn(&Ctx, &TypedModel, &TypedNode) -> TractResult<Option<TypedModelPatch>>>,
    >,
}

impl<Ctx> Rewriter<Ctx> {
    pub fn with_rule_for<O: Any + 'static>(
        mut self,
        rule: Box<dyn Fn(&Ctx, &TypedModel, &TypedNode) -> TractResult<Option<TypedModelPatch>>>,
    ) -> Self {
        self.rules.insert(TypeId::of::<O>(), rule.into());
        self
    }

    pub fn rewrite(&self, ctx: &Ctx, model: &mut TypedModel) -> TractResult<()> {
        loop {
            let mut done_anything = false;
            for n in model.eval_order()? {
                if let Some(rule) = self.rules.get(&(&*model.node(n).op).type_id()) {
                    if let Some(patch) = (rule)(ctx, model, &model.node(n))? {
                        patch.apply(model)?;
                        done_anything = true;
                    }
                }
            }
            if done_anything {
                model.compact()?;
            } else {
                return Ok(());
            }
        }
    }
}
