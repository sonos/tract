use std::any::{Any, TypeId};

use crate::internal::*;

type RewriteRule<Ctx> =
    Box<dyn Fn(&Ctx, &TypedModel, &TypedNode) -> TractResult<Option<TypedModelPatch>>>;

#[derive(Default)]
pub struct Rewriter<Ctx> {
    rules: HashMap<TypeId, (Cow<'static, str>, RewriteRule<Ctx>)>,
}

impl<Ctx> Rewriter<Ctx> {
    pub fn with_rule_for<O: Any + 'static>(
        mut self,
        name: impl Into<Cow<'static, str>>,
        rule: RewriteRule<Ctx>,
    ) -> Self {
        self.rules.insert(TypeId::of::<O>(), (name.into(), rule));
        self
    }

    pub fn rewrite(&self, ctx: &Ctx, model: &mut TypedModel) -> TractResult<()> {
        loop {
            let mut done_anything = false;
            for n in model.eval_order()? {
                if let Some((name, rule)) = self.rules.get(&(&*model.node(n).op).type_id()) {
                    if let Some(patch) = (rule)(ctx, model, &model.node(n)).with_context(|| {
                        format!("Matching rule {name} on {}", model.node(n).name)
                    })? {
                        patch.apply(model).with_context(|| {
                            format!("Applying patch for rule {name} on {}", model.node(n).name)
                        })?;
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
