use std::any::TypeId;

use crate::internal::*;

type GenRewriteRule<Ctx> =
    Box<dyn Fn(&Ctx, &TypedModel, &TypedNode) -> TractResult<Option<TypedModelPatch>>>;

#[derive(Default)]
#[allow(clippy::type_complexity)]
pub struct Rewriter<Ctx> {
    rules: HashMap<TypeId, Vec<(Cow<'static, str>, GenRewriteRule<Ctx>)>>,
}

impl<Ctx> Rewriter<Ctx> {
    pub fn with_rule_for<O: Op + 'static>(
        mut self,
        name: impl Into<Cow<'static, str>>,
        rule: impl Fn(&Ctx, &TypedModel, &TypedNode, &str, &O) -> TractResult<Option<TypedModelPatch>>
            + 'static,
    ) -> Self {
        self.rules.entry(TypeId::of::<O>()).or_default().push((
            name.into(),
            Box::new(move |c: &Ctx, m: &TypedModel, n: &TypedNode| {
                let o = n.op_as::<O>().unwrap();
                rule(c, m, n, &n.name, o)
            }),
        ));
        self
    }

    pub fn rewrite(&self, ctx: &Ctx, model: &mut TypedModel) -> TractResult<()> {
        loop {
            let mut done_anything = false;
            for n in model.eval_order()? {
                if let Some(rules) = self.rules.get(&(*model.node(n).op).type_id()) {
                    for (name, rule) in rules {
                        if let Some(patch) = (rule)(ctx, model, model.node(n))
                            .with_context(|| format!("Evaluating rewriting rule \"{name}\" on node {}", model.node(n)))?
                        {
                            patch.apply(model).with_context(|| {
                                format!("Applying patch for rewriting rule \"{name}\" on node {}", model.node(n))
                            })?;
                            done_anything = true;
                        }
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
