use crate::internal::*;
use std::fmt::Debug;

pub mod change_axes;
mod prop_const;
mod push_split_down;

use self::change_axes::ChangeAxes;
use self::prop_const::PropConst;
use self::push_split_down::PushSplitDown;

pub trait TypedPass: Debug + Send + Sync {
    fn reset(&mut self) -> TractResult<()>;
    fn next(&mut self, model: &TypedModel) -> TractResult<Option<TypedModelPatch>>;
}

pub fn declutter() -> Vec<Box<dyn TypedPass>> {
    vec![
        Box::new(OpOptim("declutter", TypedOp::declutter)),
        Box::new(PropConst),
        Box::new(PushSplitDown),
        Box::new(ChangeAxes),
    ]
}

pub fn codegen() -> Vec<Box<dyn TypedPass>> {
    vec![
        Box::new(OpOptim("codegen", TypedOp::codegen)),
        Box::new(PushSplitDown),
        Box::new(OpOptim("fuse", TypedOp::fuse)),
    ]
}

pub struct OpOptim(
    &'static str,
    fn(
        op: &dyn TypedOp,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>>,
);

impl OpOptim {
    fn full_pass(&mut self, new: &TypedModel) -> TractResult<Option<TypedModelPatch>> {
        for id in new.eval_order()? {
            let node = &new.nodes()[id];
            let patch = (self.1)(node.op.as_ref(), &new, node)
                .with_context(|| format!("{:?} node {}", self, node))?;
            if let Some(mut p) = patch {
                p.push_context(format!("{:?} {}", self, node));
                return Ok(Some(p));
            }
        }
        Ok(None)
    }
}

impl std::fmt::Debug for OpOptim {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "{}", self.0)
    }
}

impl TypedPass for OpOptim {
    fn reset(&mut self) -> TractResult<()> {
        Ok(())
    }

    fn next(&mut self, model: &TypedModel) -> TractResult<Option<TypedModelPatch>> {
        self.full_pass(model)
    }
}
