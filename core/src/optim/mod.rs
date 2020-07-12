use crate::model::*;
use crate::TractResult;
use std::fmt::Debug;

pub mod change_axes;
mod prop_const;
mod push_split_down;

use self::change_axes::ChangeAxes;
use self::prop_const::PropConst;
use self::push_split_down::PushSplitDown;

use crate::errors::TractResultExt;

pub trait TypedPass: Debug + Send + Sync {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool>;
}

pub fn declutter() -> Vec<Box<dyn TypedPass>> {
    vec![
        Box::new(PropConst),
        Box::new(OpOptim("declutter", TypedOp::declutter)),
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
    fn full_pass(&self, new: &mut TypedModel) -> TractResult<bool> {
        let mut done_something = false;
        for id in new.eval_order()? {
            let reduced = {
                let node = &new.nodes()[id];
                (self.1)(node.op.as_ref(), &new, node)
                    .chain_err(|| format!("{:?} node {}", self, node))?
            };
            if let Some(red) = reduced {
                debug!("Apply a model patch for {:?} {}", self, new.nodes()[id]);
                red.apply(new)?;
                if cfg!(debug_assertions) {
                    new.check_edges()?;
                }
                done_something = true;
            }
        }
        Ok(done_something)
    }
}

impl std::fmt::Debug for OpOptim {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.fmt(fmt)
    }
}

impl TypedPass for OpOptim {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool> {
        let mut new = model.clone();
        for i in 0..10 {
            if !self.full_pass(&mut new)? {
                std::mem::swap(model, &mut new);
                return Ok(i>0);
            }
        }

        let initial = model.signature();
        let mut hashset = std::collections::HashSet::new();
        hashset.insert(initial);

        for _ in 0.. {
            if !self.full_pass(&mut new)? {
                break;
            }

            new = new.compact()?;
            let sig = new.signature();
            if hashset.contains(&sig) {
                break;
            } else {
                hashset.insert(sig);
            }
        }
        std::mem::swap(model, &mut new);
        Ok(model.signature() != initial)
    }
}
