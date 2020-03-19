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

pub trait TypedPass: Debug {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool>;
}

pub fn declutter() -> Vec<Box<dyn TypedPass>> {
    vec![
        Box::new(PropConst),
        Box::new(TypedNodeByNodePass(Box::new(DeclutterOps))),
        Box::new(PushSplitDown),
        Box::new(ChangeAxes),
    ]
}

pub fn codegen() -> Vec<Box<dyn TypedPass>> {
    vec![
        Box::new(TypedNodeByNodePass(Box::new(CodegenOps))),
        Box::new(PushSplitDown),
        Box::new(TypedNodeByNodePass(Box::new(FuseOps))),
    ]
}

trait NodePatch: Debug + 'static {
    fn pass_one(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>>;
}

#[derive(Debug)]
struct TypedNodeByNodePass(Box<dyn NodePatch>);

impl TypedPass for TypedNodeByNodePass {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool> {
        if cfg!(debug_assertions) {
            model.check_edges().chain_err(|| format!("preliminary check, {:?}", self))?
        }
        let mut done_something = false;
        loop {
            let mut done_something_this_time = false;
            for id in model.eval_order()? {
                let patch = {
                    let node = &model.nodes()[id];
                    self.0
                        .pass_one(model, node)
                        .chain_err(|| format!("{:?} node {}", self, node))?
                };
                if let Some(patch) = patch {
                    if cfg!(debug_assertions) {
                        let saved = model.clone();
                        let saved_patch = patch.clone();
                        patch.apply(model).and_then(|_| model.check_edges()).chain_err(|| {
                            format!("Applying patch {:#?} to {:#?}", saved_patch, saved)
                        })?
                    } else {
                        patch.apply(model)?
                    }
                    done_something_this_time = true
                }
            }
            done_something = done_something || done_something_this_time;
            if !done_something_this_time {
                break;
            }
        }
        Ok(done_something)
    }
}

#[derive(Debug)]
pub struct DeclutterOps;

impl NodePatch for DeclutterOps {
    fn pass_one(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        node.op.declutter(model, node)
    }
}

#[derive(Debug)]
pub struct CodegenOps;

impl NodePatch for CodegenOps {
    fn pass_one(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        node.op.codegen(model, node)
    }
}

#[derive(Debug)]
pub struct FuseOps;

impl NodePatch for FuseOps {
    fn pass_one(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        node.op.fuse(model, node)
    }
}
