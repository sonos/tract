use crate::model::TypedModel;
use crate::TractResult;
use std::fmt::Debug;

mod prop_const;
mod push_split_down;

use self::prop_const::PropConst;
use self::push_split_down::PushSplitDown;

pub trait DeclutterPass: Debug + Send + Sync {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool>;
}

pub trait CodegenPass: Debug + Send + Sync {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool>;
}

pub fn declutter() -> Vec<Box<DeclutterPass>> {
    vec![Box::new(IncorporateOps), Box::new(PropConst) as _, Box::new(NormalizeOps)]
}

pub fn codegen() -> Vec<Box<CodegenPass>> {
    vec![Box::new(CodegenOps), Box::new(PushSplitDown)]
}

#[derive(Debug)]
pub struct IncorporateOps;

impl DeclutterPass for IncorporateOps {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool> {
        let mut done_something = false;
        loop {
            let mut done_something_this_time = false;
            for id in model.eval_order()? {
                let reduced = {
                    let node = &model.nodes()[id];
                    debug!("Incorporate {}", node);
                    node.op
                        .incorporate(model, node)
                        .map_err(|e| format!("{:?} node {}, {:?}", self, node, e))?
                };
                if let Some(red) = reduced {
                    {
                        let node = &model.nodes()[id];
                        debug!("Apply a model patch for {:?}: {}", self, node);
                    }
                    red.apply(model)?;
                    if cfg!(debug_assertions) {
                        model.check_edges()?;
                    }
                    done_something_this_time = true;
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
pub struct NormalizeOps;

impl DeclutterPass for NormalizeOps {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool> {
        let mut done_something = false;
        loop {
            let mut done_something_this_time = false;
            for id in model.eval_order()? {
                let reduced = {
                    let node = &model.nodes()[id];
                    debug!("Decluttering {}", node);
                    node.op
                        .declutter(model, node)
                        .map_err(|e| format!("{:?} node {}, {:?}", self, node, e))?
                };
                if let Some(red) = reduced {
                    {
                        let node = &model.nodes()[id];
                        debug!("Apply a model patch for {:?}: {}", self, node);
                    }
                    red.apply(model)?;
                    if cfg!(debug_assertions) {
                        model.check_edges()?;
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
pub struct CodegenOps;

impl CodegenPass for CodegenOps {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool> {
        let mut done_something = false;
        loop {
            let mut done_something_this_time = false;
            for id in model.eval_order()? {
                let reduced = {
                    let node = &model.nodes()[id];
                    debug!("Codegen {}", node);
                    node.op
                        .codegen(model, node)
                        .map_err(|e| format!("{:?} node {}, {:?}", self, node, e))?
                };
                if let Some(red) = reduced {
                    {
                        let node = &model.nodes()[id];
                        debug!("Apply a model patch for {:?} {}", self, node);
                    }
                    red.apply(model)?;
                    if cfg!(debug_assertions) {
                        model.check_edges()?;
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
