use crate::model::*;
use crate::infer::InferenceModel;
use crate::TractResult;
use std::fmt::Debug;

pub mod change_axes;
mod prop_const;
mod push_split_down;

use self::change_axes::ChangeAxes;
use self::prop_const::PropConst;
use self::push_split_down::PushSplitDown;

use crate::errors::TractResultExt;

pub trait IncorporatePass: Debug + Send + Sync {
    fn pass(&self, model: &mut InferenceModel) -> TractResult<bool>;
}

pub trait TypedPass: Debug + Send + Sync {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool>;
}

pub fn incorporate() -> Vec<Box<dyn IncorporatePass>> {
    vec![Box::new(IncorporateOps)]
}

pub fn declutter() -> Vec<Box<dyn TypedPass>> {
    vec![
        Box::new(PropConst),
        Box::new(DeclutterOps),
        Box::new(PushSplitDown),
        Box::new(ChangeAxes),
    ]
}

pub fn codegen() -> Vec<Box<dyn TypedPass>> {
    vec![Box::new(CodegenOps), Box::new(PushSplitDown), Box::new(FuseOps)]
}

#[derive(Debug)]
pub struct IncorporateOps;

impl IncorporatePass for IncorporateOps {
    fn pass(&self, model: &mut InferenceModel) -> TractResult<bool> {
        let mut done_something = false;
        loop {
            let mut done_something_this_time = false;
            for id in model.eval_order()? {
                let reduced = {
                    let node = &model.nodes()[id];
                    debug!("Incorporate {}", node);
                    node.op
                        .incorporate(model, node)
                        .chain_err(|| format!("{:?} node {}", self, node))?
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
pub struct DeclutterOps;

impl TypedPass for DeclutterOps {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool> {
        let mut done_something = false;
        loop {
            let mut done_something_this_time = false;
            for id in model.eval_order()? {
                let reduced = {
                    let node = &model.nodes()[id];
                    node.op
                        .declutter(model, node)
                        .chain_err(|| format!("{:?} node {}", self, node))?
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

impl TypedPass for CodegenOps {
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
                        .chain_err(|| format!("{:?} node {}", self, node))?
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

#[derive(Debug)]
pub struct FuseOps;

impl TypedPass for FuseOps {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool> {
        let mut done_something = false;
        loop {
            let mut done_something_this_time = false;
            for id in model.eval_order()? {
                let reduced = {
                    let node = &model.nodes()[id];
                    debug!("Fuse {}", node);
                    node.op.fuse(model, node).chain_err(|| format!("{:?} node {}", self, node))?
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
