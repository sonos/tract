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
    vec![Box::new(PropConst), Box::new(DeclutterOps), Box::new(PushSplitDown), Box::new(ChangeAxes)]
}

pub fn codegen() -> Vec<Box<dyn TypedPass>> {
    vec![Box::new(CodegenOps), Box::new(PushSplitDown), Box::new(FuseOps)]
}

#[derive(Debug)]
pub struct DeclutterOps;

impl TypedPass for DeclutterOps {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool> {
        let mut hashset = std::collections::HashSet::new();
        let initial = model.signature();
        hashset.insert(initial);

        let mut new = model.clone();
        loop {
            for id in new.eval_order()? {
                let reduced = {
                    let node = &new.nodes()[id];
                    node.op
                        .declutter(&new, node)
                        .chain_err(|| format!("{:?} node {}", self, node))?
                };
                if let Some(red) = reduced {
                    debug!("Apply a model patch for {:?} {}", self, new.nodes()[id]);
                    red.apply(&mut new)?;
                    if cfg!(debug_assertions) {
                        new.check_edges()?;
                    }
                }
            }

            new = crate::model::compact::compact(&new)?;
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
                    debug!("Apply a model patch for {:?} {}", self, model.nodes()[id]);
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
                    debug!("Apply a model patch for {:?} {}", self, model.nodes()[id]);
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
