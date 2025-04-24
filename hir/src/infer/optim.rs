use crate::infer::*;
use std::fmt;

pub fn incorporate() -> Vec<Box<dyn IncorporatePass>> {
    vec![Box::new(IncorporateOps)]
}

pub trait IncorporatePass: fmt::Debug + Send + Sync {
    fn pass(&self, model: &mut InferenceModel) -> TractResult<bool>;
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
                    trace!("Incorporate {node}");
                    node.op
                        .incorporate(model, node)
                        .with_context(|| format!("{self:?} node {node}"))?
                };
                if let Some(red) = reduced {
                    {
                        let node = &model.nodes()[id];
                        debug!("Apply a model patch for {self:?}: {node}");
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
