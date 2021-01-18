use crate::internal::*;
use std::fmt::Debug;
use tract_itertools::Itertools;

pub mod change_axes;
mod op_optim;
mod prop_const;
mod push_split_down;

use self::change_axes::ChangeAxes;
use self::prop_const::PropConst;
use self::push_split_down::PushSplitDown;
use op_optim::OpOptim;

pub trait TypedPass: Debug + Send + Sync + dyn_clone::DynClone {
    fn reset(&mut self) -> TractResult<()>;
    fn next(&mut self, model: &TypedModel) -> TractResult<Option<TypedModelPatch>>;
}

dyn_clone::clone_trait_object!(TypedPass);

pub struct Optimizer {
    passes: Vec<Box<dyn TypedPass>>,
    steps: Option<usize>,
}

impl Optimizer {
    fn passes(passes: Vec<Box<dyn TypedPass>>) -> Optimizer {
        Optimizer { passes, steps: None }
    }

    pub fn stopping_at(self, steps: usize) -> Optimizer {
        Optimizer { steps: Some(steps), ..self }
    }

    pub fn declutter() -> Optimizer {
        Optimizer::passes(vec![
            Box::new(OpOptim("declutter", TypedOp::declutter, 0)),
            Box::new(PropConst),
            Box::new(PushSplitDown),
            Box::new(ChangeAxes),
        ])
    }

    pub fn codegen() -> Optimizer {
        Optimizer::passes(vec![
            Box::new(OpOptim("codegen", TypedOp::codegen, 0)),
            Box::new(OpOptim("declutter", TypedOp::declutter, 0)),
            Box::new(PropConst),
            Box::new(PushSplitDown),
            Box::new(OpOptim("fuse", TypedOp::fuse, 0)),
        ])
    }

    pub fn optimize(&self, model: &TypedModel) -> TractResult<TypedModel> {
        #[cfg(all(debug_assertions, feature = "paranoid_assertions"))]
        {
            model.check_consistent_facts()?;
        }
        let mut model = model.clone();
        let mut patches = 0;
        let mut passes = self.passes.clone();
        for i in 0.. {
            model = model.compact()?;
            let mut done_something_this_time = false;
            'pass: for p in passes.iter_mut() {
                loop {
                    let mut done_something_this_pass = false;
                    let mut seen = std::collections::HashSet::new();
                    p.reset()?;
                    while let Some(mut patch) = p.next(&model)? {
                        patch.push_context(format!("{:?}/{}", p, i));
                        #[cfg(all(debug_assertions, feature = "paranoid_assertions"))]
                        {
                            patch.model.check_consistent_facts()?;
                            model.check_consistent_facts()?;
                            patch.model.invariants()?;
                            model.invariants()?;
                        }
                        if let Some(watchdog) = patch.dont_apply_twice.take() {
                            if !seen.contains(&watchdog) {
                                debug!("Loop detected: {} seen before", watchdog);
                                model = model.compact()?;
                                break 'pass;
                            } else {
                                seen.insert(watchdog);
                            }
                        }
                        debug!(
                            "applying patch #{}: {}",
                            patches,
                            patch.context.iter().rev().join(" >> "),
                        );
                        done_something_this_pass = true;
                        done_something_this_time = true;
                        patch.apply(&mut model)?;
                        seen.clear();
                        patches += 1;
                        if let Some(steps) = self.steps {
                            if patches >= steps {
                                return Ok(model);
                            }
                        }
                    }
                    #[cfg(all(debug_assertions, feature = "paranoid_assertions"))]
                    {
                        model.check_edges()?;
                        model
                            .check_consistent_facts()
                            .with_context(|| format!("after declutter pass {:?}", p))?
                    }
                    if !done_something_this_pass {
                        continue 'pass;
                    }
                }
            }
            if !done_something_this_time {
                return Ok(model);
            }
            model = model.compact()?;
        }
        unreachable!()
    }
}
