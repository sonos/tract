use crate::internal::*;
use std::collections::HashSet;
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

    pub fn optimize(&self, model: &mut TypedModel) -> TractResult<()> {
        #[cfg(all(debug_assertions, feature = "paranoid_assertions"))]
        {
            model.check_consistent_facts()?;
        }
        let mut seen = HashSet::new();
        model.compact()?;
        let mut counter = 0;
        for i in 0.. {
            let new_counter = self.run_all_passes(i, counter, model, &mut seen)?;
            if new_counter == counter {
                return Ok(());
            }
            counter = new_counter;
            model.compact()?;
        }
        unreachable!()
    }

    pub fn run_all_passes(
        &self,
        i: usize,
        mut counter: usize,
        model: &mut TypedModel,
        seen: &mut HashSet<String>,
    ) -> TractResult<usize> {
        let mut passes = self.passes.clone();
        for p in passes.iter_mut() {
            counter = self.run_one_pass_outer(i, p.as_mut(), counter, model, seen)?;
            model.compact()?;
        }
        Ok(counter)
    }

    pub fn run_one_pass_outer(
        &self,
        i: usize,
        p: &mut dyn TypedPass,
        mut counter: usize,
        model: &mut TypedModel,
        seen: &mut HashSet<String>,
    ) -> TractResult<usize> {
        loop {
            let new_counter = self.run_one_pass_inner(i, p, counter, model, seen)?;
            if new_counter == counter {
                return Ok(counter);
            }
            counter = new_counter;
            model.compact()?;
        }
    }

    pub fn run_one_pass_inner(
        &self,
        i: usize,
        p: &mut dyn TypedPass,
        mut counter: usize,
        model: &mut TypedModel,
        seen: &mut HashSet<String>,
    ) -> TractResult<usize> {
        p.reset()?;
        while let Some(mut patch) = p.next(&model)? {
            if let Some(steps) = self.steps {
                if counter >= steps {
                    return Ok(counter);
                }
            }
            patch.push_context(format!("{:?}/{}", p, i));
            #[cfg(all(debug_assertions, feature = "paranoid_assertions"))]
            {
                patch.model.check_consistent_facts()?;
                model.check_consistent_facts()?;
                patch.model.invariants()?;
                model.invariants()?;
            }
            if let Some(watchdog) = patch.dont_apply_twice.take() {
                if seen.contains(&watchdog) {
                    debug!("Loop detected: {} seen before", watchdog);
                    continue;
                } else {
                    seen.insert(watchdog);
                }
            }
            debug!("applying patch #{}: {}", counter, patch.context.iter().rev().join(" >> "),);
            patch.apply(model)?;
            counter += 1;
        }
        #[cfg(all(debug_assertions, feature = "paranoid_assertions"))]
        {
            model.check_edges()?;
            model
                .check_consistent_facts()
                .with_context(|| format!("after declutter pass {:?}", p))?
        }
        Ok(counter)
    }
}
