use crate::internal::*;
use std::collections::HashSet;
use std::fmt::Debug;
use tract_itertools::Itertools;

pub mod change_axes;
mod concat_then_einsum;
mod op_optim;
mod prop_const;
mod push_split_down;
mod slice;

use self::change_axes::ChangeAxes;
use self::prop_const::PropConst;
use self::push_split_down::PushSplitDown;
use self::slice::PushSliceUp;
use op_optim::OpOptim;

pub trait TypedPass: Debug + Send + Sync + dyn_clone::DynClone {
    fn reset(&mut self) -> TractResult<()>;
    fn next(
        &mut self,
        session: &mut OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>>;
}

dyn_clone::clone_trait_object!(TypedPass);

#[derive(Debug)]
pub struct Optimizer {
    pub passes: Vec<Box<dyn TypedPass>>,
    pub steps: Option<usize>,
}

impl Optimizer {
    fn passes(passes: Vec<Box<dyn TypedPass>>) -> Optimizer {
        Optimizer { passes, steps: None }
    }

    pub fn add_pass(&mut self, idx: usize, pass: Box<dyn TypedPass>) {
        let num_pass = self.passes.len();
        if idx > num_pass {
            log::warn!("Cannot add new pass {pass:?} at index {idx}. Optimizer currently as {num_pass} passes, pass will be added as the last pass.");
            self.passes.push(pass);
        } else {
            self.passes.insert(idx, pass);
        }
    }

    pub fn stopping_at(self, steps: usize) -> Optimizer {
        Optimizer { steps: Some(steps), ..self }
    }

    pub fn prop_consts() -> Optimizer {
        Optimizer::passes(vec![Box::<PropConst>::default()])
    }

    pub fn declutter() -> Optimizer {
        Optimizer::passes(vec![
            Box::<PropConst>::default(),
            Box::new(OpOptim("declutter", TypedOp::declutter_with_session, 0)),
            Box::new(PushSliceUp),
            Box::new(PushSplitDown),
            Box::<concat_then_einsum::ConcatThenEinsum>::default(),
            Box::<ChangeAxes>::default(),
        ])
    }

    pub fn codegen() -> Optimizer {
        Optimizer::passes(vec![
            Box::<PropConst>::default(),
            Box::new(OpOptim(
                "codegen",
                |op, _session, model, node| TypedOp::codegen(op, model, node),
                0,
            )),
            Box::new(OpOptim("declutter", TypedOp::declutter_with_session, 0)),
            Box::new(PushSplitDown),
            Box::new(OpOptim(
                "fuse",
                |op, _session, model, node| TypedOp::fuse(op, model, node),
                0,
            )),
        ])
    }

    pub fn optimize(&self, model: &mut TypedModel) -> TractResult<()> {
        self.session().optimize(model)
    }

    pub fn session(&self) -> OptimizerSession {
        OptimizerSession { optimizer: self, counter: 0, seen: Default::default() }
    }
}

#[derive(Debug)]
pub struct OptimizerSession<'o> {
    optimizer: &'o Optimizer,
    counter: usize,
    seen: HashSet<String>,
}

impl OptimizerSession<'_> {
    pub fn optimize(&mut self, model: &mut TypedModel) -> TractResult<()> {
        model.check_consistency().context("during optimizer preflight check")?;
        model.compact().context("during optimizer preflight compaction")?;
        model.check_names().context("after optimizer preflight compaction")?;
        for i in 0.. {
            let old = self.counter;
            self.run_all_passes(i, model)?;
            if old == self.counter {
                return Ok(());
            }
            model.compact()?;
        }
        unreachable!()
    }

    pub fn run_all_passes(&mut self, i: usize, model: &mut TypedModel) -> TractResult<()> {
        let mut passes = self.optimizer.passes.clone();
        for p in passes.iter_mut() {
            self.run_one_pass_outer(i, p.as_mut(), model)
                .with_context(|| format!("running pass {p:?}"))?;
            model.compact()?;
            model
                .check_consistency()
                .with_context(|| format!("consistency check after pass {p:?}"))?;
        }
        Ok(())
    }

    pub fn run_one_pass_outer(
        &mut self,
        i: usize,
        p: &mut dyn TypedPass,
        model: &mut TypedModel,
    ) -> TractResult<()> {
        loop {
            let old_counter = self.counter;
            self.run_one_pass_inner(i, p, model)?;
            if self.counter == old_counter {
                return Ok(());
            }
            model.compact().with_context(|| format!("after pass {p:?}"))?;
        }
    }

    pub fn run_one_pass_inner(
        &mut self,
        i: usize,
        p: &mut dyn TypedPass,
        model: &mut TypedModel,
    ) -> TractResult<()> {
        p.reset()?;
        if let Some(steps) = self.optimizer.steps {
            if self.counter >= steps {
                return Ok(());
            }
        }
        while let Some(mut patch) = p.next(self, model)? {
            patch.push_context(format!("{p:?}/{i}"));
            patch.model.check_consistency().context("checking patch internal consistency")?;
            model
                .check_consistency()
                .context("Checking target model consistency before patching")?;
            if let Some(watchdog) = patch.dont_apply_twice.take() {
                if self.seen.contains(&watchdog) {
                    debug!("Loop detected: {watchdog} seen before");
                    continue;
                } else {
                    self.seen.insert(watchdog);
                }
            }
            let patch_name = patch.context.iter().rev().join(" >> ");
            debug!("applying patch #{}: {patch_name}", self.counter);
            patch.apply(model).with_context(|| format!("Applying patch {patch_name}"))?;
            model
                .check_consistency()
                .context("Checking target model consistency after patching")?;
            self.counter += 1;
            if let Some(steps) = self.optimizer.steps {
                if self.counter >= steps {
                    return Ok(());
                }
            }
        }
        model.check_consistency().with_context(|| format!("after pass {p:?}"))?;
        Ok(())
    }
}
