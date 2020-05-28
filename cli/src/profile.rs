use tract_core::internal::*;

use crate::annotations::*;
use crate::errors::*;
use crate::BenchLimits;
use std::time::{Duration, Instant};

trait Scalable {
    fn scale(self, scale: f32) -> Self;
}

impl Scalable for std::time::Duration {
    fn scale(self, scale: f32) -> Duration {
        Duration::from_secs_f32(scale * self.as_secs_f32())
    }
}

#[derive(Debug, Clone)]
pub struct ProfileSummary {
    pub max: Duration,
    pub sum: Duration,
    pub entire: Duration,
    pub iters: usize,
}

pub fn profile(
    model: &TypedModel,
    bench_limits: &BenchLimits,
    dg: &mut Annotations,
) -> CliResult<()> {
    info!("Running entire network");
    let plan = SimplePlan::new(model)?;
    let mut state = SimpleState::new(&plan)?;
    let mut iters = 0usize;
    let start = Instant::now();
    while iters < bench_limits.max_iters && start.elapsed() < bench_limits.max_time {
        let _ = state.run_plan_with_eval(
            crate::tensor::make_inputs_for_model(model)?,
            0,
            |session_state, state, node, input| {
                let start = Instant::now();
                let r = tract_core::plan::eval(session_state, state, node, input);
                let elapsed = start.elapsed();
                *dg.node_mut(NodeQId(tvec!(), node.id))
                    .profile
                    .get_or_insert(Duration::default()) += elapsed;
                r
            },
        )?;
        iters += 1;
    }
    let entire = start.elapsed();

    info!("Running {} iterations max. for each node.", bench_limits.max_iters);
    info!("Running for {} ms max. for each node.", bench_limits.max_time.as_millis());

    for &outer_node in &plan.order {
        if let Some(m) =
            (model as &dyn Model).downcast_ref::<ModelImpl<TypedFact, Box<dyn TypedOp>>>()
        {
            let outer_node = m.node(outer_node);
            let inputs: TVec<TypedFact> = model
                .node_input_facts(outer_node.id)?
                .iter()
                .map(|&i| i.to_typed_fact())
                .collect::<TractResult<_>>()?;
            let ref_inputs: TVec<&TypedFact> = inputs.iter().collect();
            for ((inner_model_name, inner_model, _, _), (_name_, multiplier)) in outer_node
                .op
                .nested_models()
                .iter()
                .zip(outer_node.op.nested_model_multipliers(&ref_inputs).iter())
            {
                let prefix = tvec!((outer_node.id, inner_model_name.to_string()));
                if let Some(inner_model) = inner_model.downcast_ref::<TypedModel>() {
                    for _ in 0..iters {
                        let inner_plan = SimplePlan::new(inner_model)?;
                        let mut state = SimpleState::new(inner_plan)?;
                        let _ = state.run_plan_with_eval(
                            crate::tensor::make_inputs_for_model(inner_model)?,
                            0,
                            |session_state, state, node, input| {
                                let start = Instant::now();
                                let r = tract_core::plan::eval(session_state, state, node, input);
                                let elapsed = start.elapsed().scale(*multiplier as _);
                                *dg.node_mut(NodeQId(prefix.clone(), node.id))
                                    .profile
                                    .get_or_insert(Duration::default()) += elapsed;
                                let parent = dg.node_mut(NodeQId(tvec!(), outer_node.id))
                                    .profile
                                    .get_or_insert(Duration::default());
                                *parent -= elapsed.min(*parent);
                                r
                            },
                        )?;
                    }
                }
            }
        }
    }
    let denum = (iters as f32).recip();
    let entire = entire.scale(denum);
    for d in dg.tags.values_mut() {
        if let Some(d) = d.profile.as_mut() {
            *d = d.scale(denum);
        }
    }
    let max = dg.tags.values().filter_map(|t| t.profile).max().unwrap();
    let sum = dg.tags.values().filter_map(|t| t.profile).sum::<Duration>();
    dg.profile_summary = Some(ProfileSummary { max, sum, entire, iters });
    Ok(())
}
