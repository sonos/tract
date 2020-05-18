use tract_core::internal::*;

use crate::display_graph::*;
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
}

/*
#[derive(Clone, Debug, Default)]
pub struct ProfileData {
pub nodes: HashMap<TVec<(usize, String)>, Duration>,
pub sum: Duration,
pub entire: Duration,
}

impl ProfileData {
pub fn add(
&mut self,
node_id: &[(usize, String)],
dur: Duration,
) -> ::tract_core::TractResult<()> {
 *self.nodes.entry(node_id.into()).or_insert(Duration::default()) += dur;
 self.sum += dur;
 Ok(())
 }

 pub fn sub(
 &mut self,
 node_id: &[(usize, String)],
 dur: Duration,
 ) -> ::tract_core::TractResult<()> {
 *self.nodes.entry(node_id.into()).or_insert(Duration::default()) -= dur;
 self.sum -= dur;
 Ok(())
 }

 fn op_name_for_id(model: &dyn Model, id: &[(usize, String)]) -> CliResult<String> {
 if id.len() == 1 {
 Ok(model.node_op(id[0].0).name().into_owned())
 } else {
 let token = &id[0];
 let model = model
 .node_op(token.0)
 .as_typed()
 .unwrap()
 .nested_models()
 .iter()
 .find(|m| m.0 == token.1)
 .unwrap()
 .1;
 Self::op_name_for_id(model, &id[1..])
 }
 }

 pub fn by_ops(&self, model: &dyn Model) -> CliResult<HashMap<String, (Duration, usize)>> {
 let mut operations = HashMap::new();
 for (node, dur) in &self.nodes {
 let op_name = Self::op_name_for_id(model, node)?;
 let entry = operations.entry(op_name.clone()).or_insert((Duration::default(), 1));
 entry.0 += *dur;
 entry.1 += 1;
 }
 Ok(operations)
 }

 pub fn scale(&mut self, factor: f32) {
 self.nodes.values_mut().for_each(|n| {
 n.scale(factor);
 });
 self.sum.scale(factor);
 }
 }
 */

pub fn profile(
    model: &TypedModel,
    bench_limits: &BenchLimits,
    dg: &mut DisplayGraph,
) -> CliResult<()> {
    info!("Running entire network");
    let plan = SimplePlan::new(model)?;
    let mut state = SimpleState::new(&plan)?;
    let mut iters = 0;
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
                                *dg.node_mut(NodeQId(tvec!(), outer_node.id))
                                    .profile
                                    .get_or_insert(Duration::default()) -= elapsed;
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
    dg.tags.values_mut().for_each(|t| {
        t.profile.as_mut().map(|d| d.scale(denum));
    });
    let max = dg.tags.values().filter_map(|t| t.profile).max().unwrap();
    let sum = dg.tags.values().filter_map(|t| t.profile).sum::<Duration>();
    dg.profile_summary = Some(ProfileSummary { max, sum, entire });
    Ok(())
}

/*
fn prefixes_for(s: &str) -> impl Iterator<Item = String> + '_ {
use tract_itertools::*;
let split = s.split(".").count() - 1;
(0..split).map(move |n| s.split(".").take(n).join("."))
}
*/
