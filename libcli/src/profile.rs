use tract_core::internal::*;

use crate::annotations::*;
use crate::model::Model;
use crate::tensor::make_inputs_for_model;
use std::time::{Duration, Instant};

pub struct BenchLimits {
    pub max_iters: usize,
    pub max_time: std::time::Duration,
}

impl Default for BenchLimits {
    fn default() -> Self {
        BenchLimits { max_iters: 100_000, max_time: std::time::Duration::from_secs(5) }
    }
}

pub fn profile(
    model: &TypedModel,
    bench_limits: &BenchLimits,
    dg: &mut Annotations,
    inputs: &TVec<TValue>,
) -> TractResult<()> {
    info!("Running entire network");
    let plan = SimplePlan::new(model)?;
    let mut state = SimpleState::new(&plan)?;
    let mut iters = 0usize;
    let start = Instant::now();
    while iters < bench_limits.max_iters && start.elapsed() < bench_limits.max_time {
        let _ =
            state.run_plan_with_eval(inputs.clone(), |session_state, state, node, input| {
                let start = Instant::now();
                let r = tract_core::plan::eval(session_state, state, node, input);
                let elapsed = start.elapsed();
                *dg.node_mut(NodeQId(tvec!(), node.id))
                    .profile
                    .get_or_insert(Duration::default()) += elapsed;
                r
            })?;
        iters += 1;
    }
    let entire = start.elapsed();

    info!("Running {} iterations max. for each node.", bench_limits.max_iters);
    info!("Running for {} ms max. for each node.", bench_limits.max_time.as_millis());

    for &outer_node in &plan.order {
        if let Some(m) = (model as &dyn Model).downcast_ref::<Graph<TypedFact, Box<dyn TypedOp>>>()
        {
            let outer_node = m.node(outer_node);
            let inputs: TVec<TypedFact> = model
                .node_input_facts(outer_node.id)?
                .iter()
                .map(|&i| i.to_typed_fact().map(|f| f.into_owned()))
                .collect::<TractResult<_>>()?;
            let ref_inputs: TVec<&TypedFact> = inputs.iter().collect();
            for ((inner_model_name, inner_model), multiplier) in model
                .nested_models(outer_node.id)
                .iter()
                .zip(model.nested_models_iters(outer_node.id, &ref_inputs).iter())
            {
                let multi = multiplier.as_ref().unwrap_or(&TDim::Val(1)).to_isize().unwrap();
                let prefix = tvec!((outer_node.id, inner_model_name.to_string()));
                if let Some(inner_model) = inner_model.downcast_ref::<TypedModel>() {
                    for _ in 0..iters {
                        let inner_plan = SimplePlan::new(inner_model)?;
                        let mut state = SimpleState::new(inner_plan)?;
                        let _ = state.run_plan_with_eval(
                            make_inputs_for_model(inner_model)?,
                            |session_state, state, node, input| {
                                let start = Instant::now();
                                let r = tract_core::plan::eval(session_state, state, node, input);
                                let elapsed = start.elapsed().mul_f32(multi as _);
                                *dg.node_mut(NodeQId(prefix.clone(), node.id))
                                    .profile
                                    .get_or_insert(Duration::default()) += elapsed;
                                let parent = dg
                                    .node_mut(NodeQId(tvec!(), outer_node.id))
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
    let entire = entire.mul_f32(denum);
    for d in dg.tags.values_mut() {
        if let Some(d) = d.profile.as_mut() {
            *d = d.mul_f32(denum);
        }
    }
    let max = dg.tags.values().filter_map(|t| t.profile).max().unwrap();
    let sum = dg.tags.values().filter_map(|t| t.profile).sum::<Duration>();
    dg.profile_summary = Some(ProfileSummary { max, sum, entire, iters });
    Ok(())
}

pub fn extract_costs(annotations: &mut Annotations, model: &dyn Model) -> TractResult<()> {
    fn extract_costs_rec(
        annotations: &mut Annotations,
        model: &dyn Model,
        prefix: &[(usize, String)],
        multiplier: TDim,
    ) -> TractResult<()> {
        if let Some(model) = model.downcast_ref::<TypedModel>() {
            for node_id in 0..model.nodes().len() {
                let inputs = model.node_input_facts(node_id)?;
                let cost = model.node(node_id).op.cost(&inputs)?;
                annotations.node_mut(NodeQId(prefix.into(), node_id)).cost = cost
                    .into_iter()
                    .map(|(k, v)| (k, if k.is_compute() { v * &multiplier } else { v }))
                    .collect();

                let nested_subs = model.nested_models(node_id);
                let nested_multis = (model as &dyn Model).nested_models_iters(node_id, &inputs);
                for ((name, sub), multi) in nested_subs.iter().zip(nested_multis.iter()) {
                    let mut prefix: TVec<_> = prefix.into();
                    prefix.push((node_id, name.to_string()));
                    extract_costs_rec(
                        annotations,
                        *sub,
                        &prefix,
                        multi.clone().unwrap_or_else(|| 1.into()) * &multiplier,
                    )?;
                }
            }
        }
        Ok(())
    }
    extract_costs_rec(annotations, model, &[], 1.into())
}
