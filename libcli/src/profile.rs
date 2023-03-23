use tract_core::{internal::*, ops::submodel::{SubmodelOp, InnerModel}};

use crate::{annotations::*, tensor::make_inputs_for_model};
use crate::model::Model;
use std::{time::{Duration, Instant}, any::TypeId};

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
    custom_profiler: Option<HashMap<TypeId, Profiler>>,
) -> TractResult<()> {
    info!("Running entire network");
    let mut iters = 0usize;
    let start = Instant::now();
    let mut prefix = tvec!();
    
    while iters < 1 && start.elapsed() < bench_limits.max_time {
        rec_profiler(model, inputs.clone(), dg, custom_profiler.as_ref(), &mut prefix)?;
        iters += 1;
    }
    let entire = start.elapsed();
    info!("Running {} iterations max. for each node.", bench_limits.max_iters);
    info!("Running for {} ms max. for each node.", bench_limits.max_time.as_millis());

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

pub fn rec_profiler(model: &TypedModel, inputs: TVec<TValue>, dg: &mut Annotations, profilers: Option<&HashMap<TypeId, Profiler>>, prefix: &[(usize, String)]) -> TractResult<TVec<TValue>> {
    let plan = TypedSimplePlan::new(model)?;
    let mut state = TypedSimpleState::new(&plan)?;
    let mut parent_prefix: TVec<_> = prefix.into();
    parent_prefix.pop();
    let r = state.run_plan_with_eval(
        inputs,
        |session_state, state, node, input| {
            // Specific case for submodels
            if let Some(submodel) = node.op_as::<SubmodelOp>() {
                let mut prefix: TVec<_> = prefix.into();
                prefix.push((node.id, "submodel".to_string()));
                
                // Check if submodel has specific profiler
                if let Some(profiler) = profilers.map(|it| it.get(&submodel.model.as_ref().type_id())).flatten() {
                    (profiler.func)(&submodel.model, input, dg, &prefix)
                } else {
                    rec_profiler(submodel.model(), input, dg, profilers, &prefix)
                }
            // Specific case for nested models
            } else if let Some((inner_model_name, inner_model)) = model.nested_models(node.id) {
                let mut prefix: TVec<_> = prefix.into();
                prefix.push((node.id, inner_model_name.to_string()));
                let inner_model = inner_model.downcast_ref::<TypedModel>().ok_or(anyhow!("Expected inner model to be a TypedModel"))?;
                let inner_input = make_inputs_for_model(inner_model)?;
                rec_profiler(inner_model, inner_input, dg, None, &prefix)
            } else {
                let start = Instant::now();
                let r = tract_core::plan::eval(session_state, state, node, input);
                let elapsed = start.elapsed().mul_f32(1 as _);
                *dg.node_mut(NodeQId(prefix.into(), node.id))
                    .profile
                    .get_or_insert(Duration::default()) += elapsed;
               
                if prefix.len() > 0 {
                    let parent = dg
                        .node_mut(NodeQId(parent_prefix.clone(), prefix.last().map(|it| it.0).unwrap()))
                        .profile
                        .get_or_insert(Duration::default());
                    *parent += elapsed;
                }
                r
            }
        },
    )?;
    Ok(r)
}

type ProfilerFn = fn(&Box<dyn InnerModel>, TVec<TValue>, &mut Annotations, &[(usize, String)]) -> TractResult<TVec<TValue>>;

#[derive(Clone)]
pub struct Profiler {
    pub func: ProfilerFn,
    pub name: &'static str,
}

impl Hash for Profiler {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
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
                if let Some((name, sub)) = nested_subs {
                    let mut prefix: TVec<_> = prefix.into();
                    prefix.push((node_id, name.to_string()));
                    extract_costs_rec(
                        annotations,
                        sub,
                        &prefix,
                        nested_multis.clone().unwrap_or_else(|| 1.into()) * &multiplier,
                    )?;
                }
            }
        }
        Ok(())
    }
    extract_costs_rec(annotations, model, &[], 1.into())
}
