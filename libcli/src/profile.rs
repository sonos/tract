use tract_core::{
    internal::*,
    ops::{scan::State, submodel::TypedModelOpState},
};

use crate::model::Model;
use crate::{annotations::*, tensor::make_inputs_for_model};
use std::{
    any::TypeId,
    time::{Duration, Instant},
};

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
    custom_profiler: Option<HashMap<TypeId, Profiler>>,
) -> TractResult<()> {
    info!("Running entire network");
    let mut iters = 0usize;
    let mut prefix = tvec!();
    let plan = TypedSimplePlan::new(model.clone())?;
    let mut state = TypedSimpleState::new(Arc::new(plan))?;

    let start = Instant::now();
    while iters < bench_limits.max_iters && start.elapsed() < bench_limits.max_time {
        rec_profiler(&mut state, dg, custom_profiler.as_ref(), &mut prefix, None)?;
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

pub fn rec_profiler(
    state: &mut TypedSimpleState<TypedModel, Arc<TypedSimplePlan<TypedModel>>>,
    dg: &mut Annotations,
    profilers: Option<&HashMap<TypeId, Profiler>>,
    prefix: &[(usize, String)],
    multiplier: Option<isize>,
) -> TractResult<TVec<TValue>> {
    let model = state.plan().model();
    let inputs = make_inputs_for_model(model)?;
    let mut parent_prefix: TVec<_> = prefix.into();
    parent_prefix.pop();
    let r = state.run_plan_with_eval(inputs, |session_state, mut node_state, node, input| { 
        let (r, e) = if let Some(ref mut op_state) = node_state {
            // Run top node
            let start = Instant::now();
            let r = tract_core::plan::eval(session_state, Some(*op_state), node, input.clone());
            let elapsed = start.elapsed().mul_f32(multiplier.unwrap_or(1) as _);
            *dg.node_mut(NodeQId(prefix.into(), node.id)).profile.get_or_insert(Duration::default()) +=
                elapsed;
            
            // Run inner node model
            if let Some(profiler) = profilers.map(|it| it.get(&op_state.type_id())).flatten() {
                let mut prefix: TVec<_> = prefix.into();
                prefix.push((node.id, "submodel".to_string()));
                (profiler.func)(*op_state, input.clone(), dg, &prefix)?;
            } else if let Some(scan_state) = op_state.downcast_mut::<State>() {
                let mut prefix: TVec<_> = prefix.into();
                prefix.push((node.id, "loop".to_string()));

                let input_facts  = node.inputs.iter().map(|outlet| node.outputs.get(outlet.slot).map(|o| &o.fact).ok_or(anyhow!(""))).collect::<TractResult<TVec<_>>>()?;
                let multi = if let Some(lir) = node.op_as::<tract_core::ops::scan::LirScan>() {
                    lir.iteration_count(&input_facts).map(|it| it.to_isize().unwrap())
                } else if let Some(mir) = node.op_as::<tract_core::ops::scan::Scan>() {
                    mir.iteration_count(&input_facts).map(|it| it.to_isize().unwrap())
                } else {
                    None
                };
                rec_profiler(&mut scan_state.model_state, dg, None, &prefix, multi)?;
            } else if let Some(typed_model_state) = op_state.downcast_mut::<TypedModelOpState>() {
                let mut prefix: TVec<_> = prefix.into();
                prefix.push((node.id, "submodel".to_string()));
                rec_profiler(typed_model_state, dg, None, &prefix, None)?;
            }

            (r, elapsed)
        } else {
            // Profile node
            let start = Instant::now();
            let r = tract_core::plan::eval(session_state, node_state, node, input);
            let elapsed = start.elapsed().mul_f32(multiplier.unwrap_or(1) as _);
            *dg.node_mut(NodeQId(prefix.into(), node.id)).profile.get_or_insert(Duration::default()) += elapsed;
            (r, elapsed)
        };
        if prefix.len() > 0 {
            let parent = dg
                .node_mut(NodeQId(parent_prefix.clone(), prefix.last().map(|it| it.0).unwrap()))
                .profile
                .get_or_insert(Duration::default());
            *parent -= e.min(*parent);
        }
        r
    })?;
    Ok(r)
}

type ProfilerFn = fn(
    &mut dyn OpState,
    TVec<TValue>,
    &mut Annotations,
    &[(usize, String)],
) -> TractResult<TVec<TValue>>;

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
