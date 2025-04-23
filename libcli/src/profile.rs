use tract_core::internal::*;
use tract_core::num_traits::Zero;
use tract_core::ops::scan::State;
use tract_core::ops::submodel::TypedModelOpState;

use crate::annotations::*;
use crate::model::Model;
use crate::tensor::make_inputs_for_model;
use std::any::TypeId;
use std::time::Duration;

pub struct BenchLimits {
    pub warmup_loops: usize,
    pub warmup_time: std::time::Duration,
    pub max_loops: usize,
    pub max_time: std::time::Duration,
}

impl Default for BenchLimits {
    fn default() -> Self {
        BenchLimits {
            warmup_loops: 0,
            warmup_time: Duration::default(),
            max_loops: 100_000,
            max_time: std::time::Duration::from_secs(5),
        }
    }
}

impl BenchLimits {
    pub fn warmup(&self, model: &TypedModel, inputs: &TVec<TValue>) -> TractResult<()> {
        if self.warmup_time.is_zero() && self.warmup_loops.is_zero() {
            return Ok(());
        }
        let plan = TypedSimplePlan::new(model.clone())?;
        let mut state = TypedSimpleState::new(Arc::new(plan))?;
        let mut iters = 0;
        let max_loops = if self.warmup_loops.is_zero() { usize::MAX } else { self.warmup_loops };
        let max_time = if self.warmup_time.is_zero() { Duration::MAX } else { self.warmup_time };

        let start_warmup = crate::time::now();
        debug!("Warming up before profiling...");
        while iters < max_loops && start_warmup.elapsed() < max_time {
            state.run(inputs.clone())?;
            iters += 1;
        }
        debug!("Done warming up.");
        Ok(())
    }
}

pub fn profile(
    model: &TypedModel,
    bench_limits: &BenchLimits,
    dg: &mut Annotations,
    plan_options: &PlanOptions,
    inputs: &TVec<TValue>,
    custom_profiler: Option<HashMap<TypeId, Profiler>>,
    folded: bool,
) -> TractResult<()> {
    info!("Running entire network");
    let mut iters = 0usize;
    let prefix = tvec!();

    bench_limits.warmup(model, inputs)?;

    let plan = TypedSimplePlan::new_with_options(model.clone(), plan_options)?;
    let mut state = TypedSimpleState::new(Arc::new(plan))?;

    let start = crate::time::now();
    let mut time_accounted_by_inner_nodes = Duration::default();
    while iters < bench_limits.max_loops && start.elapsed() < bench_limits.max_time {
        rec_profiler(
            &mut state,
            dg,
            inputs,
            custom_profiler.as_ref(),
            &prefix,
            None,
            &mut time_accounted_by_inner_nodes,
            folded,
        )?;

        iters += 1;
    }

    let entire = start.elapsed() - time_accounted_by_inner_nodes;

    info!("Running {} iterations max. for each node.", bench_limits.max_loops);
    info!("Running for {} ms max. for each node.", bench_limits.max_time.as_millis());

    let denum = (iters as f32).recip();
    let entire = entire.mul_f32(denum);
    for d in dg.tags.values_mut() {
        if let Some(d) = d.profile.as_mut() {
            *d = d.mul_f32(denum);
        }

        if let Some(d) = d.accelerator_profile.as_mut() {
            *d = d.mul_f32(denum);
        }
    }
    let max = dg.tags.values().filter_map(|t| t.profile).max().unwrap();
    let sum = dg.tags.values().filter_map(|t| t.profile).sum::<Duration>();
    let accel_sum = dg.tags.values().filter_map(|t| t.accelerator_profile).sum::<Duration>();
    dg.profile_summary = Some(ProfileSummary { max, sum, accel_sum, entire, iters });
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn profile_metal(
    model: &TypedModel,
    bench_limits: &BenchLimits,
    dg: &mut Annotations,
    plan_options: &PlanOptions,
    inputs: &TVec<TValue>,
) -> TractResult<()> {
    info!("Running entire network");
    let mut iters = 0usize;
    let prefix = tvec!();

    bench_limits.warmup(model, inputs)?;

    let mut plan = TypedSimplePlan::new_with_options(model.clone(), plan_options)?;
    let state = TypedSimpleState::new_from_inputs(&plan, inputs.clone())?;

    let session_handler =
        tract_gpu::session_handler::DeviceSessionHandler::from_plan(&plan, &state.session_state.resolved_symbols)?;

    plan = plan.with_session_handler(session_handler);

    let mut state = TypedSimpleState::new(Arc::new(plan))?;

    let mut entire = Duration::default();
    while iters < bench_limits.max_loops && entire < bench_limits.max_time {
        entire += rec_profiler_metal(&mut state, dg, inputs, &prefix)?.1;

        iters += 1;
    }

    info!("Running {} iterations max. for each node.", bench_limits.max_loops);
    info!("Running for {} ms max. for each node.", bench_limits.max_time.as_millis());

    let denum = (iters as f32).recip();
    let entire = entire.mul_f32(denum);
    for d in dg.tags.values_mut() {
        if let Some(d) = d.profile.as_mut() {
            *d = d.mul_f32(denum);
        }

        if let Some(d) = d.accelerator_profile.as_mut() {
            *d = d.mul_f32(denum);
        }
    }
    let max = dg.tags.values().filter_map(|t| t.profile).max().unwrap();
    let sum = dg.tags.values().filter_map(|t| t.profile).sum::<Duration>();
    let accel_sum = dg.tags.values().filter_map(|t| t.accelerator_profile).sum::<Duration>();
    dg.profile_summary = Some(ProfileSummary { max, sum, accel_sum, entire, iters });
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn rec_profiler_metal(
    state: &mut TypedSimpleState<TypedModel, Arc<TypedSimplePlan<TypedModel>>>,
    dg: &mut Annotations,
    inputs: &TVec<TValue>,
    prefix: &[(usize, String)],
) -> TractResult<(TVec<TValue>, Duration)> {
    let profile_start = crate::time::now();
    let r = state.run_plan_with_eval(
        inputs.clone(),
        |session_state, mut node_state, node, input| {
            // Profile node
            let start = crate::time::now();
            let res = tract_core::plan::eval(
                session_state,
                node_state.as_deref_mut(),
                node,
                input.clone(),
            );
            let elapsed = start.elapsed();
            let node_id = NodeQId(prefix.into(), node.id);
            *dg.node_mut(node_id).profile.get_or_insert(Duration::default()) += elapsed;

            res
        },
    )?;

    Ok((r, profile_start.elapsed()))
}

#[allow(clippy::too_many_arguments)]
pub fn rec_profiler(
    state: &mut TypedSimpleState<TypedModel, Arc<TypedSimplePlan<TypedModel>>>,
    dg: &mut Annotations,
    inputs: &TVec<TValue>,
    profilers: Option<&HashMap<TypeId, Profiler>>,
    prefix: &[(usize, String)],
    multiplier: Option<usize>,
    time_accounted_by_inner_nodes: &mut Duration,
    folded: bool,
) -> TractResult<TVec<TValue>> {
    let r = state.run_plan_with_eval(
        inputs.clone(),
        |session_state, mut node_state, node, input| {
            // Profile node
            let start = crate::time::now();
            let res = tract_core::plan::eval(
                session_state,
                node_state.as_deref_mut(),
                node,
                input.clone(),
            );
            let elapsed = start.elapsed().mul_f32(multiplier.unwrap_or(1) as _);
            let node_id = NodeQId(prefix.into(), node.id);
            *dg.node_mut(node_id).profile.get_or_insert(Duration::default()) += elapsed;

            if !folded {
                let start = crate::time::now();
                profile_submodel(
                    node,
                    node_state,
                    input,
                    dg,
                    profilers,
                    prefix,
                    time_accounted_by_inner_nodes,
                )?;
                *time_accounted_by_inner_nodes += start.elapsed();
            }

            // Update parent nodes if any (childs timings are deducted from parents)
            let prefix_vec = prefix.to_vec();
            if !prefix_vec.is_empty() {
                (1..prefix_vec.len() + 1).map(|idx| prefix_vec[..idx].to_vec()).for_each(
                    |parent_path| {
                        let parent_node = parent_path.last().map(|it| it.0).unwrap();
                        let parent = dg
                            .node_mut(NodeQId(
                                parent_path[..parent_path.len() - 1].into(),
                                parent_node,
                            ))
                            .profile
                            .get_or_insert(Duration::default());
                        *parent -= elapsed.min(*parent);
                    },
                );
            }
            res
        },
    )?;
    Ok(r)
}

fn profile_submodel(
    node: &TypedNode,
    mut node_state: Option<&mut dyn OpState>,
    input: TVec<TValue>,
    dg: &mut Annotations,
    profilers: Option<&HashMap<TypeId, Profiler>>,
    prefix: &[(usize, String)],
    time_accounted_by_inner_nodes: &mut Duration,
) -> TractResult<()> {
    if let Some(ref mut op_state) = node_state {
        if let Some(profiler) = profilers.and_then(|it| it.get(&op_state.type_id())) {
            let mut new_prefix: TVec<_> = prefix.into();
            new_prefix.push((node.id, "submodel".to_string()));

            let (_, _) =
                (profiler.func)(*op_state, input, dg, &new_prefix, time_accounted_by_inner_nodes)?;
        } else if let Some(scan_state) = op_state.downcast_mut::<State>() {
            let mut new_prefix: TVec<_> = prefix.into();
            new_prefix.push((node.id, "loop".to_string()));

            let scan_inputs = make_inputs_for_model(scan_state.model_state.model())?;
            let multi = scan_state.iteration_count(&input);

            rec_profiler(
                &mut scan_state.model_state,
                dg,
                &scan_inputs,
                None,
                &new_prefix,
                Some(multi),
                time_accounted_by_inner_nodes,
                false,
            )?;
        } else if let Some(typed_model_state) = op_state.downcast_mut::<TypedModelOpState>() {
            let mut new_prefix: TVec<_> = prefix.into();
            new_prefix.push((node.id, "submodel".to_string()));

            rec_profiler(
                typed_model_state,
                dg,
                &input,
                None,
                &new_prefix,
                None,
                time_accounted_by_inner_nodes,
                false,
            )?;
        }
    }

    Ok(())
}

type ProfilerFn = fn(
    &mut dyn OpState,
    TVec<TValue>,
    &mut Annotations,
    &[(usize, String)],
    &mut Duration,
) -> TractResult<(TractResult<TVec<TValue>>, Duration)>;

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

pub fn extract_costs(
    annotations: &mut Annotations,
    model: &dyn Model,
    extra_symbols: &SymbolValues,
) -> TractResult<()> {
    fn extract_costs_rec(
        annotations: &mut Annotations,
        model: &dyn Model,
        prefix: &[(usize, String)],
        multiplier: TDim,
        extra_symbols: &SymbolValues,
    ) -> TractResult<()> {
        if let Some(model) = model.downcast_ref::<TypedModel>() {
            for node_id in 0..model.nodes().len() {
                let inputs = model.node_input_facts(node_id)?;
                let cost = model
                    .node(node_id)
                    .op
                    .cost(&inputs)
                    .with_context(|| format!("costing node {}", model.node(node_id)))?;
                annotations.node_mut(NodeQId(prefix.into(), node_id)).cost = cost
                    .into_iter()
                    .map(|(k, v)| {
                        let cost = if k.is_compute() { v * &multiplier } else { v };
                        (k, cost.eval(extra_symbols))
                    })
                    .collect();

                let nested_subs = model.nested_models(node_id);
                let nested_multis = (model as &dyn Model).nested_models_iters(node_id, &inputs);
                for (name, sub) in nested_subs {
                    let mut prefix: TVec<_> = prefix.into();
                    prefix.push((node_id, name.to_string()));
                    extract_costs_rec(
                        annotations,
                        sub,
                        &prefix,
                        nested_multis.clone().unwrap_or_else(|| 1.into()) * &multiplier,
                        extra_symbols,
                    )?;
                }
            }
        }
        Ok(())
    }
    extract_costs_rec(annotations, model, &[], 1.into(), extra_symbols)
}
