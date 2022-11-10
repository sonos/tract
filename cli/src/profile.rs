use std::time::{Duration, Instant};
use tract_core::internal::*;
use tract_libcli::annotations::*;
use tract_libcli::model::Model;
use tract_libcli::profile::*;
use tract_libcli::tensor::RunParams;
use tract_libcli::tensor::{make_inputs_for_model, retrieve_or_make_inputs};

pub fn profile(
    model: &TypedModel,
    bench_limits: &BenchLimits,
    dg: &mut Annotations,
    run_params: &RunParams,
) -> TractResult<()> {
    info!("Running entire network");
    let plan = SimplePlan::new(model)?;
    let mut state = SimpleState::new(&plan)?;
    let mut iters = 0usize;
    let start = Instant::now();
    while iters < bench_limits.max_iters && start.elapsed() < bench_limits.max_time {
        let input = retrieve_or_make_inputs(model, run_params)?;
        let _ =
            state.run_plan_with_eval(input[0].clone(), |session_state, state, node, input| {
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
                let multi = multiplier.as_ref().unwrap().to_isize().unwrap();
                let prefix = tvec!((outer_node.id, inner_model_name.to_string()));
                if let Some(inner_model) = inner_model.downcast_ref::<TypedModel>() {
                    for _ in 0..iters {
                        let inner_plan = SimplePlan::new(inner_model)?;
                        let mut state = SimpleState::new(inner_plan)?;
                        let _ = state.run_plan_with_eval(
<<<<<<< HEAD
                            make_inputs_for_model(inner_model)?,
=======
                            crate::tensor::make_inputs_for_model(inner_model)?
                                .iter()
                                .map(|t| t.clone().into_tvalue())
                                .collect(),
>>>>>>> ef1567973 (first pass, eval())
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
