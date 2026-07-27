//! What actually crosses the machine boundary per token, and how big it is.

use anyhow::Result;

use tract_core::prelude::OutletId;
use tract_distributed::llm::{load_model, partition_stages};
use tract_distributed::protocol::Role;

/// Exactly one tensor may cross a boundary: the residual. Anything else is either state
/// that belongs to a stage (the KV) or a table the receiving stage can rebuild — and a
/// table that is `S+P` wide costs more every token as the context grows, so letting one
/// back onto the wire would regress quietly and only at long context.
#[test]
#[ignore]
fn only_the_residual_crosses() -> Result<()> {
    let path = std::env::var("DISTRACT_MODEL").expect("set DISTRACT_MODEL");
    let (full, n_regular) = load_model(&path)?;
    for cut in [1usize, 14, 27] {
        let stages = partition_stages(&full, &[cut], n_regular)?;
        for (i, st) in stages.iter().enumerate() {
            let wire = st.outputs.iter().filter(|s| s.role == Role::Wire).count();
            assert_eq!(wire, 1, "cut at {cut}: stage {i} sends {wire} tensors, expected 1");
        }
    }
    Ok(())
}

#[test]
#[ignore]
fn wire_payload_per_step() -> Result<()> {
    let path = std::env::var("DISTRACT_MODEL").expect("set DISTRACT_MODEL");
    let (full, n_regular) = load_model(&path)?;
    let stages = partition_stages(&full, &[14], n_regular)?;
    for (i, st) in stages.iter().enumerate() {
        let wire_out: Vec<_> =
            st.outputs.iter().enumerate().filter(|(_, s)| s.role == Role::Wire).collect();
        let cache_out = st.outputs.len() - wire_out.len();
        println!(
            "stage {i}: {} outputs = {} wire + {} cache",
            st.outputs.len(),
            wire_out.len(),
            cache_out
        );
        for (slot, _) in &wire_out {
            let o = st.model.output_outlets()?[*slot];
            let f = st.model.outlet_fact(o)?;
            let name = st
                .model
                .outlet_labels
                .get(&o)
                .cloned()
                .unwrap_or_else(|| st.model.node(o.node).name.clone());
            println!("    wire[{slot}] {name}: {:?} {:?}", f.datum_type, f.shape);
        }
    }
    Ok(())
}

/// Placement segments by residual; the planner attributes weights by cache depth. If the
/// two disagree a shard holds weights the planner charged to its neighbour, and the fit
/// check guards the wrong number.
#[test]
#[ignore]
fn weight_distribution_matches_the_plan() -> Result<()> {
    use tract_distributed::partition::const_bytes;
    use tract_distributed::plan::{layer_weight_profile, stage_weights};
    let path = std::env::var("DISTRACT_MODEL").expect("set DISTRACT_MODEL");
    let (full, n_regular) = load_model(&path)?;
    let n_layers = full
        .input_outlets()?
        .iter()
        .filter(|o| full.node(o.node).name.contains("cache_key"))
        .count();
    println!("n_layers = {n_layers}");
    let cuts = [14usize];

    let profile = layer_weight_profile(&full, n_layers);
    let predicted = stage_weights(&profile, &cuts);
    let stages = partition_stages(&full, &cuts, n_regular)?;
    let actual: Vec<u64> = stages.iter().map(|s| const_bytes(&s.model) as u64).collect();

    for (i, (p, a)) in predicted.iter().zip(&actual).enumerate() {
        let drift = (*a as i64 - *p as i64) as f64 / *p as f64 * 100.0;
        println!(
            "stage {i}: planner {} MiB, shard {} MiB ({drift:+.1}%)",
            p / (1 << 20),
            a / (1 << 20)
        );
    }
    for (i, (p, a)) in predicted.iter().zip(&actual).enumerate() {
        let drift = (*a as f64 - *p as f64).abs() / *p as f64;
        assert!(drift < 0.02, "stage {i} holds {a} bytes, planner predicted {p}");
    }
    Ok(())
}

/// What the shared tables depend on. Reporting no sources is what says a stage can rebuild
/// one for itself rather than be sent it: the cone is constants and arithmetic over the
/// sequence dimensions, which every stage has.
#[test]
#[ignore]
fn what_the_crossing_tensors_depend_on() -> Result<()> {
    let path = std::env::var("DISTRACT_MODEL").expect("set DISTRACT_MODEL");
    let (full, _n_regular) = load_model(&path)?;

    let names = [
        "model_model_rotaryEmb_cosTo0",
        "model_model_rotaryEmb_sinTo0",
        "model_model_unsqueeze1.1",
    ];
    for want in names {
        let Some(node) = full.nodes().iter().find(|n| {
            full.outlet_labels.get(&OutletId::new(n.id, 0)).map(|s| s.as_str()) == Some(want)
                || n.name == want
        }) else {
            println!("{want}: not found");
            continue;
        };
        // Walk its cone back to the sources.
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![node.id];
        let mut sources = vec![];
        let mut ops = std::collections::BTreeMap::new();
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            let nd = full.node(n);
            *ops.entry(nd.op().name().to_string()).or_insert(0) += 1;
            if nd.inputs.is_empty() && !nd.op_is::<tract_core::ops::konst::Const>() {
                sources.push(nd.name.clone());
            }
            stack.extend(nd.inputs.iter().map(|i| i.node));
        }
        println!("{want}: {} nodes in cone", seen.len());
        println!("   sources: {sources:?}");
        let shapey: usize = ops
            .iter()
            .filter(|(k, _)| {
                k.contains("Shape")
                    || k.contains("Dim")
                    || k.contains("Range")
                    || k.contains("Cast")
            })
            .map(|(_, v)| *v)
            .sum();
        println!("   ops: {ops:?}");
        println!("   shape-ish ops: {shapey}");
    }
    Ok(())
}
