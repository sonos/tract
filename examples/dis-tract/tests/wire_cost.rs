//! What actually crosses the machine boundary per token, and how big it is.

use anyhow::Result;

use tract_distributed::llm::{load_model, partition_stages};
use tract_distributed::protocol::Role;

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
