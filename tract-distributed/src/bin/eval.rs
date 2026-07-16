//! Run the whole CPU model with a per-node hook and report the first node whose
//! output is not finite. Unlike `distract-trace` this does not touch the graph,
//! so it observes exactly what the optimizer produced.
//!
//!   distract-eval <model.nnef.tgz> <comma,separated,prompt,ids>

use tract_core::prelude::*;
use tract_distributed::llm::{empty_inputs, load_model};

fn scan(t: &TValue) -> Option<(f32, usize)> {
    if !t.datum_type().is_float() || !t.is_plain() {
        return None;
    }
    let f = t.cast_to::<f32>().ok()?;
    let v = f.view();
    let s = v.as_slice::<f32>().ok()?;
    let bad = s.iter().filter(|x| !x.is_finite()).count();
    let mx = s.iter().filter(|x| x.is_finite()).fold(0f32, |a, b| a.max(b.abs()));
    Some((mx, bad))
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let path = std::env::args().nth(1).expect("model");
    let prompt: Vec<i64> = std::env::args()
        .nth(2)
        .expect("prompt ids")
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();

    let (full, _n) = load_model(&path)?;
    let inputs = empty_inputs(&full, &prompt)?;

    let opt = full.into_optimized()?;
    let names: Vec<String> = opt
        .nodes()
        .iter()
        .map(|n| {
            let ins: Vec<String> = n
                .inputs
                .iter()
                .map(|i| format!("{}[{}]", opt.node(i.node).name, opt.node(i.node).op().name()))
                .collect();
            ins.join("  <-  ")
        })
        .collect();
    let plan = opt.into_runnable()?;
    let mut state = tract_core::plan::SimpleState::new(&plan)?;

    let mut first = true;
    state.run_plan_with_eval(inputs, |session, op_state, node, input| {
        let ins: Vec<String> = input
            .iter()
            .map(|i| match scan(i) {
                Some((m, b)) => format!("max|x|={m:.1} nf={b}"),
                None => "?".into(),
            })
            .collect();
        let r = tract_core::plan::eval(session, op_state, node, input)?;
        if std::env::var("DISTRACT_VERBOSE").is_ok() {
            for o in r.iter() {
                if let Some((mx, bad)) = scan(o)
                    && (mx > 1000.0
                        || bad > 0
                        || node.name.contains("slice")
                        || node.name.contains("pack_b"))
                {
                    println!(
                        "  {:<62} {:<14} max {:>9.1} nf {bad}",
                        node.name,
                        node.op().name(),
                        mx
                    );
                }
            }
        }
        if first {
            for o in r.iter() {
                if let Some((mx, bad)) = scan(o)
                    && bad > 0
                {
                    println!("\n>>> FIRST NON-FINITE <<<");
                    println!("    node : {}", node.name);
                    println!("    op   : {}", node.op().name());
                    println!("    out  : max|x|={mx:.1} non_finite={bad}");
                    println!("    from : {}", names[node.id]);
                    for (ix, s) in ins.iter().enumerate() {
                        println!("    in {ix}: {s}");
                    }
                    first = false;
                }
            }
        }
        Ok(r) as TractResult<TVec<TValue>>
    })?;
    if first {
        println!("no non-finite value produced by any node");
    }
    Ok(())
}
