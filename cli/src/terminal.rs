use std::time::Duration;

use crate::annotations::*;
use crate::display_params::*;
use crate::draw::DrawingState;
use crate::CliResult;
use ansi_term::Color::*;
#[allow(unused_imports)]
use std::convert::TryFrom;
use tract_core::internal::*;
use tract_core::itertools::Itertools;

pub fn render(
    model: &dyn Model,
    annotations: &Annotations,
    options: &DisplayParams,
) -> CliResult<()> {
    render_prefixed(model, "", &[], annotations, options)
}

pub fn render_node(
    model: &dyn Model,
    node_id: usize,
    annotations: &Annotations,
    options: &DisplayParams,
) -> CliResult<()> {
    render_node_prefixed(model, "", &[], node_id, None, annotations, options)
}

fn render_prefixed(
    model: &dyn Model,
    prefix: &str,
    scope: &[(usize, String)],
    annotations: &Annotations,
    options: &DisplayParams,
) -> CliResult<()> {
    let mut drawing_state =
        if options.should_draw() { Some(DrawingState::default()) } else { None };
    let node_ids = if options.natural_order {
        (0..model.nodes_len()).collect()
    } else {
        model.eval_order()?
    };
    for node in node_ids {
        if options.filter(model, scope, node)? {
            render_node_prefixed(
                model,
                prefix,
                scope,
                node,
                drawing_state.as_mut(),
                annotations,
                options,
            )?
        } else if let Some(ref mut ds) = drawing_state {
            let _prefix = ds.draw_node_vprefix(model, node, &options)?;
            let _body = ds.draw_node_body(model, node, &options)?;
            let _suffix = ds.draw_node_vsuffix(model, node, &options)?;
        }
    }
    Ok(())
}

fn render_node_prefixed(
    model: &dyn Model,
    prefix: &str,
    scope: &[(usize, String)],
    node_id: usize,
    mut drawing_state: Option<&mut DrawingState>,
    annotations: &Annotations,
    options: &DisplayParams,
) -> CliResult<()> {
    let qid = NodeQId(scope.into(), node_id);
    let tags = annotations.tags.get(&qid).cloned().unwrap_or_default();
    let name_color = tags.style.clone().unwrap_or(White.into());
    let node_name = model.node_name(node_id);
    let node_op_name = model.node_op(node_id).name();
    let cost_column_pad = format!("{:>1$}", "", options.cost as usize * 25);
    let profile_column_pad = format!("{:>1$}", "", options.profile as usize * 20);

    if let Some(ref mut ds) = &mut drawing_state {
        for l in ds.draw_node_vprefix(model, node_id, &options)? {
            println!("{}{}{}{} ", cost_column_pad, profile_column_pad, prefix, l);
        }
    }

    // cost column
    let mut cost_column = if options.cost {
        Some(
            tags.cost.iter().map(|c| format!("{:1$}", format!("{:?}:{}", c.0, c.1), 25)).peekable(),
        )
    } else {
        None
    };

    // profile column
    let mut profile_column = tags.profile.map(|measure| {
        let profile_summary = annotations.profile_summary.as_ref().unwrap();
        let ratio = measure.as_secs_f64() / profile_summary.sum.as_secs_f64();
        let ratio_for_color = measure.as_secs_f64() / profile_summary.max.as_secs_f64();
        let color = colorous::RED_YELLOW_GREEN.eval_continuous(1.0 - ratio_for_color);
        let color = ansi_term::Color::RGB(color.r, color.g, color.b);
        let label = format!(
            "{:7.3} ms/i {}  ",
            measure.as_secs_f64() * 1e3,
            color.bold().paint(format!("{:>4.1}%", ratio * 100.0))
        );
        std::iter::once(label)
    });

    // drawing column
    let mut drawing_lines: Box<dyn Iterator<Item = String>> =
        if let Some(ds) = drawing_state.as_mut() {
            let body = ds.draw_node_body(model, node_id, options)?;
            let suffix = ds.draw_node_vsuffix(model, node_id, options)?;
            let filler = ds.draw_node_vfiller(model, node_id)?;
            Box::new(body.into_iter().chain(suffix.into_iter()).chain(std::iter::repeat(filler)))
        } else {
            Box::new(std::iter::repeat(cost_column_pad.clone()))
        };

    macro_rules! prefix {
        () => {
            let cost = cost_column
                .as_mut()
                .map(|it| it.next().unwrap_or_else(|| cost_column_pad.to_string()))
                .unwrap_or("".to_string());
            let profile = profile_column
                .as_mut()
                .map(|it| it.next().unwrap_or_else(|| profile_column_pad.to_string()))
                .unwrap_or("".to_string());
            print!("{}{}{}{} ", cost, profile, prefix, drawing_lines.next().unwrap(),)
        };
    };

    prefix!();
    println!(
        "{} {} {}",
        White.bold().paint(format!("{}", node_id)),
        (if node_name == "UnimplementedOp" {
            Red.bold()
        } else {
            if options.expect_canonic && !model.node_op(node_id).is_canonic() {
                Yellow.bold()
            } else {
                Blue.bold()
            }
        })
        .paint(node_op_name),
        name_color.italic().paint(node_name)
    );
    for label in tags.labels.iter() {
        prefix!();
        println!("  * {}", label);
    }
    match options.io {
        Io::Long => {
            for (ix, i) in model.node_inputs(node_id).iter().enumerate() {
                let star = if ix == 0 { '*' } else { ' ' };
                prefix!();
                println!(
                    "  {} input fact  #{}: {} {}",
                    star,
                    ix,
                    White.bold().paint(format!("{:?}", i)),
                    model.outlet_fact_format(*i),
                );
            }
            for ix in 0..model.node_output_count(node_id) {
                let star = if ix == 0 { '*' } else { ' ' };
                let io = if let Some(id) =
                    model.input_outlets().iter().position(|n| n.node == node_id && n.slot == ix)
                {
                    format!(
                        "{} {}",
                        Cyan.bold().paint(format!("MODEL INPUT #{}", id)).to_string(),
                        tags.model_input.as_ref().map(|s| &**s).unwrap_or("")
                    )
                } else if let Some(id) =
                    model.output_outlets().iter().position(|n| n.node == node_id && n.slot == ix)
                {
                    format!(
                        "{} {}",
                        Yellow.bold().paint(format!("MODEL OUTPUT #{}", id)).to_string(),
                        tags.model_output.as_ref().map(|s| &**s).unwrap_or("")
                    )
                } else {
                    "".to_string()
                };
                let outlet = OutletId::new(node_id, ix);
                let successors = model.outlet_successors(outlet);
                prefix!();
                println!(
                    "  {} output fact #{}: {} {} {}",
                    star,
                    ix,
                    model.outlet_fact_format(outlet),
                    White.bold().paint(successors.iter().map(|s| format!("{:?}", s)).join(" ")),
                    io
                );
                if options.outlet_labels {
                    if let Some(label) = model.outlet_label(OutletId::new(node_id, ix)) {
                        prefix!();
                        println!("            {} ", White.italic().paint(label));
                    }
                }
            }
        }
        Io::Short => {
            let same = model.node_inputs(node_id).len() > 0
                && model.node_output_count(node_id) == 1
                && model.outlet_fact_format(node_id.into())
                    == model.outlet_fact_format(model.node_inputs(node_id)[0]);
            if !same {
                let style = drawing_state
                    .and_then(|w| w.wires.last())
                    .and_then(|w| w.color)
                    .unwrap_or(White.into());
                for ix in 0..model.node_output_count(node_id) {
                    prefix!();
                    println!(
                        "  {}{}{} {}",
                        style.paint(box_drawing::heavy::HORIZONTAL),
                        style.paint(box_drawing::heavy::HORIZONTAL),
                        style.paint(box_drawing::heavy::HORIZONTAL),
                        model.outlet_fact_format((node_id, ix).into())
                    );
                }
            }
        }
        Io::None => (),
    }
    if options.info {
        for info in model.node_op(node_id).info()? {
            prefix!();
            println!("  * {}", info);
        }
    }
    if options.invariants {
        if let Some(typed) = model.downcast_ref::<TypedModel>() {
            let node = typed.node(node_id);
            prefix!();
            println!("  * {:?}", node.op().as_typed().unwrap().invariants(&typed, &node)?);
        }
    }
    if options.debug_op {
        prefix!();
        println!("  * {:?}", model.node_op(node_id));
    }
    for section in tags.sections {
        if section.is_empty() {
            continue;
        }
        prefix!();
        println!("  * {}", section[0]);
        for s in &section[1..] {
            prefix!();
            println!("    {}", s);
        }
    }
    if let Some(tmodel) = model.downcast_ref::<TypedModel>() {
        for (label, sub, _, _) in tmodel.node(node_id).op.nested_models() {
            let prefix = drawing_lines.next().unwrap();
            let mut scope: TVec<_> = scope.into();
            scope.push((node_id, label.to_string()));
            render_prefixed(
                sub,
                &format!("{} [{}] ", prefix, label),
                &*scope,
                annotations,
                options,
            )?
        }
    }
    while cost_column.as_mut().map(|cost| cost.peek().is_some()).unwrap_or(false) {
        prefix!();
        println!("");
    }
    Ok(())
}

pub fn render_summaries(
    model: &dyn Model,
    annotations: &Annotations,
    options: &DisplayParams,
) -> CliResult<()> {
    let total = annotations.tags.values().sum::<NodeTags>();

    if options.cost {
        println!("{}", White.bold().paint("Cost summary"));
        for (c, i) in &total.cost {
            println!(" * {:?}: {}", c, i);
        }
    }

    if options.profile {
        let summary = annotations.profile_summary.as_ref().unwrap();
        println!("{}", White.bold().paint("Most time consuming operations"));
        for (op, (dur, n)) in annotations
            .tags
            .iter()
            .map(|(k, v)| {
                (
                    k.model(model).unwrap().node_op(k.1).name(),
                    v.profile.unwrap_or(Duration::default()),
                )
            })
            .sorted_by_key(|a| a.0.to_string())
            .group_by(|(n, _)| n.clone())
            .into_iter()
            .map(|(a, group)| {
                (
                    a,
                    group
                        .into_iter()
                        .fold((Duration::default(), 0), |acc, d| (acc.0 + d.1, acc.1 + 1)),
                )
            })
            .into_iter()
            .sorted_by_key(|(_, d)| d.0)
            .rev()
        {
            println!(
                " * {} {:3} nodes: {}",
                Blue.bold().paint(format!("{:20}", op)),
                n,
                dur_avg_ratio(dur, summary.sum)
            );
        }
        println!(
            "Not accounted by ops: {}",
            dur_avg_ratio(summary.entire - summary.sum.min(summary.entire), summary.entire)
        );
        println!("Entire network performance: {}", dur_avg(summary.entire));
    }

    Ok(())
}

/// Format a rusage::Duration showing avgtime in ms.
pub fn dur_avg(measure: Duration) -> String {
    White.bold().paint(format!("{:.3} ms/i", measure.as_secs_f64() * 1e3)).to_string()
}

/// Format a rusage::Duration showing avgtime in ms, with percentage to a global
/// one.
pub fn dur_avg_ratio(measure: Duration, global: Duration) -> String {
    format!(
        "{} {}",
        White.bold().paint(format!("{:7.3} ms/i", measure.as_secs_f64() * 1e3)),
        Yellow
            .bold()
            .paint(format!("{:>4.1}%", measure.as_secs_f64() / global.as_secs_f64() * 100.)),
    )
}
