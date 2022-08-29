use std::time::Duration;

use crate::annotations::*;
use crate::display_params::*;
use crate::draw::DrawingState;
use crate::model::Model;
use crate::CliResult;
use ansi_term::ANSIString;
use ansi_term::Color::*;
#[allow(unused_imports)]
use std::convert::TryFrom;
use tract_core::internal::*;
use tract_itertools::Itertools;

pub fn render(
    model: &dyn Model,
    annotations: &Annotations,
    options: &DisplayParams,
) -> CliResult<()> {
    if options.quiet {
        return Ok(());
    }
    render_prefixed(model, "", &[], annotations, options)?;
    if model.properties().len() > 0 {
        println!("{}", White.bold().paint("# Properties"));
    }
    for (k, v) in model.properties().iter().sorted_by_key(|(k, _)| k.to_string()) {
        println!("* {}: {:?}", White.paint(k), v)
    }
    Ok(())
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
            let _prefix = ds.draw_node_vprefix(model, node, options)?;
            let _body = ds.draw_node_body(model, node, options)?;
            let _suffix = ds.draw_node_vsuffix(model, node, options)?;
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
    let name_color = tags.style.unwrap_or_else(|| White.into());
    let node_name = model.node_name(node_id);
    let node_op_name = model.node_op(node_id).name();
    let profile_column_pad = format!("{:>1$}", "", options.profile as usize * 20);
    let cost_column_pad = format!("{:>1$}", "", options.cost as usize * 25);
    let flops_column_pad = format!("{:>1$}", "", (options.profile && options.cost) as usize * 20);

    if let Some(ref mut ds) = &mut drawing_state {
        for l in ds.draw_node_vprefix(model, node_id, options)? {
            println!(
                "{}{}{}{}{} ",
                cost_column_pad, profile_column_pad, flops_column_pad, prefix, l
            );
        }
    }

    // profile column
    let mut profile_column = tags.profile.map(|measure| {
        let profile_summary = annotations.profile_summary.as_ref().unwrap();
        let use_micros = profile_summary.sum < Duration::from_millis(1);
        let ratio = measure.as_secs_f64() / profile_summary.sum.as_secs_f64();
        let ratio_for_color = measure.as_secs_f64() / profile_summary.max.as_secs_f64();
        let color = colorous::RED_YELLOW_GREEN.eval_continuous(1.0 - ratio_for_color);
        let color = ansi_term::Color::RGB(color.r, color.g, color.b);
        let label = format!(
            "{:7.3} {}s/i {}  ",
            measure.as_secs_f64() * if use_micros { 1e6 } else { 1e3 },
            if use_micros { "µ" } else { "m" },
            color.bold().paint(format!("{:>4.1}%", ratio * 100.0))
        );
        std::iter::once(label)
    });

    // cost column
    let mut cost_column = if options.cost {
        Some(
            tags.cost
                .iter()
                .map(|c| {
                    let key = format!("{:?}", c.0);
                    let value = render_tdim(&c.1);
                    let value_visible_len = c.1.to_string().len();
                    let padding = 24usize.saturating_sub(value_visible_len + key.len());
                    key + &*std::iter::repeat(' ').take(padding).join("") + &value + " "
                })
                .peekable(),
        )
    } else {
        None
    };

    // flops column
    let mut flops_column = if options.profile && options.cost {
        let timing: f64 = tags.profile.as_ref().unwrap().as_secs_f64();
        let flops_column_pad = flops_column_pad.clone();
        let it = tags.cost.iter().map(move |c| {
            if c.0.is_compute() {
                let flops = c.1.to_usize().unwrap_or(0) as f64 / timing;
                let unpadded = if flops > 1e9 {
                    format!("{:.3} GF/s", flops / 1e9)
                } else if flops > 1e6 {
                    format!("{:.3} MF/s", flops / 1e6)
                } else if flops > 1e3 {
                    format!("{:.3} kF/s", flops / 1e3)
                } else {
                    format!("{:.3}  F/s", flops)
                };
                format!("{:>1$} ", unpadded, 19)
            } else {
                flops_column_pad.clone()
            }
        });
        Some(it)
    } else {
        None
    };

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
            let flops = flops_column
                .as_mut()
                .map(|it| it.next().unwrap_or_else(|| flops_column_pad.to_string()))
                .unwrap_or("".to_string());
            print!("{}{}{}{}{} ", profile, cost, flops, prefix, drawing_lines.next().unwrap(),)
        };
    }

    prefix!();
    println!(
        "{} {} {}",
        White.bold().paint(format!("{}", node_id)),
        (if node_name == "UnimplementedOp" {
            Red.bold()
        } else if options.expect_core && !model.node_op(node_id).op_families().contains(&"core") {
            Yellow.bold()
        } else {
            Blue.bold()
        })
        .paint(node_op_name),
        name_color.italic().paint(node_name)
    );
    for label in tags.labels.iter() {
        prefix!();
        println!("  * {}", label);
    }
    if let Io::Long = options.io {
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
        for slot in 0..model.node_output_count(node_id) {
            let star = if slot == 0 { '*' } else { ' ' };
            let outlet = OutletId::new(node_id, slot);
            let mut model_io = vec![];
            for (ix, _) in model.input_outlets().iter().enumerate().filter(|(_, o)| **o == outlet) {
                model_io.push(Cyan.bold().paint(format!("MODEL INPUT #{}", ix)).to_string());
            }
            if let Some(t) = &tags.model_input {
                model_io.push(t.to_string());
            }
            for (ix, _) in model.output_outlets().iter().enumerate().filter(|(_, o)| **o == outlet)
            {
                model_io.push(Yellow.bold().paint(format!("MODEL OUTPUT #{}", ix)).to_string());
            }
            if let Some(t) = &tags.model_output {
                model_io.push(t.to_string());
            }
            let successors = model.outlet_successors(outlet);
            prefix!();
            println!(
                "  {} output fact #{}: {} {} {} {}",
                star,
                slot,
                model.outlet_fact_format(outlet),
                White.bold().paint(successors.iter().map(|s| format!("{:?}", s)).join(" ")),
                model_io.join(", "),
                Blue.bold().italic().paint(
                    tags.outlet_labels
                        .get(slot)
                        .map(|s| s.join(","))
                        .unwrap_or_else(|| "".to_string())
                )
            );
            if options.outlet_labels {
                if let Some(label) = model.outlet_label(OutletId::new(node_id, slot)) {
                    prefix!();
                    println!("            {} ", White.italic().paint(label));
                }
            }
        }
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
            let (inputs, outputs) = typed.node_facts(node.id)?;
            let invariants = node.op().as_typed().unwrap().invariants(&inputs, &outputs)?;
            prefix!();
            println!("  * {:?}", invariants);
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
    for (label, sub) in model.nested_models(node_id) {
        let prefix = drawing_lines.next().unwrap();
        let mut scope: TVec<_> = scope.into();
        scope.push((node_id, label.to_string()));
        render_prefixed(sub, &format!("{} [{}] ", prefix, label), &*scope, annotations, options)?
    }

    if let Io::Short = options.io {
        let same = model.node_inputs(node_id).len() > 0
            && model.node_output_count(node_id) == 1
            && model.outlet_fact_format(node_id.into())
                == model.outlet_fact_format(model.node_inputs(node_id)[0]);
        if !same || model.output_outlets().iter().any(|o| o.node == node_id) {
            let style = drawing_state
                .map(|s| s.wires.last().and_then(|w| w.color).unwrap_or(s.latest_node_color))
                .unwrap_or_else(|| White.into());
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

    while cost_column.as_mut().map(|cost| cost.peek().is_some()).unwrap_or(false) {
        prefix!();
        println!();
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
            println!(" * {:?}: {}", c, render_tdim(i));
        }
    }

    if options.profile {
        let summary = annotations.profile_summary.as_ref().unwrap();

        println!("{}", White.bold().paint("Most time consuming operations"));
        for (op, (dur, n)) in annotations
            .tags
            .iter()
            .map(|(k, v)| {
                (k.model(model).unwrap().node_op(k.1).name(), v.profile.unwrap_or_default())
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

        println!("{}", White.bold().paint("By prefix"));
        fn prefixes_for(s: &str) -> impl Iterator<Item = String> + '_ {
            use tract_itertools::*;
            let split = s.split('.').count();
            (0..split).map(move |n| s.split('.').take(n).join("."))
        }
        let all_prefixes = annotations
            .tags
            .keys()
            .flat_map(|id| prefixes_for(id.model(model).unwrap().node_name(id.1)))
            .filter(|s| s.len() > 0)
            .sorted()
            .unique()
            .collect::<Vec<String>>();
        for prefix in &all_prefixes {
            let sum = annotations
                .tags
                .iter()
                .filter(|(k, _v)| k.model(model).unwrap().node_name(k.1).starts_with(prefix))
                .map(|(_k, v)| v)
                .sum::<NodeTags>();
            if sum.profile.unwrap_or_default().as_secs_f64() / summary.entire.as_secs_f64() < 0.01 {
                continue;
            }
            print!("{}    ", dur_avg_ratio(sum.profile.unwrap_or_default(), summary.sum));
            for _ in prefix.chars().filter(|c| *c == '.') {
                print!("   ");
            }
            println!("{}", prefix);
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

fn render_tdim(d: &TDim) -> ANSIString<'static> {
    if let Ok(i) = d.to_i64() {
        render_big_integer(i)
    } else {
        d.to_string().into()
    }
}

fn render_big_integer(i: i64) -> ansi_term::ANSIString<'static> {
    let raw = i.to_string();
    let mut blocks = raw
        .chars()
        .rev()
        .chunks(3)
        .into_iter()
        .map(|mut c| c.join("").chars().rev().join(""))
        .enumerate()
        .map(|(ix, s)| if ix % 2 == 1 { White.bold().paint(s).to_string() } else { s })
        .collect::<Vec<_>>();
    blocks.reverse();
    blocks.into_iter().join("").into()
}
