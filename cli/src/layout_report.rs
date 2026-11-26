//! `tract <model> layout-report`: where a decluttered model could switch the layout of its
//! convolutions, and what the switched graph would need.
//!
//! From every convolution not yet claimed by a region, the channel axis is moved to the other end
//! (channel-last for a channel-first conv, channel-first for a channel-last one) and the change is
//! flooded through the graph with each op's own `change_axes`. Unlike the `ChangeAxes` pass, the
//! flood does not give up when an op refuses: the refusing edge becomes a region boundary, where a
//! transpose would sit, and the op is reported as a blocker. Model interfaces are boundaries too.
//!
//! For each region the report lists its convolutions with the shape figures that decide a layout
//! (kind, kernel, stride, channels, fan-out, multiply-adds, bytes an im2col would gather), its
//! boundary transposes with their bytes, and the ops that would run in the switched layout. Across
//! regions it sums the blockers and the switched ops. Nothing is changed and nothing is run.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::Serialize;
use tract_core::internal::*;
use tract_core::ops::array::{Slice, TypedConcat};
use tract_core::ops::change_axes::{AxisChangeConsequence, InOut};
use tract_core::ops::cnn::Conv;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::konst::Const;
use tract_core::ops::source::TypedSource;

use crate::params::Parameters;

pub fn command() -> clap::Command {
    clap::Command::new("layout-report")
        .about("Report the regions of a decluttered model that could switch convolution layout, what blocks them, and what they would need")
        .arg(clap::Arg::new("json").long("json").action(clap::ArgAction::SetTrue).help("Emit JSON"))
        .arg(clap::Arg::new("allow-random-input").long("allow-random-input").action(clap::ArgAction::SetTrue).hide(true))
        .arg(clap::Arg::new("allow-float-casts").long("allow-float-casts").action(clap::ArgAction::SetTrue).hide(true))
}

#[derive(Serialize, Default)]
struct Report {
    total_macs: u64,
    conv_macs: u64,
    regions: Vec<Region>,
    blockers: BTreeMap<String, BlockerSum>,
    switched_ops: BTreeMap<String, OpSum>,
    unclaimed_convs: Vec<ConvInfo>,
}

#[derive(Serialize, Default)]
struct BlockerSum {
    count: usize,
    bytes: u64,
    nodes: Vec<String>,
}

#[derive(Serialize, Default)]
struct OpSum {
    count: usize,
    macs: u64,
}

#[derive(Serialize)]
struct Region {
    direction: String,
    nodes: usize,
    macs: u64,
    convs: Vec<ConvInfo>,
    einsums: Vec<EinSumInfo>,
    boundaries: Vec<Boundary>,
    boundary_bytes: u64,
    channel_concats_and_slices: Vec<String>,
}

#[derive(Serialize, Clone)]
struct ConvInfo {
    name: String,
    format: String,
    kind: String,
    kernel: Vec<usize>,
    stride: Vec<usize>,
    dilated: bool,
    cin: usize,
    cout: usize,
    group: usize,
    fan_out: usize,
    points: Option<u64>,
    macs: u64,
    input_bytes: Option<u64>,
    gathered_bytes_channel_first: Option<u64>,
    dt: String,
}

#[derive(Serialize)]
struct EinSumInfo {
    name: String,
    expr: String,
    macs: u64,
    contraction_innermost_after: bool,
}

#[derive(Serialize, Clone)]
struct Boundary {
    wire: String,
    reason: String,
    op: String,
    bytes: Option<u64>,
}

/// An op that took the change: its substitute and the wire changes it asked for.
type Accepted = (Box<dyn TypedOp>, TVec<(InOut, AxisOp)>);

struct Flood {
    wires: HashMap<OutletId, AxisOp>,
    accepted: HashMap<usize, Accepted>,
    boundaries: Vec<(OutletId, usize, String, String)>,
}

fn macs(model: &TypedModel, node: &TypedNode) -> u64 {
    let Ok(inputs) = model.node_input_facts(node.id) else { return 0 };
    let Ok(costs) = node.op.cost(&inputs) else { return 0 };
    costs
        .iter()
        .filter(|(c, _)| matches!(c, Cost::FMA(_)))
        .filter_map(|(_, d)| d.to_usize().ok())
        .map(|v| v as u64)
        .sum()
}

fn bytes(model: &TypedModel, outlet: OutletId) -> Option<u64> {
    let fact = model.outlet_fact(outlet).ok()?;
    let volume = fact.shape.volume().to_usize().ok()?;
    Some((volume * fact.datum_type.size_of()) as u64)
}

fn wire_name(model: &TypedModel, outlet: OutletId) -> String {
    let node = model.node(outlet.node);
    if node.outputs.len() > 1 {
        format!("{}:{}", node.name, outlet.slot)
    } else {
        node.name.clone()
    }
}

fn channel_move(conv: &Conv, rank: usize, shape: &[TDim]) -> TractResult<AxisOp> {
    let data_shape = conv.pool_spec.data_format.shape(shape.to_vec())?;
    let c = data_shape.c_axis();
    Ok(if conv.pool_spec.data_format.c_is_last() {
        AxisOp::Move(c, data_shape.n_axis().map(|_| 1).unwrap_or(0))
    } else {
        AxisOp::Move(c, rank - 1)
    })
}

fn flood(model: &TypedModel, start: OutletId, op: AxisOp) -> TractResult<Flood> {
    let inputs: HashSet<OutletId> = model.input_outlets()?.iter().copied().collect();
    let outputs: HashSet<OutletId> = model.output_outlets()?.iter().copied().collect();
    let mut f = Flood { wires: HashMap::new(), accepted: HashMap::new(), boundaries: vec![] };
    f.wires.insert(start, op.clone());
    let mut todo = vec![(start, op, None::<usize>)];
    while let Some((outlet, op, emitter)) = todo.pop() {
        if outputs.contains(&outlet) {
            f.boundaries.push((outlet, outlet.node, "model output".into(), "Output".into()));
        }
        let mut sites: Vec<(usize, InOut)> = vec![(outlet.node, InOut::Out(outlet.slot))];
        for inlet in model.outlet_successors(outlet) {
            sites.push((inlet.node, InOut::In(inlet.slot)));
        }
        for (node_id, io) in sites {
            if Some(node_id) == emitter {
                continue;
            }
            let node = model.node(node_id);
            if node.op_is::<TypedSource>() || inputs.contains(&outlet) && io.is_output() {
                f.boundaries.push((outlet, node_id, "model input".into(), "Source".into()));
                continue;
            }
            if let Some(einsum) = node.op_as::<EinSum>() {
                let current = f
                    .accepted
                    .get(&node_id)
                    .and_then(|(op, _)| op.downcast_ref::<EinSum>())
                    .unwrap_or(einsum)
                    .clone();
                let Some(AxisChangeConsequence { substitute_op: Some(new_op), wire_changes }) =
                    current.change_axes(model, node, io, &op)?
                else {
                    f.boundaries.push((outlet, node_id, "refuses".into(), "EinSum".into()));
                    continue;
                };
                let after = new_op.downcast_ref::<EinSum>().cloned();
                let entry = f.accepted.entry(node_id).or_insert_with(|| (new_op.clone(), tvec!()));
                entry.0 = new_op;
                entry.1.extend(wire_changes);
                let out: OutletId = node.id.into();
                if let (InOut::In(slot), AxisOp::Move(_, to), Some(after)) = (io, &op, after)
                    && !f.wires.contains_key(&out)
                    && let Some(change) = einsum_output_move(&current, &after, node, slot, *to)
                {
                    f.wires.insert(out, change.clone());
                    todo.push((out, change, None));
                }
                continue;
            }
            if let Some((_, changes)) = f.accepted.get(&node_id) {
                if !changes.iter().any(|(w, c)| *w == io && *c == op) {
                    f.boundaries.push((
                        outlet,
                        node_id,
                        "conflicting change".into(),
                        node.op.name().to_string(),
                    ));
                }
                continue;
            }
            let answer = node.op.change_axes(model, node, io, &op);
            let Ok(Some(AxisChangeConsequence { substitute_op, wire_changes })) = answer else {
                let reason = match answer {
                    Err(e) => format!("error: {e}"),
                    _ => "refuses".to_string(),
                };
                f.boundaries.push((outlet, node_id, reason, node.op.name().to_string()));
                continue;
            };
            let new_op = substitute_op.unwrap_or_else(|| node.op.clone());
            f.accepted.insert(node_id, (new_op, wire_changes.clone()));
            for (wire, change) in wire_changes {
                let other = wire.as_outlet(node);
                if wire.is_input() && model.node(other.node).op_is::<Const>() {
                    continue;
                }
                match f.wires.get(&other) {
                    Some(previous) if *previous == change => {}
                    Some(_) => f.boundaries.push((
                        other,
                        node_id,
                        "conflicting change".into(),
                        node.op.name().to_string(),
                    )),
                    None => {
                        f.wires.insert(other, change.clone());
                        todo.push((other, change, Some(node_id)));
                    }
                }
            }
        }
    }
    Ok(f)
}

/// The output side of a layout switch entering an EinSum on input `slot` (`change`, already
/// applied to `after`), when the EinSum reads like a pointwise conv: one non-unit output axis
/// absent from that input (the output channel). It moves to where the input channel went,
/// relative to the axes the output shares with the input.
fn einsum_output_move(
    before: &EinSum,
    after: &EinSum,
    node: &TypedNode,
    slot: usize,
    to: usize,
) -> Option<AxisOp> {
    let shape = &node.outputs[0].fact.shape;
    let mut own = vec![];
    let mut shared_before_channel = 0;
    for axis in before.axes.iter_all_axes() {
        let Some(&pos) = axis.outputs[0].first() else { continue };
        if axis.inputs[slot].is_empty() {
            if !shape[pos].is_one() {
                own.push(pos);
            }
        } else {
            let new_in = after.axes.axis(axis.repr).ok()?.inputs[slot].first().copied()?;
            if new_in < to {
                shared_before_channel += 1;
            }
        }
    }
    let [from] = own[..] else { return None };
    let unit_only_before = before
        .axes
        .iter_all_axes()
        .filter(|a| a.inputs[slot].is_empty())
        .filter_map(|a| a.outputs[0].first().copied())
        .filter(|&p| p < from && shape[p].is_one())
        .count();
    let target = shared_before_channel + unit_only_before;
    (target != from).then_some(AxisOp::Move(from, target))
}

fn conv_info(model: &TypedModel, node: &TypedNode, conv: &Conv) -> TractResult<ConvInfo> {
    let input = model.outlet_fact(node.inputs[0])?;
    let output = &node.outputs[0].fact;
    let spec = &conv.pool_spec;
    let out_shape = spec.data_format.shape(output.shape.to_tvec())?;
    let in_shape = spec.data_format.shape(input.shape.to_tvec())?;
    let kernel: Vec<usize> = spec.kernel_shape.to_vec();
    let rank = kernel.len();
    let stride: Vec<usize> = (0..rank).map(|a| spec.stride(a)).collect();
    let dilated = (0..rank).any(|a| spec.dilation(a) > 1);
    let taps: usize = kernel.iter().product();
    let (cin, cout, group) = (conv.input_channels(), conv.output_channels(), conv.group);
    let kind = if group > 1 && group == cin && group == cout {
        "depthwise"
    } else if group > 1 {
        "grouped"
    } else if taps == 1 {
        "1x1"
    } else {
        "dense"
    };
    let n = out_shape.n().and_then(|n| n.to_usize().ok()).unwrap_or(1);
    let points = out_shape
        .hw_dims()
        .iter()
        .map(|d| d.to_usize().ok())
        .product::<Option<usize>>()
        .map(|p| (p * n) as u64);
    let size = input.datum_type.size_of() as u64;
    let gathered = points.map(|p| match kind {
        "depthwise" => 0,
        _ => p * (cin as u64) * (taps as u64) * size,
    });
    let input_bytes = in_shape
        .shape
        .iter()
        .map(|d| d.to_usize().ok())
        .product::<Option<usize>>()
        .map(|v| v as u64 * size);
    Ok(ConvInfo {
        name: node.name.clone(),
        format: format!("{:?}", spec.data_format),
        kind: kind.into(),
        kernel,
        stride,
        dilated,
        cin,
        cout,
        group,
        fan_out: cout / group,
        points,
        macs: macs(model, node),
        input_bytes,
        gathered_bytes_channel_first: gathered,
        dt: format!("{:?}", input.datum_type),
    })
}

fn kernel_key(c: &ConvInfo, new_format: &str) -> String {
    let k = c.kernel.iter().map(|k| k.to_string()).collect::<Vec<_>>().join("x");
    let s = c.stride.iter().max().copied().unwrap_or(1);
    let d = if c.dilated { " dilated" } else { "" };
    let thin = if c.kind == "depthwise" {
        if c.cin < 16 { " C<16" } else { "" }
    } else if c.fan_out <= 32 {
        " Cout<=32"
    } else {
        ""
    };
    format!("Conv {new_format} {} {k} s{s}{d}{thin} {}", c.kind, c.dt)
}

fn build(model: &TypedModel) -> TractResult<Report> {
    let mut report = Report::default();
    let mut claimed: HashSet<usize> = HashSet::new();
    for node in model.nodes() {
        report.total_macs += macs(model, node);
        if node.op_is::<Conv>() {
            report.conv_macs += macs(model, node);
        }
    }
    for id in model.eval_order()? {
        let node = model.node(id);
        let Some(conv) = node.op_as::<Conv>() else { continue };
        if claimed.contains(&id) || conv.q_params.is_some() {
            continue;
        }
        let shape = model.outlet_fact(node.inputs[0])?.shape.to_tvec();
        let change = channel_move(conv, shape.len(), &shape)?;
        let direction = if conv.pool_spec.data_format.c_is_last() {
            "to channel-first"
        } else {
            "to channel-last"
        };
        let f = flood(model, node.inputs[0], change)?;
        if !f.accepted.contains_key(&id) {
            let info = conv_info(model, node, conv)?;
            report.unclaimed_convs.push(info);
            claimed.insert(id);
            continue;
        }
        let mut region = Region {
            direction: direction.into(),
            nodes: f.accepted.len(),
            macs: 0,
            convs: vec![],
            einsums: vec![],
            boundaries: vec![],
            boundary_bytes: 0,
            channel_concats_and_slices: vec![],
        };
        let mut ids: Vec<usize> = f.accepted.keys().copied().collect();
        ids.sort();
        for nid in ids {
            claimed.insert(nid);
            let n = model.node(nid);
            let m = macs(model, n);
            region.macs += m;
            let (new_op, _) = &f.accepted[&nid];
            let key = if let Some(c) = n.op_as::<Conv>() {
                let info = conv_info(model, n, c)?;
                let new_format = new_op
                    .downcast_ref::<Conv>()
                    .map(|c| format!("{:?}", c.pool_spec.data_format))
                    .unwrap_or_default();
                let key = kernel_key(&info, &new_format);
                region.convs.push(info);
                key
            } else if let Some(e) = n.op_as::<EinSum>() {
                let after = new_op.downcast_ref::<EinSum>().unwrap_or(e);
                let rank0 = after.axes.rank(InOut::In(0));
                let innermost = (0..n.inputs.len()).any(|slot| {
                    let r = after.axes.rank(InOut::In(slot));
                    r > 0
                        && after
                            .axes
                            .iter_all_axes()
                            .find(|a| a.inputs[slot].contains(&(r - 1)))
                            .is_some_and(|a| a.outputs[0].is_empty())
                        && !model.node(n.inputs[slot].node).op_is::<Const>()
                });
                let _ = rank0;
                region.einsums.push(EinSumInfo {
                    name: n.name.clone(),
                    expr: format!("{} -> {}", e.axes, after.axes),
                    macs: m,
                    contraction_innermost_after: innermost,
                });
                if m > 0 {
                    format!(
                        "EinSum (matmul) {}",
                        if innermost { "activation contraction innermost" } else { "other layout" }
                    )
                } else {
                    "EinSum (no multiply-add)".to_string()
                }
            } else {
                let rank = n.outputs.first().map(|o| o.fact.rank()).unwrap_or(0);
                if let Some(c) = new_op.downcast_ref::<TypedConcat>()
                    && c.axis + 1 == rank
                {
                    region.channel_concats_and_slices.push(format!("concat {}", n.name));
                    "Concat on innermost axis".to_string()
                } else if let Some(s) = new_op.downcast_ref::<Slice>()
                    && s.axis + 1 == rank
                {
                    region.channel_concats_and_slices.push(format!("slice {}", n.name));
                    "Slice on innermost axis".to_string()
                } else {
                    n.op.name().to_string()
                }
            };
            let sum = report.switched_ops.entry(key).or_default();
            sum.count += 1;
            sum.macs += m;
        }
        let mut boundaries = f.boundaries.clone();
        for &nid in f.accepted.keys() {
            for slot in 0..model.node(nid).outputs.len() {
                let out = OutletId::new(nid, slot);
                if !f.wires.contains_key(&out) && !model.outlet_successors(out).is_empty() {
                    boundaries.push((
                        out,
                        nid,
                        "seam: the op takes the new layout in and keeps its output layout, no transpose".into(),
                        model.node(nid).op.name().to_string(),
                    ));
                }
            }
        }
        let mut seen = HashSet::new();
        for (outlet, at, reason, op) in &boundaries {
            if !seen.insert((*outlet, *at)) {
                continue;
            }
            let b = bytes(model, *outlet);
            let seam = reason.starts_with("seam");
            if !seam {
                region.boundary_bytes += b.unwrap_or(0);
            }
            let blocker = if reason.starts_with("model") {
                reason.clone()
            } else if seam {
                format!("{op} (seam, free)")
            } else {
                op.clone()
            };
            let sum = report.blockers.entry(blocker).or_default();
            sum.count += 1;
            sum.bytes += b.unwrap_or(0);
            if sum.nodes.len() < 4 {
                sum.nodes.push(model.node(*at).name.clone());
            }
            region.boundaries.push(Boundary {
                wire: wire_name(model, *outlet),
                reason: reason.clone(),
                op: format!("{} ({})", model.node(*at).name, op),
                bytes: b,
            });
        }
        report.regions.push(region);
    }
    Ok(report)
}

fn mb(b: u64) -> String {
    format!("{:.2} MB", b as f64 / 1e6)
}

fn print(report: &Report) {
    println!(
        "model: {:.1} MMAC, {:.1} MMAC in convolutions, {} switchable region(s)",
        report.total_macs as f64 / 1e6,
        report.conv_macs as f64 / 1e6,
        report.regions.len()
    );
    for (ix, r) in report.regions.iter().enumerate() {
        let share = 100.0 * r.macs as f64 / report.total_macs.max(1) as f64;
        println!(
            "\nregion {ix}: {}, {} nodes, {} convs, {:.1}% of model MACs, {} boundary transposes ({})",
            r.direction,
            r.nodes,
            r.convs.len(),
            share,
            r.boundaries.len(),
            mb(r.boundary_bytes)
        );
        for c in &r.convs {
            println!(
                "  conv {:9} k{:?} s{:?}{} cin {:4} cout {:4} F {:4} pts {:>7} {:>8.1} MMAC gather {:>10} {}",
                c.kind,
                c.kernel,
                c.stride,
                if c.dilated { " dil" } else { "" },
                c.cin,
                c.cout,
                c.fan_out,
                c.points.map(|p| p.to_string()).unwrap_or("?".into()),
                c.macs as f64 / 1e6,
                c.gathered_bytes_channel_first.map(mb).unwrap_or("?".into()),
                c.name
            );
        }
        let ein_macs: u64 = r.einsums.iter().map(|e| e.macs).sum();
        let inner =
            r.einsums.iter().filter(|e| e.contraction_innermost_after && e.macs > 0).count();
        if !r.einsums.is_empty() {
            println!(
                "  einsums: {} ({:.1} MMAC), {} with an activation contraction axis innermost after the switch",
                r.einsums.len(),
                ein_macs as f64 / 1e6,
                inner
            );
        }
        if !r.channel_concats_and_slices.is_empty() {
            println!("  innermost-axis concat/slice: {}", r.channel_concats_and_slices.len());
        }
        for b in &r.boundaries {
            println!(
                "  boundary {:>10} at {} [{}] on {}",
                b.bytes.map(mb).unwrap_or("?".into()),
                b.op,
                b.reason,
                b.wire
            );
        }
    }
    if !report.unclaimed_convs.is_empty() {
        println!("\nconvs whose own layout change is refused: {}", report.unclaimed_convs.len());
    }
    println!("\nblockers (op: boundaries, bytes):");
    let mut blockers: Vec<_> = report.blockers.iter().collect();
    blockers.sort_by_key(|(_, s)| std::cmp::Reverse(s.count));
    for (op, s) in blockers {
        println!("  {op:30} {:4} {:>12}  e.g. {}", s.count, mb(s.bytes), s.nodes.join(", "));
    }
    println!("\nops in switched regions (count, MMAC):");
    let mut ops: Vec<_> = report.switched_ops.iter().collect();
    ops.sort_by_key(|(_, s)| std::cmp::Reverse((s.macs, s.count)));
    for (op, s) in ops {
        println!("  {op:60} {:4} {:>10.1}", s.count, s.macs as f64 / 1e6);
    }
}

pub fn handle(params: &Parameters, matches: &clap::ArgMatches) -> TractResult<()> {
    let model = params
        .tract_model
        .downcast_ref::<TypedModel>()
        .context("layout-report needs a typed model")?;
    let report = build(model)?;
    if matches.get_flag("json") {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print(&report);
    }
    Ok(())
}
