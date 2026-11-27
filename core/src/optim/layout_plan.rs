//! Layout planner: decides per region whether depthwise convolutions and the ops around them run channel-last.
//!
//! Declutter leaves convolutions channel-first. With [`TRACT_LAYOUT_PLANNER`] on, [`TypedModel::optimize`] first calls
//! [`plan`] on the decluttered model: each depthwise convolution with a channel-last kernel seeds a region (the axis
//! change propagated as far as `change_axes` goes, so seeds in one block share a region). A [`LayoutTable`] estimates
//! the time change between the optimized plans with no region and with every region flipped, and the change of each
//! node is credited to the region it comes from. The regions that gain are flipped together when their summed gain
//! exceeds [`MARGIN`] of the model's estimated time.
//!
//! A table holds, per op key of an optimized node, channel-first and channel-last times and channel-first work as
//! measured in models by `tract layout-calibrate`. A node's time is estimated as its work times its key's rate, and a
//! node the flip changes contributes that time times its key's channel-last over channel-first ratio, minus one.

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fmt::Write;

use crate::internal::*;
use crate::ops::change_axes::AxisChange;
use crate::ops::cnn::Conv;
use crate::ops::matmul::optimized::OptMatMul;
use crate::optim::change_axes::change_axes;
use tract_data::knobs::KnobScope;

crate::declare_knob!(
    scopable TRACT_LAYOUT_PLANNER,
    bool,
    false,
    "Choose per region whether depthwise convolutions run channel-last, from measured layout costs, instead of \
     declutter flipping all of them on hosts with a channel-last depthwise kernel. Read when the model is optimized."
);

crate::declare_knob!(
    scopable TRACT_LAYOUT_TABLE,
    String,
    String::new(),
    "Path of a layout cost table (written by `tract layout-calibrate`) for the layout planner, in place of the one \
     built in for the host."
);

/// Minimum estimated gain, as a fraction of the model's estimated time, for the planner to flip anything.
pub const MARGIN: f64 = 0.05;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayoutEntry {
    pub nodes: usize,
    pub channel_first_secs: f64,
    pub channel_last_secs: f64,
    pub work_unit: String,
    pub channel_first_work: f64,
}

/// Per op key layout costs. Text form: one `model\t<name>` line per calibrated model and one
/// `entry\t<key>\t<nodes>\t<channel-first secs>\t<channel-last secs>\t<work unit>\t<channel-first work>` line per key.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayoutTable {
    pub models: Vec<String>,
    pub entries: BTreeMap<String, LayoutEntry>,
}

impl LayoutTable {
    pub fn parse(text: &str) -> TractResult<LayoutTable> {
        let mut table = LayoutTable::default();
        for line in text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
            let fields: Vec<&str> = line.split('\t').collect();
            match &*fields {
                ["model", name] => table.models.push(name.to_string()),
                ["entry", key, nodes, cf, cl, unit, work] => {
                    table.entries.insert(
                        key.to_string(),
                        LayoutEntry {
                            nodes: nodes.parse()?,
                            channel_first_secs: cf.parse()?,
                            channel_last_secs: cl.parse()?,
                            work_unit: unit.to_string(),
                            channel_first_work: work.parse()?,
                        },
                    );
                }
                _ => bail!("Invalid layout table line: {line:?}"),
            }
        }
        Ok(table)
    }

    pub fn render(&self) -> String {
        let mut s = String::new();
        for m in &self.models {
            writeln!(s, "model\t{m}").unwrap();
        }
        for (k, e) in &self.entries {
            writeln!(
                s,
                "entry\t{k}\t{}\t{:e}\t{:e}\t{}\t{:e}",
                e.nodes,
                e.channel_first_secs,
                e.channel_last_secs,
                e.work_unit,
                e.channel_first_work
            )
            .unwrap();
        }
        s
    }

    pub fn add(
        &mut self,
        key: &str,
        unit: &str,
        channel_first_secs: f64,
        channel_last_secs: f64,
        work: f64,
    ) {
        let entry = self.entries.entry(key.to_string()).or_default();
        entry.nodes += 1;
        entry.channel_first_secs += channel_first_secs;
        entry.channel_last_secs += channel_last_secs;
        entry.work_unit = unit.to_string();
        entry.channel_first_work += work;
    }

    pub fn merge(&mut self, other: &LayoutTable) {
        self.models.extend(other.models.iter().cloned());
        for (k, e) in &other.entries {
            let entry = self.entries.entry(k.clone()).or_default();
            entry.nodes += e.nodes;
            entry.channel_first_secs += e.channel_first_secs;
            entry.channel_last_secs += e.channel_last_secs;
            entry.work_unit = e.work_unit.clone();
            entry.channel_first_work += e.channel_first_work;
        }
    }

    /// Channel-last over channel-first time, and channel-first time per unit of work, for `key`. A key absent from the
    /// table uses all entries sharing its longest present prefix of space-separated tokens.
    pub fn lookup(&self, key: &str) -> Option<(f64, f64)> {
        let tokens: Vec<&str> = key.split(' ').collect();
        (1..=tokens.len()).rev().find_map(|n| {
            let (cf, cl, work) = self
                .entries
                .iter()
                .filter(|(k, _)| k.split(' ').take(n).eq(tokens[..n].iter().copied()))
                .fold((0.0, 0.0, 0.0), |(a, b, c), (_, e)| {
                    (a + e.channel_first_secs, b + e.channel_last_secs, c + e.channel_first_work)
                });
            (cf > 0.0 && work > 0.0).then_some((cl / cf, cf / work))
        })
    }

    /// The table built in for the host, if any.
    pub fn builtin() -> Option<LayoutTable> {
        #[cfg(target_arch = "aarch64")]
        {
            LayoutTable::parse(include_str!("layout_tables/aarch64.tsv")).ok()
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            None
        }
    }

    /// The table named by [`TRACT_LAYOUT_TABLE`], else the built-in one.
    pub fn current() -> TractResult<Option<LayoutTable>> {
        let path = TRACT_LAYOUT_TABLE.get();
        if path.is_empty() {
            return Ok(LayoutTable::builtin());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("Reading layout table {path}"))?;
        Ok(Some(LayoutTable::parse(&text).with_context(|| format!("Parsing layout table {path}"))?))
    }
}

fn bucket(v: usize, edges: &[usize]) -> usize {
    edges.iter().copied().find(|e| v <= *e).unwrap_or(edges[edges.len() - 1] * 2)
}

fn volume(fact: &TypedFact) -> f64 {
    fact.shape.as_concrete().map(|s| s.iter().product::<usize>() as f64).unwrap_or(0.0)
}

fn is_packing(name: &str) -> bool {
    name == "OptMatMulPack" || name.to_lowercase().contains("im2col")
}

fn matmul_key(model: &TypedModel, node: &TypedNode) -> String {
    let Some(mm) = node.op_as::<OptMatMul>() else {
        return "matmul other".into();
    };
    let shape = mm.c_fact.shape.as_concrete();
    let axis_len = |a: Option<usize>| a.and_then(|a| shape.map(|s| s[a])).unwrap_or(1);
    let (m, n) = (axis_len(mm.c_m_axis), axis_len(mm.c_n_axis));
    if m.min(n) < 4 || m.max(n) < 64 {
        return "matmul gemv".into();
    }
    let fed_by_im2col =
        node.inputs.iter().any(|i| model.node(i.node).op.name().to_lowercase().contains("im2col"));
    let kind = if fed_by_im2col { "im2col" } else { "1x1" };
    format!("matmul {kind} cout<={}", bucket(m.min(n), &[16, 32, 64, 128, 256]))
}

/// Depthwise keys put channels first and taps last: the channel-last ratio follows the channel count, the time per
/// multiply-add the output width, and an unseen kernel size borrows from its channel and width class.
fn depthwise_key(model: &TypedModel, node: &TypedNode) -> TractResult<String> {
    let input = model.outlet_fact(node.inputs[0])?;
    let kernel = model.outlet_fact(node.inputs[1])?;
    let output = &node.outputs[0].fact;
    let (Some(i), Some(o)) = (input.shape.as_concrete(), output.shape.as_concrete()) else {
        return Ok("depthwise".into());
    };
    let c_axis = (i.len() == 4) as usize;
    let c = i[c_axis];
    let taps = (volume(kernel) as usize) / c.max(1);
    let stride = i[c_axis + 1] / o[c_axis + 1].max(1);
    let width = o[o.len() - 1];
    Ok(format!(
        "depthwise c<={} w<={} s{stride} taps{taps}",
        bucket(c, &[64, 256]),
        bucket(width, &[8, 20, 40])
    ))
}

/// Op key, work unit and work of a node of an optimized channel-first plan.
pub fn node_key(model: &TypedModel, node: &TypedNode) -> TractResult<(String, String, f64)> {
    let inputs = model.node_input_facts(node.id)?;
    let fma: f64 = node
        .op
        .cost(&inputs)?
        .iter()
        .filter(|(c, _)| matches!(c, Cost::FMA(_)))
        .filter_map(|(_, d)| d.to_usize().ok())
        .map(|v| v as f64)
        .sum();
    let name = node.op.name();
    let key = if name == "DepthWiseConv" {
        depthwise_key(model, node)?
    } else if name == "OptMatMul" {
        matmul_key(model, node)
    } else if is_packing(&name) {
        match node.outputs[0].successors.first().map(|s| model.node(s.node)) {
            Some(c) if c.op.name() == "OptMatMul" => format!("{name} for {}", matmul_key(model, c)),
            _ => name.to_string(),
        }
    } else {
        name.to_string()
    };
    let (unit, work) = if fma > 0.0 {
        ("fma", fma)
    } else if is_packing(&name) {
        ("elements", inputs.iter().map(|f| volume(f)).sum())
    } else {
        ("elements", node.outputs.iter().map(|o| volume(&o.fact)).sum())
    };
    Ok((key, unit.to_string(), work))
}

/// Op name and first output shape of each node, by node name: two plans of a model differ where these do.
pub fn signatures(model: &TypedModel) -> HashMap<String, String> {
    model
        .nodes()
        .iter()
        .map(|n| {
            (
                n.name.clone(),
                format!("{} {:?}", n.op.name(), n.outputs.first().map(|o| &o.fact.shape)),
            )
        })
        .collect()
}

/// The nodes of the optimized channel-first plan `cf` that the flip to `cl` changes: those whose op or output shape
/// differ in `cl`, the nodes consuming them, and the packing and im2col nodes feeding a changed matrix product.
pub fn region(cf: &TypedModel, cl: &TypedModel) -> HashSet<usize> {
    let (cf_signatures, cl_signatures) = (signatures(cf), signatures(cl));
    let changed: HashSet<usize> = cf
        .nodes()
        .iter()
        .filter(|n| {
            cl_signatures.get(&n.name).is_some_and(|s| Some(s) != cf_signatures.get(&n.name))
        })
        .map(|n| n.id)
        .collect();
    let mut region = changed.clone();
    for node in cf.nodes() {
        if node.inputs.iter().any(|i| changed.contains(&i.node)) {
            region.insert(node.id);
        }
    }
    for node in cf.nodes() {
        if is_packing(&node.op.name())
            && node.outputs.iter().flat_map(|o| &o.successors).any(|s| region.contains(&s.node))
        {
            region.insert(node.id);
        }
    }
    region
}

#[derive(Clone, Debug, Default)]
pub struct Estimate {
    /// Estimated time of the channel-first plan, over the nodes the table knows.
    pub total: f64,
    /// Estimated time change of the flip.
    pub delta: f64,
    /// Time change per op key, largest first.
    pub by_key: Vec<(String, f64)>,
    pub region_nodes: usize,
    /// Nodes with work the table has no entry for, left out of `total` and `delta`.
    pub unknown: usize,
    /// Time change per node of the channel-first plan, by node id, for the nodes the flip changes.
    pub node_deltas: HashMap<usize, f64>,
}

/// Estimated time change from the optimized channel-first plan `cf` to the optimized channel-last plan `cl`.
pub fn estimate(table: &LayoutTable, cf: &TypedModel, cl: &TypedModel) -> TractResult<Estimate> {
    let region = region(cf, cl);
    let mut est = Estimate { region_nodes: region.len(), ..Estimate::default() };
    let mut by_key: HashMap<String, f64> = HashMap::new();
    for node in cf.nodes() {
        let (key, _, work) = node_key(cf, node)?;
        let Some((ratio, rate)) = table.lookup(&key) else {
            est.unknown += (work > 0.0) as usize;
            continue;
        };
        est.total += work * rate;
        if region.contains(&node.id) {
            let delta = work * rate * (ratio - 1.0);
            est.delta += delta;
            est.node_deltas.insert(node.id, delta);
            *by_key.entry(key).or_default() += delta;
        }
    }
    est.by_key = by_key.into_iter().collect();
    est.by_key.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    Ok(est)
}

fn codegen(model: &TypedModel) -> TractResult<TypedModel> {
    let mut model = model.clone();
    model.optimize_with_session(&mut crate::optim::Optimizer::codegen().session())?;
    Ok(model)
}

/// The patch propagating the channel-last request of convolution `conv` through the decluttered `model`, if it
/// propagates.
fn seed_patch(model: &TypedModel, conv: &str) -> TractResult<Option<TypedModelPatch>> {
    let node = model.node_by_name(conv)?;
    let Some((io, op)) = node.op_as::<Conv>().and_then(|c| c.channel_last_request()) else {
        return Ok(None);
    };
    let mut interfaces = model.output_outlets()?.to_vec();
    interfaces.extend(model.input_outlets()?.iter());
    let change = AxisChange { outlet: io.as_outlet(node), op };
    Ok(change_axes(model, &change, &interfaces, &[], &mut HashSet::default())?
        .map(|(patch, _)| patch))
}

/// Copy of the decluttered `model` with the channel-last requests of `convs` propagated, decluttered.
fn flip(model: &TypedModel, convs: &[String]) -> TractResult<TypedModel> {
    let mut flipped = model.clone();
    for conv in convs {
        if let Some(patch) = seed_patch(&flipped, conv)? {
            patch.apply(&mut flipped)?;
        }
    }
    flipped.declutter()?;
    Ok(flipped)
}

fn is_channel_last(model: &TypedModel, conv: &str) -> bool {
    model
        .node_by_name(conv)
        .ok()
        .and_then(|n| n.op_as::<Conv>())
        .is_some_and(|c| c.pool_spec.data_format.c_is_last())
}

/// One region the planner considered: the convolutions it covers, its estimated time change, and the decision.
#[derive(Clone, Debug)]
pub struct RegionPlan {
    pub convs: Vec<String>,
    pub delta: f64,
    pub flipped: bool,
}

struct Region {
    convs: Vec<String>,
    /// Names of the decluttered nodes the flip changes, and of their consumers.
    nodes: HashSet<String>,
}

/// The regions of the decluttered `model`: the seeds one channel-last request covers, and the nodes it changes.
fn regions(model: &TypedModel, seeds: &[String]) -> TractResult<Vec<Region>> {
    let before = signatures(model);
    let mut regions: Vec<Region> = vec![];
    for seed in seeds {
        if regions.iter().any(|r| r.convs.contains(seed)) {
            continue;
        }
        let Some(patch) = seed_patch(model, seed)? else { continue };
        let mut patched = model.clone();
        patch.apply(&mut patched)?;
        let convs: Vec<String> =
            seeds.iter().filter(|s| is_channel_last(&patched, s)).cloned().collect();
        if !convs.contains(seed) {
            continue;
        }
        let after = signatures(&patched);
        let changed: Vec<&TypedNode> =
            model.nodes().iter().filter(|n| after.get(&n.name) != before.get(&n.name)).collect();
        let mut nodes: HashSet<String> = changed.iter().map(|n| n.name.clone()).collect();
        for node in &changed {
            for succ in node.outputs.iter().flat_map(|o| &o.successors) {
                nodes.insert(model.node(succ.node).name.clone());
            }
        }
        regions.push(Region { convs, nodes });
    }
    Ok(regions)
}

/// Index in `regions` of the region a node of the optimized plan `cf` comes from: the region holding the longest
/// dot-separated prefix of its name that names a decluttered node, else the region of a neighbour.
fn attribute(
    cf: &TypedModel,
    regions: &[Region],
    ids: impl Iterator<Item = usize>,
) -> HashMap<usize, usize> {
    let region_of = |name: &str| -> Option<usize> {
        let mut prefix = name;
        loop {
            if let Some(ix) = regions.iter().position(|r| r.nodes.contains(prefix)) {
                return Some(ix);
            }
            prefix = &prefix[..prefix.rfind('.')?];
        }
    };
    let ids: Vec<usize> = ids.collect();
    let mut found: HashMap<usize, usize> =
        ids.iter().filter_map(|&id| region_of(&cf.node(id).name).map(|r| (id, r))).collect();
    for _ in 0..ids.len() {
        let before = found.len();
        for &id in &ids {
            if found.contains_key(&id) {
                continue;
            }
            let node = cf.node(id);
            let neighbour = node
                .inputs
                .iter()
                .map(|i| i.node)
                .chain(node.outputs.iter().flat_map(|o| o.successors.iter().map(|s| s.node)))
                .find_map(|n| found.get(&n).copied());
            if let Some(r) = neighbour {
                found.insert(id, r);
            }
        }
        if found.len() == before {
            break;
        }
    }
    found
}

/// Flip the regions of the decluttered, channel-first `model` that `table` predicts faster channel-last, and return the
/// regions considered. No-op when the host has no channel-last depthwise kernel or no convolution asks for it.
///
/// The estimate compares the channel-first plan with the plan where every region is flipped, and splits the change
/// between regions by where the changed nodes come from, so the model is optimized twice whatever its region count.
pub fn plan(model: &mut TypedModel, table: &LayoutTable) -> TractResult<Vec<RegionPlan>> {
    KnobScope::new([("TRACT_LAYOUT_PLANNER", "true")])?.enter(|| plan_in_scope(model, table))
}

fn plan_in_scope(model: &mut TypedModel, table: &LayoutTable) -> TractResult<Vec<RegionPlan>> {
    let seeds: Vec<String> = model
        .nodes()
        .iter()
        .filter(|n| n.op_as::<Conv>().is_some_and(|c| c.channel_last_request().is_some()))
        .map(|n| n.name.clone())
        .collect();
    let regions = regions(model, &seeds)?;
    if regions.is_empty() {
        return Ok(vec![]);
    }
    let all_convs: Vec<String> = regions.iter().map(|r| r.convs[0].clone()).collect();
    let all_flipped = flip(model, &all_convs)?;
    let baseline = codegen(model)?;
    let est = estimate(table, &baseline, &codegen(&all_flipped)?)?;
    let owner = attribute(&baseline, &regions, est.node_deltas.keys().copied());
    let mut plans: Vec<RegionPlan> = regions
        .iter()
        .map(|r| RegionPlan { convs: r.convs.clone(), delta: 0.0, flipped: false })
        .collect();
    for (id, delta) in &est.node_deltas {
        if let Some(&r) = owner.get(id) {
            plans[r].delta += delta;
        }
    }
    let gain: f64 = plans.iter().filter(|p| p.delta < 0.0).map(|p| -p.delta).sum();
    if gain > MARGIN * est.total {
        plans.iter_mut().for_each(|p| p.flipped = p.delta < 0.0);
    }
    if plans.iter().all(|p| p.flipped) {
        *model = all_flipped;
    } else if plans.iter().any(|p| p.flipped) {
        let convs: Vec<String> =
            plans.iter().filter(|p| p.flipped).map(|p| p.convs[0].clone()).collect();
        *model = flip(model, &convs)?;
    }
    let unattributed: f64 =
        est.node_deltas.iter().filter(|(id, _)| !owner.contains_key(id)).map(|(_, d)| d).sum();
    for p in &plans {
        info!(
            "Layout region {:?}: estimated {:+.2}% of {:.3} ms, {}",
            p.convs,
            100.0 * p.delta / est.total.max(f64::MIN_POSITIVE),
            est.total * 1e3,
            if p.flipped { "channel-last" } else { "channel-first" }
        );
    }
    if unattributed != 0.0 {
        info!(
            "Layout planner: {:+.2}% estimated on nodes no region claims",
            100.0 * unattributed / est.total.max(f64::MIN_POSITIVE)
        );
    }
    Ok(plans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_round_trips_and_falls_back_on_prefixes() {
        let mut table = LayoutTable::default();
        table.models.push("a.onnx".into());
        table.add("depthwise c<=64 w<=80 s1 taps9", "fma", 2.0, 4.0, 100.0);
        table.add("depthwise c<=64 w<=80 s1 taps9", "fma", 1.0, 2.0, 50.0);
        table.add("depthwise c<=512 w<=20 s1 taps25", "fma", 10.0, 1.0, 100.0);
        assert_eq!(LayoutTable::parse(&table.render()).unwrap(), table);
        assert_eq!(table.lookup("depthwise c<=64 w<=80 s1 taps9"), Some((2.0, 0.02)));
        assert_eq!(table.lookup("depthwise c<=64 w<=80 s1 taps25"), Some((2.0, 0.02)));
        assert_eq!(
            table.lookup("depthwise c<=128 w<=8 s2 taps9"),
            Some((7.0 / 13.0, 13.0 / 250.0))
        );
        assert_eq!(table.lookup("OptMatMul"), None);
    }

    #[test]
    fn builtin_table_parses() {
        LayoutTable::parse(include_str!("layout_tables/aarch64.tsv")).unwrap();
    }
}
