use nu_ansi_term::Style;
use std::collections::HashMap;
#[allow(unused_imports)]
use std::convert::TryFrom;
use std::time::Duration;
use tract_core::internal::*;
use tract_core::ops::scan::Scan;
use tract_itertools::izip;
use tract_itertools::Itertools;

use crate::model::Model;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct NodeQId(pub TVec<(usize, String)>, pub usize);

impl From<usize> for NodeQId {
    fn from(id: usize) -> NodeQId {
        NodeQId(tvec!(), id)
    }
}

impl NodeQId {
    pub fn model<'a>(&self, model: &'a dyn Model) -> Option<&'a dyn Model> {
        fn scope<'a>(path: &[(usize, String)], model: &'a dyn Model) -> Option<&'a dyn Model> {
            if path.is_empty() {
                Some(model)
            } else {
                model
                    .nested_models(path[0].0)
                    .into_iter()
                    .find(|(name, _sub)| name == &path[0].1)
                    .map(|(_, sub)| sub)
            }
        }
        scope(&self.0, model)
    }
}

#[derive(Debug, Default, Clone)]
pub struct NodeTags {
    pub cost: Vec<(Cost, TDim)>,
    pub tmp_mem_usage: Option<TDim>,
    pub style: Option<Style>,
    pub labels: Vec<String>,
    pub sections: Vec<Vec<String>>,
    pub profile: Option<Duration>,
    pub accelerator_profile: Option<Duration>,
    pub model_input: Option<String>,
    pub model_output: Option<String>,
    pub outlet_labels: Vec<Vec<String>>,
    pub outlet_axes: Vec<Vec<String>>,
}

impl<'a> std::ops::Add<&'a NodeTags> for &'a NodeTags {
    type Output = NodeTags;
    fn add(self, other: &'a NodeTags) -> NodeTags {
        let cost = self
            .cost
            .iter()
            .chain(other.cost.iter())
            .sorted_by_key(|(a, _)| a)
            .group_by(|(a, _)| a)
            .into_iter()
            .map(|(cost, dims)| (*cost, dims.into_iter().fold(0.to_dim(), |acc, d| acc + &d.1)))
            .collect::<Vec<(Cost, TDim)>>();

        let tmp_mem_usage = match (self.tmp_mem_usage.clone(), other.tmp_mem_usage.clone()) {
            (Some(self_mem), Some(other_mem)) => Some(self_mem + other_mem),
            (_, Some(mem)) | (Some(mem), _) => Some(mem),
            (None, None) => None
        };

        let profile = self.profile.unwrap_or_default() + other.profile.unwrap_or_default();
        let profile = if profile != Duration::default() { Some(profile) } else { None };
        let accelerator_profile = self.accelerator_profile.unwrap_or_default()
            + other.accelerator_profile.unwrap_or_default();
        let accelerator_profile = if accelerator_profile != Duration::default() {
            Some(accelerator_profile)
        } else {
            None
        };

        let style = self.style.or(other.style);
        let labels = self.labels.iter().chain(other.labels.iter()).cloned().collect();
        let sections = self.sections.iter().chain(other.sections.iter()).cloned().collect();
        let model_input = self.model_input.clone().or_else(|| other.model_input.clone());
        let model_output = self.model_output.clone().or_else(|| other.model_output.clone());
        let outlet_labels = izip!(&self.outlet_labels, &other.outlet_labels)
            .map(|(s, o)| s.iter().chain(o.iter()).cloned().collect())
            .collect();
        let outlet_axes = izip!(&self.outlet_axes, &other.outlet_axes)
            .map(|(s, o)| s.iter().chain(o.iter()).cloned().collect())
            .collect();
        NodeTags {
            cost,
            tmp_mem_usage,
            profile,
            accelerator_profile,
            style,
            labels,
            sections,
            model_input,
            model_output,
            outlet_labels,
            outlet_axes,
        }
    }
}

impl<'a> std::iter::Sum<&'a NodeTags> for NodeTags {
    fn sum<I>(iter: I) -> NodeTags
    where
        I: std::iter::Iterator<Item = &'a NodeTags>,
    {
        iter.fold(EMPTY, |a, b| &a + b)
    }
}

const EMPTY: NodeTags = NodeTags {
    cost: Vec::new(),
    tmp_mem_usage: None,
    style: None,
    labels: Vec::new(),
    sections: Vec::new(),
    profile: None,
    accelerator_profile: None,
    model_output: None,
    model_input: None,
    outlet_labels: Vec::new(),
    outlet_axes: Vec::new(),
};

#[derive(Debug, Clone, Default)]
pub struct Annotations {
    pub tags: HashMap<NodeQId, NodeTags>,
    pub profile_summary: Option<ProfileSummary>,
    pub memory_summary: Option<MemorySummary>,
}

impl Annotations {
    pub fn node_mut(&mut self, qid: NodeQId) -> &mut NodeTags {
        self.tags.entry(qid).or_default()
    }


    pub fn track_tmp_memory_usage<Flushable>(
        &mut self,
        model: &dyn Model,
        flushable: Flushable,
        skip_order_opt_ram: bool,
    ) -> TractResult<()> 
    where
        Flushable: Fn(&TypedNode) -> bool {

        let Some(model) = model.downcast_ref::<TypedModel>() else { return Ok(()) };
        let order = if skip_order_opt_ram {
            tract_core::model::order::eval_order(model)?
        } else {
            tract_core::model::order::eval_order_opt_ram(model)?
        };

        let tmp_mem_usage = model.eval_tmp_memory_usage(&order, &flushable)?;

        let peak_tmp_mem_usage = tmp_mem_usage.iter()
            .map(|(n, mem)| mem.to_usize().map(|m| (*n, m)))
            .collect::<TractResult<TVec<_>>>()
            .ok()
            .and_then(|mems| {
                mems.into_iter()
                    .map(|(n, mem)| (NodeQId(tvec![], n), mem))
                    .max_by_key(|it| it.1)
            });

        self.memory_summary = peak_tmp_mem_usage
            .map(|(n, mem)| MemorySummary { max: mem, max_reached_by_node: n });

        for (n, mem_size) in tmp_mem_usage.into_iter() {
            let qid = NodeQId(tvec![], n);
            let tags = self.tags.entry(qid).or_default();
            tags.tmp_mem_usage = Some(mem_size.simplify());
        }
        Ok(())
    }

    pub fn track_axes(
        &mut self,
        model: &dyn Model,
        hints: &HashMap<OutletId, TVec<String>>,
    ) -> TractResult<()> {
        let Some(model) = model.downcast_ref::<TypedModel>() else { return Ok(()) };
        fn sub(
            annotations: &mut Annotations,
            prefix: &[(usize, String)],
            name_prefix: &str,
            model: &TypedModel,
            hints: &HashMap<OutletId, TVec<String>>,
        ) -> TractResult<()> {
            let tracking = tract_core::axes::full_axis_tracking(model)?;
            for (ix, axis) in tracking.iter().enumerate() {
                let name = axis
                    .creators
                    .iter()
                    .find_map(|cre| hints.get(cre).and_then(|hints| hints.get(axis.outlets[cre])))
                    .cloned()
                    .unwrap_or_else(|| format!("{name_prefix}x{ix}"));
                for outlet in axis.outlets.keys() {
                    let axis = axis.outlets[&outlet];
                    let qid = NodeQId(prefix.into(), outlet.node);
                    let tags = annotations.tags.entry(qid).or_default();
                    while tags.outlet_axes.len() <= outlet.slot {
                        tags.outlet_axes.push(vec![]);
                    }
                    while tags.outlet_axes[outlet.slot].len() <= axis {
                        tags.outlet_axes[outlet.slot].push(Default::default());
                    }
                    tags.outlet_axes[outlet.slot][axis].clone_from(&name);
                }
            }
            for node in &model.nodes {
                if let Some(scan) = node.op_as::<Scan>() {
                    let mut prefix: TVec<_> = prefix.into();
                    prefix.push((node.id, "loop".to_string()));
                    sub(
                        annotations,
                        &prefix,
                        &format!("{name_prefix}loop_"),
                        &scan.body,
                        &Default::default(),
                    )?;
                }
            }
            Ok(())
        }
        sub(self, &[], "", model, hints)
    }

    pub fn from_model(model: &dyn Model) -> TractResult<Annotations> {
        let mut annotations = Annotations::default();
        fn set_subio_labels(
            model: &dyn Model,
            prefix: &[(usize, String)],
            annotations: &mut Annotations,
        ) {
            for n in 0..model.nodes_len() {
                for output in 0..model.node_output_count(n) {
                    if let Some(label) = model.outlet_label((n, output).into()) {
                        let qid = NodeQId(prefix.into(), n);
                        annotations
                            .tags
                            .entry(qid.clone())
                            .or_default()
                            .outlet_labels
                            .resize(output + 1, vec![]);
                        annotations.tags.entry(qid).or_default().outlet_labels[output] =
                            vec![label.to_string()];
                    }
                }
                for (label, sub /*, ins, outs*/) in model.nested_models(n) {
                    let mut prefix: TVec<(usize, String)> = prefix.into();
                    prefix.push((n, label.to_string()));
                    set_subio_labels(sub, &prefix, annotations);
                    /*
                    ins.into_iter().enumerate().for_each(|(ix, i)| {
                    annotations.tags.entry(qid).or_default().model_input = Some(i);
                    });
                    outs.into_iter().enumerate().for_each(|(ix, o)| {
                    let qid = NodeQId(prefix.clone(), ix);
                    annotations.tags.entry(qid).or_default().model_output = Some(o);
                    });
                    */
                }
            }
        }
        set_subio_labels(model, &[], &mut annotations);
        Ok(annotations)
    }
}

#[derive(Debug, Clone)]
pub struct ProfileSummary {
    pub max: Duration,
    pub sum: Duration,
    pub accel_sum: Duration,
    pub entire: Duration,
    pub iters: usize,
}

#[derive(Debug, Clone)]
pub struct MemorySummary {
    pub max: usize,
    pub max_reached_by_node: NodeQId,
}
