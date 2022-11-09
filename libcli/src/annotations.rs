use ansi_term::Style;
use std::collections::HashMap;
#[allow(unused_imports)]
use std::convert::TryFrom;
use std::time::Duration;
use tract_core::internal::*;
use tract_itertools::izip;
use tract_itertools::Itertools;
#[cfg(feature = "onnx")]
use tract_onnx::pb::ModelProto;
#[cfg(feature = "tf")]
use tract_tensorflow::tfpb::tensorflow::GraphDef;

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
                    .iter()
                    .find(|(name, _)| name == &*path[0].1)
                    .map(|(_, sub)| *sub)
            }
        }
        scope(&self.0, model)
    }
}

#[derive(Debug, Default, Clone)]
pub struct NodeTags {
    pub cost: Vec<(Cost, TDim)>,
    pub style: Option<Style>,
    pub labels: Vec<String>,
    pub sections: Vec<Vec<String>>,
    pub profile: Option<Duration>,
    pub model_input: Option<String>,
    pub model_output: Option<String>,
    pub outlet_labels: Vec<Vec<String>>,
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
        let profile = self.profile.unwrap_or_default() + other.profile.unwrap_or_default();
        let profile = if profile != Duration::default() { Some(profile) } else { None };
        let style = self.style.or(other.style);
        let labels = self.labels.iter().chain(other.labels.iter()).cloned().collect();
        let sections = self.sections.iter().chain(other.sections.iter()).cloned().collect();
        let model_input = self.model_input.clone().or_else(|| other.model_input.clone());
        let model_output = self.model_output.clone().or_else(|| other.model_output.clone());
        let outlet_labels = izip!(&self.outlet_labels, &other.outlet_labels)
            .map(|(s, o)| s.iter().chain(o.iter()).cloned().collect())
            .collect();
        NodeTags {
            cost,
            profile,
            style,
            labels,
            sections,
            model_input,
            model_output,
            outlet_labels,
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
    style: None,
    labels: Vec::new(),
    sections: Vec::new(),
    profile: None,
    model_output: None,
    model_input: None,
    outlet_labels: Vec::new(),
};

#[derive(Debug, Clone, Default)]
pub struct Annotations {
    pub tags: HashMap<NodeQId, NodeTags>,
    pub profile_summary: Option<ProfileSummary>,
}

impl Annotations {
    pub fn node_mut(&mut self, qid: NodeQId) -> &mut NodeTags {
        self.tags.entry(qid).or_default()
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
    pub entire: Duration,
    pub iters: usize,
}
