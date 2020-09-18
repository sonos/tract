use crate::{CliResult, SomeGraphDef};
use ansi_term::Style;
use std::collections::HashMap;
#[allow(unused_imports)]
use std::convert::TryFrom;
use std::time::Duration;
use tract_core::internal::*;
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
            if path.len() == 0 {
                Some(model)
            } else {
                model
                    .nested_models(path[0].0)
                    .iter()
                    .find(|(name, _)| name == &*path[0].1)
                    .map(|(_, sub)| *sub)
            }
        }
        scope(&*self.0, model)
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
        let profile = self.profile.unwrap_or(Duration::default())
            + other.profile.unwrap_or(Duration::default());
        let profile = if profile != Duration::default() { Some(profile) } else { None };
        let style = self.style.or(other.style);
        let labels = self.labels.iter().chain(other.labels.iter()).cloned().collect();
        let sections = self.sections.iter().chain(other.sections.iter()).cloned().collect();
        let model_input = self.model_input.clone().or(other.model_input.clone());
        let model_output = self.model_output.clone().or(other.model_output.clone());
        NodeTags { cost, profile, style, labels, sections, model_input, model_output }
    }
}

impl<'a> std::iter::Sum<&'a NodeTags> for NodeTags {
    fn sum<I>(iter: I) -> NodeTags
    where
        I: std::iter::Iterator<Item = &'a NodeTags>,
    {
        iter.fold(EMPTY.clone(), |a, b| &a + b)
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
};

#[derive(Debug, Clone, Default)]
pub struct Annotations {
    pub tags: HashMap<NodeQId, NodeTags>,
    pub profile_summary: Option<crate::profile::ProfileSummary>,
}

impl Annotations {
    pub fn node_mut(&mut self, qid: NodeQId) -> &mut NodeTags {
        self.tags.entry(qid).or_default()
    }

    pub fn from_model(model: &dyn Model) -> CliResult<Annotations> {
        let mut annotations = Annotations::default();
        fn set_subio_labels(
            model: &dyn Model,
            prefix: &[(usize, String)],
            annotations: &mut Annotations,
        ) {
            for n in 0..model.nodes_len() {
                for (label, sub /*, ins, outs*/) in model.nested_models(n) {
                    let mut prefix: TVec<(usize, String)> = prefix.into();
                    prefix.push((n, label.to_string()));
                    set_subio_labels(sub, &*prefix, annotations);
                    /*
                    ins.into_iter().enumerate().for_each(|(ix, i)| {
                    let qid = NodeQId(prefix.clone(), ix);
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

    #[allow(unused_variables)]
    pub fn with_graph_def(
        self,
        model: &dyn Model,
        graph_def: &SomeGraphDef,
    ) -> CliResult<Annotations> {
        match graph_def {
            SomeGraphDef::NoGraphDef => Ok(self),
            #[cfg(feature = "kaldi")]
            SomeGraphDef::Kaldi(kaldi) => self.with_kaldi(model, kaldi),
            SomeGraphDef::Nnef(_) => todo!(),
            #[cfg(feature = "onnx")]
            SomeGraphDef::Onnx(onnx, _) => self.with_onnx_model(model, onnx),
            #[cfg(feature = "tf")]
            SomeGraphDef::Tf(tf) => self.with_tf_graph_def(model, tf),
        }
    }

    #[cfg(feature = "kaldi")]
    pub fn with_kaldi(
        mut self,
        model: &dyn Model,
        proto_model: &tract_kaldi::KaldiProtoModel,
    ) -> CliResult<Annotations> {
        use tract_kaldi::model::NodeLine;
        let bold = Style::new().bold();
        for (name, proto_node) in &proto_model.config_lines.nodes {
            if let Ok(node_id) = model.node_id_by_name(&*name) {
                let mut vs = vec![];
                match proto_node {
                    NodeLine::Component(compo) => {
                        let comp = &proto_model.components[&compo.component];
                        for (k, v) in &comp.attributes {
                            let value = format!("{:?}", v);
                            vs.push(format!("Attr {}: {:.240}", bold.paint(k), value));
                        }
                    }
                    _ => (),
                }
                self.node_mut(node_id.into()).sections.push(vs)
            }
        }
        Ok(self)
    }

    #[cfg(feature = "tf")]
    pub fn with_tf_graph_def(
        mut self,
        model: &dyn Model,
        graph_def: &GraphDef,
    ) -> CliResult<Annotations> {
        let bold = Style::new().bold();
        for gnode in graph_def.node.iter() {
            if let Ok(node_id) = model.node_id_by_name(&gnode.name) {
                let mut v = vec![];
                for a in gnode.attr.iter() {
                    let value = if let Some(
                        tract_tensorflow::tfpb::tensorflow::attr_value::Value::Tensor(r),
                    ) = &a.1.value
                    {
                        format!("{:?}", r)
                    } else {
                        format!("{:?}", a.1)
                    };
                    v.push(format!("Attr {}: {:.240}", bold.paint(a.0), value));
                }
                self.node_mut(node_id.into()).sections.push(v);
            }
        }
        Ok(self)
    }

    #[cfg(feature = "onnx")]
    pub fn with_onnx_model(
        mut self,
        model: &dyn Model,
        model_proto: &ModelProto,
    ) -> CliResult<Annotations> {
        let bold = Style::new().bold();
        for gnode in model_proto.graph.as_ref().unwrap().node.iter() {
            let mut node_name = &gnode.name;
            if node_name == "" && gnode.output.len() > 0 {
                node_name = &gnode.output[0];
            }
            if let Ok(id) = model.node_id_by_name(&*node_name) {
                let mut v = vec![];
                for a in gnode.attribute.iter() {
                    let value = if let Some(t) = &a.t {
                        format!("{:?}", Tensor::try_from(t)?)
                    } else {
                        format!("{:?}", a)
                    };
                    v.push(format!("Attr {}: {:.240}", bold.paint(&a.name), value));
                }
                self.node_mut(id.into()).sections.push(v);
            }
        }
        Ok(self)
    }

    pub fn extract_costs(&mut self, model: &dyn Model) -> CliResult<()> {
        fn extract_costs_rec(
            annotations: &mut Annotations,
            model: &dyn Model,
            prefix: &[(usize, String)],
            multiplier: TDim,
        ) -> CliResult<()> {
            if let Some(model) = model.downcast_ref::<TypedModel>() {
                for node_id in 0..model.nodes().len() {
                    let inputs = model.node_input_facts(node_id)?;
                    let cost = model.node(node_id).op.cost(&*inputs)?;
                    annotations.node_mut(NodeQId(prefix.into(), node_id)).cost = cost
                        .into_iter()
                        .map(|(k, v)| {
                            (k, if k.is_compute() { v.maybe_mul(&multiplier).unwrap() } else { v })
                        })
                        .collect();

                    let nested_subs = model.nested_models(node_id);
                    let nested_multis =
                        (model as &dyn Model).nested_models_iters(node_id, &*inputs);
                    for ((name, sub), multi) in nested_subs.iter().zip(nested_multis.iter()) {
                        let mut prefix: TVec<_> = prefix.into();
                        prefix.push((node_id, name.to_string()));
                        extract_costs_rec(
                            annotations,
                            *sub,
                            &*prefix,
                            multiplier.maybe_mul(multi.as_ref().unwrap()).unwrap(),
                        )?;
                    }
                }
            }
            Ok(())
        }
        extract_costs_rec(self, model, &[], 1.into())
    }
}
