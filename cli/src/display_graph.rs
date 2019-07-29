use crate::CliResult;
use crate::SomeGraphDef;
use ansi_term::Color::*;
use ansi_term::Style;
use itertools::Itertools;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::sync::Arc;
use tract_core::internal::*;
#[cfg(feature = "onnx")]
use tract_onnx::pb::ModelProto;
#[cfg(feature = "tf")]
use tract_tensorflow::tfpb::graph::GraphDef;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DisplayOptions {
    pub konst: bool,
    pub quiet: bool,
    pub natural_order: bool,
    pub debug_op: bool,
    pub node_ids: Option<Vec<usize>>,
    pub op_name: Option<String>,
    pub node_name: Option<String>,
    pub successors: Option<usize>,
}

impl DisplayOptions {
    pub fn filter(&self, model: &Model, node_id: usize) -> CliResult<bool> {
        if let Some(nodes) = self.node_ids.as_ref() {
            return Ok(nodes.contains(&node_id));
        }
        if let Some(node_name) = self.node_name.as_ref() {
            return Ok(model.node_name(node_id).starts_with(&*node_name));
        }
        if let Some(op_name) = self.op_name.as_ref() {
            return Ok(model.node_op(node_id).name().starts_with(op_name));
        }
        if let Some(successor) = self.successors {
            return Ok(model.node_inputs(node_id).iter().any(|i| i.node == successor));
        }
        Ok(model.node_op(node_id).name() != "Const" || self.konst)
    }
}

#[derive(Debug, Clone)]
pub struct DisplayGraph<'a> {
    model: &'a Model,
    pub options: Arc<DisplayOptions>,
    node_color: HashMap<usize, Style>,
    node_labels: HashMap<usize, Vec<String>>,
    node_sections: HashMap<usize, Vec<Vec<String>>>,
    node_nested_graphs: HashMap<usize, Vec<(String, DisplayGraph<'a>)>>,
}

impl<'a> DisplayGraph<'a> {
    pub fn render(&self) -> CliResult<()> {
        self.render_prefixed("")
    }

    pub fn render_prefixed(&self, prefix: &str) -> CliResult<()> {
        if self.options.quiet {
            return Ok(());
        }
        let model = self.model.borrow();
        let node_ids = if self.options.natural_order {
            (0..model.nodes_len()).collect()
        } else {
            model.eval_order()?
        };
        for node in node_ids {
            if self.options.filter(model, node)? {
                self.render_node_prefixed(node, prefix)?
            }
        }
        Ok(())
    }

    pub fn render_node(&self, node_id: usize) -> CliResult<()> {
        self.render_node_prefixed(node_id, "")
    }

    pub fn render_node_prefixed(&self, node_id: usize, prefix: &str) -> CliResult<()> {
        let model = self.model.borrow();
        let name_color = self.node_color.get(&node_id).cloned().unwrap_or(White.into());
        let node_name = model.node_name(node_id);
        let node_op_name = model.node_op(node_id).name();
        println!(
            "{}{} {} {}",
            prefix,
            White.bold().paint(format!("{}", node_id)),
            (if node_name == "UnimplementedOp" { Red.bold() } else { Blue.bold() })
                .paint(node_op_name),
            name_color.italic().paint(node_name)
        );
        for label in self.node_labels.get(&node_id).unwrap_or(&vec![]).iter() {
            println!("{}  * {}", prefix, label);
        }
        if model.node_control_inputs(node_id).len() > 0 {
            println!("{}  * control nodes: {}", prefix, model.node_control_inputs(node_id).iter().join(", "));
        }
        for (ix, i) in model.node_inputs(node_id).iter().enumerate() {
            let star = if ix == 0 { '*' } else { ' ' };
            println!(
                "{}  {} input fact  #{}: {} {:?}",
                prefix,
                star,
                ix,
                White.bold().paint(format!("{:?}", i)),
                self.model.borrow().outlet_tensorfact(*i)
            );
        }
        for ix in 0..model.node_output_count(node_id) {
            let star = if ix == 0 { '*' } else { ' ' };
            let io = if let Some(id) = self
                .model
                .borrow()
                .input_outlets()
                .iter()
                .position(|n| n.node == node_id && n.slot == ix)
            {
                Cyan.bold().paint(format!("MODEL INPUT #{}", id)).to_string()
            } else if let Some(id) = self
                .model
                .borrow()
                .output_outlets()
                .iter()
                .position(|n| n.node == node_id && n.slot == ix)
            {
                Yellow.bold().paint(format!("MODEL OUTPUT #{}", id)).to_string()
            } else {
                "".to_string()
            };
            let outlet = OutletId::new(node_id, ix);
            let fact = model.outlet_tensorfact(outlet);
            let successors = model.outlet_successors(outlet);
            println!(
                "{}  {} output fact #{}: {:?} {} {}",
                prefix,
                star,
                format!("{:?}", ix),
                fact,
                White.bold().paint(successors.iter().map(|s| format!("{:?}", s)).join(" ")),
                io
            );
        }
        for info in model.node_op(node_id).info()? {
            println!("{}  * {}", prefix, info);
        }
        if self.options.debug_op {
            println!("{}  * {:?}", prefix, model.node_op(node_id));
        }
        if let Some(node_sections) = self.node_sections.get(&node_id) {
            for section in node_sections {
                if section.is_empty() {
                    continue;
                }
                println!("{}  * {}", prefix, section[0]);
                for s in &section[1..] {
                    println!("{}    {}", prefix, s);
                }
            }
        }
        for (label, sub) in self.node_nested_graphs.get(&node_id).unwrap_or(&vec![]) {
            sub.render_prefixed(&format!(" {}{}.{} >> ", prefix, model.node_name(node_id), label))?
        }
        Ok(())
    }

    pub fn from_model_and_options(
        model: &'a Model,
        options: Arc<DisplayOptions>,
    ) -> CliResult<DisplayGraph<'a>> {
        let mut node_nested_graphs = HashMap::new();
        for n in 0..model.nodes_len() {
            let subs = model.node_op(n).nested_models();
            if subs.len() > 0 {
                node_nested_graphs.insert(
                    n,
                    subs.into_iter()
                        .map(|(label, sub)| {
                            Ok((
                                label.into_owned(),
                                Self::from_model_and_options(sub, Arc::clone(&options))?,
                            ))
                        })
                        .collect::<CliResult<_>>()?,
                );
            }
        }
        Ok(DisplayGraph {
            model,
            options,
            node_color: HashMap::new(),
            node_labels: HashMap::new(),
            node_sections: HashMap::new(),
            node_nested_graphs,
        })
    }

    pub fn with_graph_def(self, graph_def: &SomeGraphDef) -> CliResult<DisplayGraph<'a>> {
        match graph_def {
            SomeGraphDef::NoGraphDef => Ok(self),
            #[cfg(feature = "kaldi")]
            SomeGraphDef::Kaldi(kaldi) => self.with_kaldi(kaldi),
            #[cfg(feature = "onnx")]
            SomeGraphDef::Onnx(onnx, _) => self.with_onnx_model(onnx),
            #[cfg(feature = "tf")]
            SomeGraphDef::Tf(tf) => self.with_tf_graph_def(tf),
        }
    }

    pub fn set_node_color<S: Into<Style>>(&mut self, id: usize, color: S) -> CliResult<()> {
        self.node_color.insert(id, color.into());
        Ok(())
    }

    pub fn add_node_label<S: Into<String>>(&mut self, id: usize, label: S) -> CliResult<()> {
        self.node_labels.entry(id).or_insert(vec![]).push(label.into());
        Ok(())
    }

    pub fn add_node_section(&mut self, id: usize, section: Vec<String>) -> CliResult<()> {
        self.node_sections.entry(id).or_insert(vec![]).push(section);
        Ok(())
    }

    #[cfg(feature = "kaldi")]
    pub fn with_kaldi(
        mut self,
        proto_model: &tract_kaldi::KaldiProtoModel,
    ) -> CliResult<DisplayGraph<'a>> {
        use tract_kaldi::model::NodeLine;
        let bold = Style::new().bold();
        for (name, proto_node) in &proto_model.config_lines.nodes {
            if let Ok(node_id) = self.model.borrow().node_id_by_name(&*name) {
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
                self.add_node_section(node_id, vs)?;
            }
        }
        Ok(self)
    }

    #[cfg(feature = "tf")]
    pub fn with_tf_graph_def(mut self, graph_def: &GraphDef) -> CliResult<DisplayGraph<'a>> {
        let bold = Style::new().bold();
        for gnode in graph_def.get_node().iter() {
            if let Ok(node_id) = self.model.borrow().node_id_by_name(gnode.get_name()) {
                let mut v = vec![];
                for a in gnode.get_attr().iter() {
                    let value = if a.1.has_tensor() {
                        format!("{:?}", a.1.get_tensor())
                    } else {
                        format!("{:?}", a.1)
                    };
                    v.push(format!("Attr {}: {:.240}", bold.paint(a.0), value));
                }
                self.add_node_section(node_id, v)?;
            }
        }
        Ok(self)
    }

    #[cfg(feature = "onnx")]
    pub fn with_onnx_model(mut self, model_proto: &ModelProto) -> CliResult<DisplayGraph<'a>> {
        let bold = Style::new().bold();
        for gnode in model_proto.get_graph().get_node().iter() {
            let mut node_name = gnode.get_name();
            if node_name == "" && gnode.get_output().len() > 0 {
                node_name = &gnode.get_output()[0];
            }
            if let Ok(id) = self.model.borrow().node_id_by_name(node_name) {
                let mut v = vec![];
                for a in gnode.get_attribute().iter() {
                    let value = if a.has_t() {
                        format!("{:?}", Tensor::try_from(a.get_t())?)
                    } else {
                        format!("{:?}", a)
                    };
                    v.push(format!("Attr {}: {:.240}", bold.paint(a.get_name()), value));
                }
                self.add_node_section(id, v)?;
            }
        }
        Ok(self)
    }
}
