use crate::CliResult;
use crate::SomeGraphDef;
use ansi_term::Color::*;
use ansi_term::Style;
use itertools::Itertools;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt::{Debug, Display};
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
    pub fn filter<TI, O>(&self, _model: &Model<TI, O>, node: &BaseNode<TI, O>) -> CliResult<bool>
    where
        TI: TensorInfo,
        O: AsRef<Op> + AsMut<Op> + Display + Debug,
    {
        if let Some(nodes) = self.node_ids.as_ref() {
            return Ok(nodes.contains(&node.id));
        }
        if let Some(node_name) = self.node_name.as_ref() {
            return Ok(node.name.starts_with(&*node_name));
        }
        if let Some(op_name) = self.op_name.as_ref() {
            return Ok(node.op().name().starts_with(op_name));
        }
        if let Some(successor) = self.successors {
            return Ok(node.inputs.iter().any(|i| i.node == successor));
        }
        Ok(node.op().name() != "Const" || self.konst)
    }
}

#[derive(Debug, Clone)]
pub struct DisplayGraph<TI, O, M>
where
    TI: TensorInfo,
    O: AsRef<Op> + AsMut<Op> + Display + Debug,
    M: Borrow<Model<TI, O>>,
{
    model: M,
    pub options: DisplayOptions,
    node_color: HashMap<usize, Style>,
    node_labels: HashMap<usize, Vec<String>>,
    node_sections: HashMap<usize, Vec<Vec<String>>>,
    _bloody_baron: ::std::marker::PhantomData<(TI, O)>,
}

impl<TI, O, M> DisplayGraph<TI, O, M>
where
    TI: TensorInfo,
    O: AsRef<Op> + AsMut<Op> + Display + Debug,
    M: Borrow<Model<TI, O>>,
{
    pub fn render(&self) -> CliResult<()> {
        if self.options.quiet {
            return Ok(());
        }
        let model = self.model.borrow();
        let node_ids = if self.options.natural_order {
            (0..model.nodes().len()).collect()
        } else {
            ::tract_core::model::eval_order(&model)?
        };
        for node in node_ids {
            let node = &model.nodes()[node];
            if self.options.filter(model, node)? {
                self.render_node(&node)?
            }
        }
        Ok(())
    }

    pub fn render_node(&self, node: &BaseNode<TI, O>) -> CliResult<()> {
        let name_color = self.node_color.get(&node.id).cloned().unwrap_or(White.into());
        println!(
            "{} {} {}",
            White.bold().paint(format!("{}", node.id)),
            (if node.op_is::<tract_core::ops::unimpl::UnimplementedOp>() { Red.bold() } else { Blue.bold() }).paint(node.op().name()),
            name_color.italic().paint(&node.name)
        );
        for label in self.node_labels.get(&node.id).unwrap_or(&vec!()).iter() {
            println!("  * {}", label);
        }
        if node.control_inputs.len() > 0 {
            println!("  * control nodes: {}", node.control_inputs.iter().join(", "));
        }
        for (ix, i) in node.inputs.iter().enumerate() {
            let star = if ix == 0 { '*' } else { ' ' };
            println!(
                "  {} input fact  #{}: {:?} {:?}",
                star,
                ix,
                i,
                self.model.borrow().outlet_fact(*i)?
            );
        }
        for (ix, o) in node.outputs.iter().enumerate() {
            let star = if ix == 0 { '*' } else { ' ' };
            let io = if let Some(id) = self
                .model
                .borrow()
                .input_outlets()?
                .iter()
                .position(|n| n.node == node.id && n.slot == ix)
            {
                Cyan.bold().paint(format!("MODEL INPUT #{}", id)).to_string()
            } else if let Some(id) = self
                .model
                .borrow()
                .output_outlets()?
                .iter()
                .position(|n| n.node == node.id && n.slot == ix)
            {
                Yellow.bold().paint(format!("MODEL OUTPUT #{}", id)).to_string()
            } else {
                "".to_string()
            };
            println!("  {} output fact #{}: {:?} {}", star, format!("{:?}", ix), o.fact, io);
        }
        if let Some(info) = node.op().info()? {
            println!("  * {}", info);
        }
        if self.options.debug_op {
            println!("  * {:?}", node.op());
        }
        if let Some(node_sections) = self.node_sections.get(&node.id) {
            for section in node_sections {
                if section.is_empty() {
                    continue;
                }
                println!("  * {}", section[0]);
                for s in &section[1..] {
                    println!("    {}", s);
                }
            }
        }
        Ok(())
    }

    pub fn from_model_and_options(
        model: M,
        options: DisplayOptions,
    ) -> CliResult<DisplayGraph<TI, O, M>> {
        Ok(DisplayGraph {
            model,
            options,
            node_color: HashMap::new(),
            node_labels: HashMap::new(),
            node_sections: HashMap::new(),
            _bloody_baron: std::marker::PhantomData,
        })
    }

    pub fn with_graph_def(self, graph_def: &SomeGraphDef) -> CliResult<DisplayGraph<TI, O, M>> {
        match graph_def {
            SomeGraphDef::NoGraphDef => Ok(self),
            #[cfg(feature = "tf")]
            SomeGraphDef::Tf(tf) => self.with_tf_graph_def(tf),
            #[cfg(feature = "onnx")]
            SomeGraphDef::Onnx(onnx, _) => self.with_onnx_model(onnx),
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

    #[cfg(feature = "tf")]
    pub fn with_tf_graph_def(mut self, graph_def: &GraphDef) -> CliResult<DisplayGraph<TI, O, M>> {
        let bold = Style::new().bold();
        for gnode in graph_def.get_node().iter() {
            if let Ok(node_id) = self.model.borrow().node_by_name(gnode.get_name()).map(|n| n.id) {
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
    pub fn with_onnx_model(
        mut self,
        model_proto: &ModelProto,
    ) -> CliResult<DisplayGraph<TI, O, M>> {
        let bold = Style::new().bold();
        for gnode in model_proto.get_graph().get_node().iter() {
            let mut node_name = gnode.get_name();
            if node_name == "" && gnode.get_output().len() > 0 {
                node_name = &gnode.get_output()[0];
            }
            if let Ok(id) = self.model.borrow().node_by_name(node_name).map(|n| n.id) {
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
