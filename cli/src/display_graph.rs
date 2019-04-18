use crate::format::Row;
use crate::CliResult;
use crate::SomeGraphDef;
use ansi_term::Color::*;
use ansi_term::Style;
use std::borrow::Borrow;
use std::collections::HashMap;
use tract_core::prelude::{Model, Node, TensorInfo };
use tract_core::Tractify;
#[cfg(feature = "onnx")]
use tract_onnx::pb::ModelProto;
#[cfg(feature = "tf")]
use tract_tensorflow::tfpb::graph::GraphDef;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DisplayOptions {
    pub konst: bool,
    pub quiet: bool,
    pub debug_op: bool,
    pub node_ids: Option<Vec<usize>>,
    pub op_name: Option<String>,
    pub node_name: Option<String>,
    pub successors: Option<usize>,
}

impl DisplayOptions {
    pub fn filter<TI: TensorInfo>(&self, _model: &Model<TI>, node: &Node<TI>) -> CliResult<bool> {
        if let Some(nodes) = self.node_ids.as_ref() {
            return Ok(nodes.contains(&node.id))
        }
        if let Some(node_name) = self.node_name.as_ref() {
            return Ok(node.name.starts_with(&*node_name));
        }
        if let Some(op_name) = self.op_name.as_ref() {
            return Ok(op_name == &node.op().name());
        }
        if let Some(successor) = self.successors {
            return Ok(node.inputs.iter().any(|i| i.node == successor))

        }
        Ok(node.op().name() != "Const" || self.konst)
    }
}

#[derive(Debug, Clone)]
pub struct DisplayGraph<TI: TensorInfo, M: Borrow<Model<TI>>> {
    model: M,
    pub options: DisplayOptions,
    node_labels: HashMap<usize, Vec<String>>,
    node_sections: HashMap<usize, Vec<Vec<Row>>>,
    _bloody_baron: ::std::marker::PhantomData<TI>,
}

impl<TI: TensorInfo, M: Borrow<Model<TI>>> DisplayGraph<TI, M> {
    pub fn render(&self) -> CliResult<()> {
        if self.options.quiet {
            return Ok(());
        }
        let model = self.model.borrow();
        let node_ids = ::tract_core::model::eval_order(&model)?;
        for node in node_ids {
            let node = &model.nodes()[node];
            if self.options.filter(model, node)? {
                self.render_node(&node)?
            }
        }
        Ok(())
    }

    pub fn render_node(&self, node: &Node<TI>) -> CliResult<()> {
        let bold = Style::new().bold();
        let mut sections: Vec<Vec<Row>> = vec![];
        if let Some(id) =
            self.model.borrow().input_outlets()?.iter().position(|n| n.node == node.id)
        {
            sections.push(vec![Row::Simple(
                Yellow.bold().paint(format!("MODEL INPUT {}", id)).to_string(),
            )]);
        }
        sections.push(
            node.inputs
                .iter()
                .enumerate()
                .map(|(ix, a)| {
                    Ok(Row::Double(
                        format!(
                            "Input {}: Node #{}/{}",
                            bold.paint(format!("{}", ix)),
                            bold.paint(format!("{}", a.node)),
                            bold.paint(format!("{}", a.slot)),
                        ),
                        format!("{:?}", self.model.borrow().outlet_fact(*a)?),
                    ))
                })
                .collect::<CliResult<_>>()?,
        );
        sections.push(
            node.outputs
                .iter()
                .enumerate()
                .map(|(ix, outlet)| {
                    if let Some(pos) = self
                        .model
                        .borrow()
                        .output_outlets()
                        .unwrap()
                        .iter()
                        .position(|&o| o == ::tract_core::model::OutletId::new(node.id, ix))
                    {
                        Row::Double(
                            format!("Output {}:", bold.paint(ix.to_string())),
                            format!("{:?} {} #{}", outlet.fact, bold.paint("Model output"), pos),
                        )
                    } else {
                        Row::Double(
                            format!("Output {}:", bold.paint(ix.to_string())),
                            format!("{:?}", outlet.fact),
                        )
                    }
                })
                .collect(),
        );
        if let Some(info) = node.op().info()? {
            sections.push(vec![Row::Simple(info)])
        }
        if self.options.debug_op {
            sections.push(vec![Row::Simple(format!("{:?}", node.op))]);
        }
        if let Some(node_sections) = self.node_sections.get(&node.id) {
            for s in node_sections {
                sections.push(s.clone());
            }
        }
        crate::format::print_box(
            &node.id.to_string(),
            &node.op.name(),
            &node.name,
            self.node_labels.get(&node.id).map(|v| v.as_slice()).unwrap_or(&[]),
            sections,
        );
        Ok(())
    }

    pub fn from_model_and_options(
        model: M,
        options: DisplayOptions,
    ) -> CliResult<DisplayGraph<TI, M>> {
        Ok(DisplayGraph {
            model,
            options,
            node_labels: HashMap::new(),
            node_sections: HashMap::new(),
            _bloody_baron: std::marker::PhantomData,
        })
    }

    pub fn with_graph_def(self, graph_def: &SomeGraphDef) -> CliResult<DisplayGraph<TI, M>> {
        match graph_def {
            #[cfg(feature = "tf")]
            SomeGraphDef::Tf(tf) => self.with_tf_graph_def(tf),
            #[cfg(feature = "onnx")]
            SomeGraphDef::Onnx(onnx) => self.with_onnx_model(onnx),
            SomeGraphDef::_NoGraph => unreachable!(),
        }
    }

    pub fn add_node_label(&mut self, id: usize, label: String) -> CliResult<()> {
        self.node_labels.entry(id).or_insert(vec![]).push(label);
        Ok(())
    }

    pub fn add_node_section(&mut self, id: usize, section: Vec<Row>) -> CliResult<()> {
        self.node_sections.entry(id).or_insert(vec![]).push(section);
        Ok(())
    }

    #[cfg(feature = "tf")]
    pub fn with_tf_graph_def(mut self, graph_def: &GraphDef) -> CliResult<DisplayGraph<TI, M>> {
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
                    v.push(Row::Double(format!("Attr {}:", bold.paint(a.0)), value));
                }
                self.add_node_section(node_id, v)?;
            }
        }
        Ok(self)
    }

    #[cfg(feature = "onnx")]
    pub fn with_onnx_model(mut self, model_proto: &ModelProto) -> CliResult<DisplayGraph<TI, M>> {
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
                        format!("{:?}", ::tract_core::prelude::Tensor::tractify(a.get_t())?)
                    } else {
                        format!("{:?}", a)
                    };
                    v.push(Row::Double(format!("Attr {}:", bold.paint(a.get_name())), value));
                }
                self.add_node_section(id, v)?;
            }
        }
        Ok(self)
    }
}
