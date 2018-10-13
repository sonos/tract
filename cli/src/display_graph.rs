use format::Row;
use std::borrow::Borrow;
use std::collections::HashMap;
use tfdeploy::{ Model, Node };
use tfdeploy_tf::tfpb::graph::GraphDef;
use tfdeploy_onnx::pb::ModelProto;
use CliResult;
use SomeGraphDef;

#[derive(Debug, Clone, Default)]
pub struct DisplayOptions {
    pub konst: bool,
    pub quiet: bool,
    pub node_ids: Option<Vec<usize>>,
    pub op_name: Option<String>,
    pub node_name: Option<String>,
    pub successors: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct DisplayGraph<M:Borrow<Model>> {
    model: M,
    pub options: DisplayOptions,
    node_labels: HashMap<usize, Vec<String>>,
    node_sections: HashMap<usize, Vec<Vec<Row>>>,
}

impl<M:Borrow<Model>> DisplayGraph<M> {
    pub fn render(&self) -> CliResult<()> {
        if self.options.quiet {
            return Ok(())
        }
        let model = self.model.borrow();
        if let Some(nodes) = &self.options.node_ids {
            for &node in nodes {
                let node = &model.nodes()[node];
                self.render_node(node)?;
            }
            return Ok(())
        }
        if let Some(node_name) = &self.options.node_name {
            if let Ok(node) = model.node_by_name(node_name) {
                return self.render_node(node);
            } else {
                return Ok(());
            }
        }
        for node in model.nodes().iter() {
            if node.op().name() == "Const" && !self.options.konst {
                continue;
            }
            /*
            if node.hidden {
                continue;
            }
            */
            if self.options
                .op_name
                .as_ref()
                .map(|name| name != &*node.op.name())
                .unwrap_or(false)
            {
                continue;
            }
            if self.options
                .successors
                .as_ref()
                .map(|id| !node.inputs.iter().any(|i| i.slot == *id))
                .unwrap_or(false)
            {
                continue;
            }
            self.render_node(&node)?
        }
        Ok(())
    }

    pub fn render_node(&self, node: &Node) -> CliResult<()> {
        use colored::Colorize;
        // node output are not ordered by slot number
        /*
        let output_ports: HashMap<usize, String> = node
            .outputs
            .iter()
            .map(|edge| {
                let edge = &self.edges[*edge];
                (edge.src.slot, edge.label.clone().unwrap_or_else(|| "".to_string()))
            })
            .collect();
        let mut output_ports:Vec<(usize, String)> = output_ports.into_iter().collect();
        output_ports.sort();
        */
        let sections = vec![
            /*
            node.attrs
                .iter()
                .map(|a| Row::Double(format!("Attribute {}:", a.0.bold()), a.1.clone()))
                .collect(),
            */
            node.inputs
                .iter()
                .enumerate()
                .map(|(ix, a)| {
                    Ok(Row::Double(
                        format!(
                            "Input {}: Node #{}/{}",
                            ix.to_string().bold(),
                            a.node.to_string().bold(),
                            a.slot.to_string().bold()
                        ),
                        format!("{:?}", self.model.borrow().fact(*a)?)
                    ))
                })
                .collect::<CliResult<_>>()?,
            node.outputs.iter().enumerate()
                .map(|(ix,outlet)| {
                    if let Some(pos) = self.model.borrow().outputs().unwrap().iter().position(|&o| o==::tfdeploy::model::OutletId::new(node.id, ix)) {
                        Row::Double(
                            format!("Output {}:", ix.to_string().bold()),
                            format!("{:?} {} #{}", outlet.fact, "Model output".bold(),
                                    pos)
                        )
                    } else {
                        Row::Double(
                            format!("Output {}:", ix.to_string().bold()),
                            format!("{:?}", outlet.fact)
                        )
                    }
                })
                .collect(),
        ];
                /*
        if node.more_lines.len() > 0 {
            sections.push(
                node.more_lines
                    .iter()
                    .map(|s| Row::Simple(s.clone()))
                    .collect(),
            );
        }
        */
        ::format::print_box(
            &node.id.to_string(),
            &node.op.name(),
            &node.name,
            self.node_labels.get(&node.id).map(|v|v.as_slice()).unwrap_or(&[]),
            sections,
        );
        Ok(())
    }

    pub fn from_model_and_options(model: M, options: DisplayOptions) -> CliResult<DisplayGraph<M>> {
        Ok(DisplayGraph { model, options, node_labels: HashMap::new(), node_sections: HashMap::new() })
    }

    pub fn with_graph_def(self, graph_def: &SomeGraphDef) -> CliResult<DisplayGraph<M>> {
        match graph_def {
            SomeGraphDef::Tf(tf) => self.with_tf_graph_def(tf),
            SomeGraphDef::Onnx(onnx) => self.with_onnx_model(onnx),
        }
    }

    pub fn add_node_label(&mut self, id: usize, label: String) -> CliResult<()> {
        self.node_labels.entry(id).or_insert(vec!()).push(label);
            Ok(())
    }

    pub fn with_tf_graph_def(mut self, graph_def: &GraphDef) -> CliResult<DisplayGraph<M>> {
        /*
        let index_to_graph_def: HashMap<String, usize> =
            self.nodes.iter().map(|n| (n.name.clone(), n.id)).collect();
        for gnode in graph_def.get_node().iter() {
            if let Some(node_id) = index_to_graph_def.get(gnode.get_name()) {
                for a in gnode.get_attr().iter() {
                    let value = if a.1.has_tensor() {
                        format!("{:?}", tfdeploy::tensor::Tensor::tfd_from(a.1.get_tensor())?)
                    } else {
                        format!("{:?}", a.1)
                    };
                    self.nodes[*node_id].attrs.push((a.0.to_owned(), value));
                }
                self.nodes[*node_id].attrs.sort();
            }
        }
        */
        Ok(self)
    }

    pub fn with_onnx_model(mut self, model_proto: &ModelProto) -> CliResult<DisplayGraph<M>> {
        /*
        let index_to_graph_def: HashMap<String, usize> =
            self.nodes.iter().map(|n| (n.name.clone(), n.id)).collect();
        for gnode in model_proto.get_graph().get_node().iter() {
            let mut node_name = gnode.get_name();
            if node_name == "" && gnode.get_output().len() > 0 {
                node_name = &gnode.get_output()[0];
            }
            if let Some(node_id) = index_to_graph_def.get(node_name) {
                for a in gnode.get_attribute().iter() {
                    let value = if a.has_t() {
                        format!("{:?}", tfdeploy::tensor::Tensor::tfd_from(a.get_t())?)
                    } else {
                        format!("{:?}", a)
                    };
                    self.nodes[*node_id].attrs.push((a.get_name().to_owned(), value));
                }
                self.nodes[*node_id].attrs.sort();
            }
        }
        */
        Ok(self)
    }
}
