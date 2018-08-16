use format::Row;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fs;
use tfdeploy;
use tfdeploy::analyser::Analyser;
use tfdeploy::tfpb::graph::GraphDef;
use OutputParameters;
use Result as CliResult;

#[derive(Debug, Serialize)]
pub struct Edge {
    pub id: usize,
    pub src_node_id: usize,
    pub src_node_output: usize,
    pub dst_node_id: usize,
    pub dst_node_input: usize,
    pub main: bool,
    pub label: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Node {
    pub id: usize,
    pub name: String,
    pub op: String,
    pub label: Option<String>,
    pub more_lines: Vec<String>,
    pub attrs: Vec<(String, String)>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub hidden: bool,
}

#[derive(Debug, Serialize)]
pub struct DisplayGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl DisplayGraph {
    pub fn render(&self, params: &OutputParameters) -> CliResult<()> {
        if params.web {
            ::web::open_web(&self, params)
        } else if let Some(json) = params.json.as_ref() {
            ::serde_json::to_writer(fs::File::create(json)?, self)?;
            Ok(())
        } else {
            self.render_console(params)
        }
    }

    pub fn render_console(&self, params: &OutputParameters) -> CliResult<()> {
        if let Some(node) = params.node_id {
            let node = &self.nodes[node];
            return self.render_node(node, params)
        }
        if let Some(node_name) = &params.node_name {
            if let Some(node) = &self.nodes.iter().find(|n| n.name == &**node_name) {
                return self.render_node(node, params)
            } else {
                return Ok(())
            }
        }
        for node in &self.nodes {
            if node.op == "Const" && !params.konst {
                continue;
            }
            if node.hidden {
                continue;
            }
            if params.op_name.as_ref().map(|name| name != &*node.op).unwrap_or(false) {
                continue;
            }
            if params.successors.as_ref().map(|id| {
                !node.inputs.iter().any(|i| self.edges[*i].src_node_id == *id)
            }).unwrap_or(false) {
                continue;
            }
            self.render_node(&node, params)?
        }
        Ok(())
    }
    pub fn render_node(&self, node: &Node, params:&OutputParameters) -> CliResult<()> {
        use colored::Colorize;
        let output_ports: HashMap<usize, Option<String>> = node
            .outputs
            .iter()
            .map(|edge| {
                let edge = &self.edges[*edge];
                (edge.src_node_output, edge.label.clone())
            })
            .collect();
        let mut sections = vec![
            node.attrs
                .iter()
                .map(|a| Row::Double(format!("Attribute {}:", a.0.bold()), a.1.clone()))
                .collect(),
            node.inputs
                .iter()
                .enumerate()
                .map(|(ix, a)| {
                    let edge = &self.edges[*a];
                    Row::Double(
                        if edge.src_node_output == 0 {
                            format!(
                                "Input {}: Node #{}",
                                ix.to_string().bold(),
                                edge.src_node_id.to_string().bold()
                            )
                        } else {
                            format!(
                                "Input {}: Node #{}/{}",
                                ix.to_string().bold(),
                                edge.src_node_id.to_string().bold(),
                                edge.src_node_output.to_string().bold()
                            )
                        },
                        edge.label.clone().unwrap_or_else(|| "".to_string()),
                    )
                })
                .collect(),
            (0..output_ports.len())
                .map(|ix| {
                    let edge = &output_ports[&ix];
                    Row::Double(
                        format!("Output {}:", ix.to_string().bold()),
                        edge.clone().unwrap_or_else(|| "".to_string()),
                    )
                })
                .collect(),
        ];
        if node.more_lines.len() > 0 {
            sections.push(
                node.more_lines
                    .iter()
                    .map(|s| Row::Simple(s.clone()))
                    .collect(),
            );
        }
        ::format::print_box(
            &node.id.to_string(),
            &node.op,
            &node.name,
            &*node.label.as_ref().map(|a| vec![a]).unwrap_or(vec![]),
            sections,
        );
        Ok(())
    }

    pub fn from_nodes(tfnodes: &[impl Borrow<tfdeploy::Node>]) -> CliResult<DisplayGraph> {
        let mut nodes: Vec<Node> = tfnodes
            .iter()
            .map(|n| Node {
                id: n.borrow().id,
                name: n.borrow().name.clone(),
                op: n.borrow().op_name.clone(),
                label: None,
                more_lines: vec![],
                attrs: vec![],
                inputs: vec![],
                outputs: vec![],
                hidden: false,
            })
            .collect();
        let mut edges = vec![];
        for node in tfnodes.iter() {
            for (ix, input) in node.borrow().inputs.iter().enumerate() {
                let edge = Edge {
                    id: edges.len(),
                    src_node_id: input.0,
                    src_node_output: input.1,
                    dst_node_id: node.borrow().id,
                    dst_node_input: ix,
                    main: ix == 0,
                    label: tfnodes[input.0]
                        .borrow()
                        .op()
                        .const_value()
                        .map(|v| format!("Const {:?}", v)),
                };
                nodes[edge.src_node_id].outputs.push(edges.len());
                nodes[node.borrow().id].inputs.push(edges.len());
                edges.push(edge);
            }
        }
        Ok(DisplayGraph { nodes, edges })
    }

    pub fn with_graph_def(mut self, graph_def: &GraphDef) -> CliResult<DisplayGraph> {
        let index_to_graph_def: HashMap<String, usize> =
            self.nodes.iter().map(|n| (n.name.clone(), n.id)).collect();
        for gnode in graph_def.get_node().iter() {
            if let Some(node_id) = index_to_graph_def.get(gnode.get_name()) {
                for a in gnode.get_attr().iter() {
                    let value = if a.1.has_tensor() {
                        format!("{:?}", tfdeploy::tensor::Tensor::from_pb(a.1.get_tensor())?)
                    } else {
                        format!("{:?}", a.1)
                    };
                    self.nodes[*node_id].attrs.push((a.0.to_owned(), value));
                }
                self.nodes[*node_id].attrs.sort();
            }
        }
        Ok(self)
    }

    pub fn with_analyser(mut self, analyser: &Analyser) -> CliResult<DisplayGraph> {
        {
            let index: HashMap<(usize, usize, usize, usize), usize> = self
                .edges
                .iter()
                .enumerate()
                .map(|(ix, edge)| {
                    (
                        (
                            edge.src_node_id,
                            edge.src_node_output,
                            edge.dst_node_id,
                            edge.dst_node_input,
                        ),
                        ix,
                    )
                })
                .collect();
            for an_edge in &analyser.edges {
                if let (Some(from_node), Some(to_node)) = (an_edge.from_node, an_edge.to_node) {
                    let key = (from_node, an_edge.from_out, to_node, an_edge.to_input);
                    self.edges[index[&key]].label = Some(format!("{:?}", an_edge.fact));
                }
            }
        }
        Ok(self)
    }
}
