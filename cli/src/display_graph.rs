use format::Row;
use std::collections::HashMap;
use tfdeploy;
use tfdeploy::analyser::Analyser;
use tfdeploy::tfpb::graph::GraphDef;
use OutputParameters;
use Result as CliResult;

#[derive(Serialize)]
pub struct Edge {
    pub id: usize,
    pub src_node_id: usize,
    pub src_node_output: usize,
    pub dst_node_id: usize,
    pub dst_node_input: usize,
    pub main: bool,
    pub label: Option<String>,
}

#[derive(Serialize)]
pub struct Node {
    pub id: usize,
    pub name: String,
    pub op: String,
    pub attrs: Vec<(String, String)>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

#[derive(Serialize)]
pub struct DisplayGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl DisplayGraph {
    pub fn render(&self, params: &OutputParameters) -> CliResult<()> {
        if params.web {
            ::web::open_web(&self, params)
        } else {
            self.render_console(params)
        }
    }

    pub fn render_console(&self, params: &OutputParameters) -> CliResult<()> {
        use colored::Colorize;
        for node in &self.nodes {
            if node.op == "Const" && !params.konst {
                continue;
            }
            ::format::print_box(
                &node.id.to_string(),
                &node.op,
                &node.name,
                &[] as &[String],
                vec![
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
                ],
            );
        }
        Ok(())
    }

    pub fn from_nodes(tfnodes: &[&tfdeploy::Node]) -> CliResult<DisplayGraph> {
        let mut nodes:Vec<Node> = vec!();
        let mut edges = vec!();
        for n in tfnodes {
            let mut incoming = vec!();
            for (ix, i) in n.inputs.iter().enumerate() {
                let edge = Edge {
                    id: edges.len(),
                    src_node_id: i.0,
                    src_node_output: i.1.unwrap_or(0),
                    dst_node_id: n.id,
                    dst_node_input: ix,
                    main: ix == 0,
                    label: None,
                };
                nodes[i.0].outputs.push(edges.len());
                incoming.push(edges.len());
                edges.push(edge);
            }
            let dnode = Node {
                id: n.id,
                name: n.name.clone(),
                op: n.op_name.clone(),
                attrs: vec![],
                inputs: incoming,
                outputs: vec!()
            };
            nodes.push(dnode);
        };
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
            let index:HashMap<(usize, usize, usize, usize), usize> = self.edges.iter().enumerate().map(|(ix, edge)| {
                ( (edge.src_node_id, edge.src_node_output, edge.dst_node_id, edge.dst_node_input), ix)
            }).collect();
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
