#![allow(dead_code)]
use std::io;
use std::io::Write;
use std::borrow::Borrow;

use dot;
use errors::*;
use tfdeploy::Model;
use tfdeploy::analyser::*;

type Nd<'a> = (usize, &'a String, &'a String, bool);
type Ed<'a> = (usize, &'a Edge, bool);

struct Graph<'a> {
    nodes: &'a [Nd<'a>],
    edges: &'a [Ed<'a>],
}

/// An implementation of dot::Labeller for the analyser.
impl<'a> dot::Labeller<'a, Nd<'a>, Ed<'a>> for Graph<'a> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("analysed_graph").unwrap()
    }
    fn node_id(&'a self, n: &Nd) -> dot::Id<'a> {
        dot::Id::new(format!("node_{:?}", &n.0)).unwrap()
    }
    fn node_label<'b>(&'b self, n: &Nd) -> dot::LabelText<'b> {
        dot::LabelText::LabelStr(format!("[{:?}] {} ({})", n.0, n.1, n.2).into())
    }
    fn node_color<'b>(&'b self, n: &Nd) -> Option<dot::LabelText<'b>> {
        if n.3 {
            Some(dot::LabelText::LabelStr("crimson".into()))
        } else {
            None
        }
    }
    fn edge_label<'b>(&'b self, e: &Ed) -> dot::LabelText<'b> {
        use self::GenericFact::*;

        let mut label = format!(
            "[{:?}] {:?}\n{:?}\n{}",
            e.0,
            e.1.fact.datum_type,
            e.1.fact.shape,
            match e.1.fact.value {
                Any => "Any",
                Only(_) => "Only(_)",
            }
        );
        label.truncate(150);

        dot::LabelText::LabelStr(label.into())
    }
    fn edge_color<'b>(&'b self, e: &Ed) -> Option<dot::LabelText<'b>> {
        if e.2 {
            Some(dot::LabelText::LabelStr("crimson".into()))
        } else {
            None
        }
    }
}

/// An implementation of dot::GraphWalk for the analyser.
impl<'a> dot::GraphWalk<'a, Nd<'a>, Ed<'a>> for Graph<'a> {
    fn nodes(&self) -> dot::Nodes<'a, Nd> {
        self.nodes.into()
    }
    fn edges(&'a self) -> dot::Edges<'a, Ed> {
        self.edges.into()
    }
    fn source(&self, e: &Ed) -> Nd {
        match e.1.from.map(|e| e.node) {
            Some(n) => self.nodes[n].clone(),
            None => panic!("{:?}", e),
        }
    }
    fn target(&self, e: &Ed) -> Nd {
        match e.1.to_node {
            Some(n) => self.nodes[n].clone(),
            None => self.nodes[self.nodes.len() - 1].clone(),
        }
    }
}

/// Writes a DOT export of the analysed graph to a given Writer.
pub fn render_dot<W: Write>(
    analyser: &Analyser<impl Borrow<Model>>,
    red_nodes: &Vec<usize>,
    red_edges: &Vec<usize>,
    writer: &mut W,
) -> CliResult<()> {
    let output_node_name = "output".to_string();

    let mut nodes: Vec<_> = analyser
        .nodes
        .iter()
        .map(|n| (n.id, &n.name, &n.op_name, red_nodes.contains(&n.id)))
        .collect();

    // Add a special output node.
    let output_node_id = nodes.len();
    nodes.push((output_node_id, &output_node_name, &output_node_name, false));

    let edges: Vec<_> = analyser
        .edges
        .iter()
        .enumerate()
        .map(|(i, e)| (i, e, red_edges.contains(&i)))
        .collect();

    let graph = Graph {
        nodes: nodes.as_slice(),
        edges: edges.as_slice(),
    };

    dot::render(&graph, writer)?;

    Ok(())
}

/// Displays a DOT export of the analysed graph on the standard output.
pub fn display_dot(
    analyser: &Analyser<impl Borrow<Model>>,
    red_nodes: &Vec<usize>,
    red_edges: &Vec<usize>,
) -> CliResult<()> {
    render_dot(analyser, red_nodes, red_edges, &mut io::stdout())
}

/// Displays a render of the analysed graph using the `dot` command.
pub fn display_graph(
    analyser: &Analyser<impl Borrow<Model>>,
    red_nodes: &Vec<usize>,
    red_edges: &Vec<usize>,
) -> CliResult<()> {
    use std::process::{Command, Stdio};
    use std::{thread, time};

    let renderer = Command::new("dot")
        .arg("-Tpdf")
        .arg("-o")
        .arg("/tmp/tfd-graph.pdf")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    render_dot(analyser, red_nodes, red_edges, &mut renderer.stdin.unwrap())?;

    thread::sleep(time::Duration::from_secs(1));

    let _ = Command::new("evince")
        .arg("--fullscreen")
        .arg("/tmp/tfd-graph.pdf")
        .output();

    Ok(())
}
