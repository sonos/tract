#![allow(dead_code)]

use dot;
use errors::*;
use std::io;
use std::io::Write;
use tfdeploy::analyser::*;

type Nd = (String, String, bool);
type Ed = Edge;

struct Graph<'a> {
    forward: bool,
    nodes: &'a [Nd],
    edges: &'a [Ed],
}

/// Removes special characters from strings.
fn slugify(text: &String) -> String {
    text.replace("/", "").replace(".", "").replace(" ", "_")
}

/// An implementation of dot::Labeller for the analyser.
impl<'a> dot::Labeller<'a, Nd, Ed> for Graph<'a> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("analysed_graph").unwrap()
    }
    fn node_id(&'a self, n: &Nd) -> dot::Id<'a> {
        dot::Id::new(format!("node_{}", slugify(&n.0))).unwrap()
    }
    fn node_label<'b>(&'b self, n: &Nd) -> dot::LabelText<'b> {
        dot::LabelText::LabelStr(format!("{} ({})", n.0, n.1).into())
    }
    fn node_color<'b>(&'b self, n: &Nd) -> Option<dot::LabelText<'b>> {
        if n.2 {
            Some(dot::LabelText::LabelStr("crimson".into()))
        } else {
            None
        }
    }
    fn edge_label<'b>(&'b self, e: &Ed) -> dot::LabelText<'b> {
        use self::ValueFact::*;

        let mut label = format!(
            "{:?}\n{:?}\n{}",
            e.fact.datatype,
            e.fact.shape,
            match e.fact.value { Any => "Any", Only(_) => "Only(_)" }
        );
        label.truncate(150);

        dot::LabelText::LabelStr(label.into())
    }
    fn edge_color<'b>(&'b self, e: &Ed) -> Option<dot::LabelText<'b>> {
        if self.forward && self.nodes[e.from_node].2 {
            Some(dot::LabelText::LabelStr("crimson".into()))
        } else if !self.forward && self.nodes[e.to_node].2 {
            Some(dot::LabelText::LabelStr("crimson".into()))
        } else {
            None
        }
    }
}

/// An implementation of dot::GraphWalk for the analyser.
impl<'a> dot::GraphWalk<'a, Nd, Ed> for Graph<'a> {
    fn nodes(&self) -> dot::Nodes<'a, Nd> {
        self.nodes.into()
    }
    fn edges(&'a self) -> dot::Edges<'a, Ed> {
        self.edges.into()
    }
    fn source(&self, e: &Ed) -> Nd {
        self.nodes[e.from_node].clone()
    }
    fn target(&self, e: &Ed) -> Nd {
        self.nodes[e.to_node].clone()
    }
}

/// Writes a DOT export of the analysed graph to a given Writer.
pub fn render_dot<W: Write>(
    analyser: &Analyser,
    highlighted: &Vec<usize>,
    writer: &mut W
) -> Result<()> {
    let nodes: Vec<_> = analyser.nodes
        .iter()
        .map(|n| match n {
            Some(n) => (
                n.name.clone(),
                n.op_name.clone(),
                highlighted.contains(&n.id)
            ),
            None => ("output".to_string(), "output".to_string(), false)
        })
        .collect();

    let graph = Graph {
        forward: analyser.current_direction,
        nodes: nodes.as_slice(),
        edges: analyser.edges.as_slice(),
    };

    dot::render(&graph, writer)?;

    Ok(())
}

/// Displays a DOT export of the analysed graph on the standard output.
pub fn display_dot(analyser: &Analyser, highlighted: &Vec<usize>) -> Result<()> {
    render_dot(analyser, highlighted, &mut io::stdout())
}

/// Displays a render of the analysed graph using the `dot` command.
pub fn display_graph(analyser: &Analyser, highlighted: &Vec<usize>) -> Result<()> {
    use std::process::{Command, Stdio};

    let renderer = Command::new("dot")
        .arg("-Tpdf")
        .arg("-o")
        .arg("/tmp/tfd-graph.pdf")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    render_dot(analyser, highlighted, &mut renderer.stdin.unwrap())?;

    let _ = Command::new("xdg-open")
        .arg("/tmp/tfd-graph.pdf")
        .output();

    Ok(())
}
