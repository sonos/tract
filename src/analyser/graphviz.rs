use analyser::Edge;
use dot;
use errors::*;
use std::io;
use std::io::Write;

type Nd = (String, String, bool);
type Ed = Edge;

struct Graph<'a> {
    name: String,
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
        dot::Id::new(format!("graph_{}", slugify(&self.name))).unwrap()
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
        use analyser::ValueFact::*;

        let mut label = format!(
            "{:?}\n{:?}\n{}",
            e.tensor.datatype,
            e.tensor.shape,
            match e.tensor.value { Any => "Any", Only(_) => "Only(_)" }
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
    name: String,
    nodes: &Vec<(usize, String, String)>,
    edges: &Vec<Edge>,
    highlighted: &Vec<usize>,
    forward: bool,
    writer: &mut W,
) -> Result<()> {
    let nodes: Vec<_> = nodes
        .iter()
        .map(|(id, name, op)| (name.clone(), op.clone(), highlighted.contains(id)))
        .collect();

    let graph = Graph {
        name,
        forward,
        nodes: nodes.as_slice(),
        edges: edges.as_slice(),
    };

    dot::render(&graph, writer)?;

    Ok(())
}

/// Displays a DOT export of the analysed graph on the standard output.
pub fn display_dot(
    name: String,
    nodes: &Vec<(usize, String, String)>,
    edges: &Vec<Edge>,
    highlighted: &Vec<usize>,
    forward: bool,
) -> Result<()> {
    render_dot(name, nodes, edges, highlighted, forward, &mut io::stdout())
}

/// Displays a render of the analysed graph using the `dot` command.
pub fn display_graph(
    name: String,
    nodes: &Vec<(usize, String, String)>,
    edges: &Vec<Edge>,
    highlighted: &Vec<usize>,
    forward: bool,
) -> Result<()> {
    use std::process::{Command, Stdio};

    let renderer = Command::new("dot")
        .arg("-Tpng")
        .arg("-o")
        .arg("/tmp/tfd-graph.png")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    render_dot(
        name,
        nodes,
        edges,
        highlighted,
        forward,
        &mut renderer.stdin.unwrap(),
    )?;

    let _ = Command::new("eog")
        .arg("--fullscreen")
        .arg("/tmp/tfd-graph.png")
        .output();

    Ok(())
}
