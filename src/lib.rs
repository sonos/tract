#[macro_use]
extern crate downcast_rs;
#[macro_use]
extern crate error_chain;
extern crate image;
#[macro_use]
extern crate log;
#[allow(unused_imports)]
#[macro_use]
extern crate ndarray;
extern crate num_traits;
extern crate protobuf;

pub mod tfpb;
pub mod matrix;
pub mod ops;

use std::{fs, path, rc, str};
use std::collections::{HashMap, HashSet};
use ops::Op;

pub use matrix::Matrix;

error_chain!{
    foreign_links {
        Image(image::ImageError);
        Io(::std::io::Error);
        NdarrayShape(::ndarray::ShapeError);
        Protobuf(::protobuf::ProtobufError);
        StrUtf8(::std::str::Utf8Error);
    }
}

#[derive(Debug)]
pub struct Node {
    name: String,
    inputs: Vec<(rc::Rc<Node>, usize)>,
    op: Box<Op>,
}

impl Node {
    pub fn dump_eval_tree(&self) -> String {
        self._dump_eval_tree(0, &mut HashSet::new())
    }

    fn _dump_eval_tree(&self, depth: usize, dups: &mut HashSet<String>) -> String {
        let pad: String = ::std::iter::repeat("  ").take(depth).collect();
        let mut s = format!("{}{}\n", pad, self.name);
        for i in &self.inputs {
            s.push_str(&*format!("{}", i.0._dump_eval_tree(depth + 1, dups)));
        }
        s
    }
}

pub struct GraphAnalyser {
    graph: tfpb::GraphDef,
    op_builder: ops::OpBuilder,
    nodes: HashMap<String, rc::Rc<Node>>,
    outputs: HashMap<String, Vec<Matrix>>,
}

impl GraphAnalyser {
    pub fn new(graph: tfpb::GraphDef) -> Result<GraphAnalyser> {
        Ok(GraphAnalyser {
            graph,
            op_builder: ops::OpBuilder::new(),
            nodes: HashMap::new(),
            outputs: HashMap::new(),
        })
    }

    pub fn from_file<P: AsRef<path::Path>>(p: P) -> Result<GraphAnalyser> {
        Self::from_reader(fs::File::open(p)?)
    }

    pub fn from_reader<R: ::std::io::Read>(mut r: R) -> Result<GraphAnalyser> {
        let loaded = ::protobuf::core::parse_from_reader::<::tfpb::graph::GraphDef>(&mut r)?;
        GraphAnalyser::new(loaded)
    }

    pub fn node_names(&self) -> Vec<&str> {
        self.graph.get_node().iter().map(|n| n.get_name()).collect()
    }

    pub fn get_node(&mut self, name: &str) -> Result<rc::Rc<Node>> {
        if !self.nodes.contains_key(name) {
            let node = self.make_node(name)?;
            self.nodes.insert(name.to_string(), rc::Rc::new(node));
        }
        Ok(self.nodes.get(name).ok_or(format!("node {} do not exist", name))?.clone())
    }

    fn make_node(&mut self, name: &str) -> Result<Node> {
        let pbnode = self.graph
            .get_node()
            .iter()
            .find(|n| n.get_name() == name)
            .unwrap()
            .clone();
        Ok(Node {
            name: name.to_string(),
            inputs: pbnode
                .get_input()
                .iter()
                .map(|s| self.get_node(&s).map(|n| (n, 0)))
                .collect::<Result<_>>()?,
            op: self.op_builder.build(&pbnode)?,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.outputs.clear();
        Ok(())
    }

    pub fn set_value(&mut self, name: &str, value: Matrix) -> Result<()> {
        let node = self.get_node(name)?;
        if let Some(ph) = node.op.downcast_ref::<ops::trivial::Placeholder>() {
            ph.set(value);
            Ok(())
        } else {
            Err(format!("node {} is not a placeholder", name))?
        }
    }

    fn compute(&mut self, name: &str) -> Result<()> {
        if self.outputs.contains_key(name) {
            return Ok(());
        }
        let node: rc::Rc<Node> = self.get_node(name)?;
        let inputs: Vec<Matrix> = node.inputs
            .iter()
            .map(|i| {
                self.compute(&*i.0.name)?;
                Ok(self.outputs.get(&i.0.name).unwrap()[i.1].clone())
            })
            .collect::<Result<_>>()?;
        let outputs = node.op.eval(inputs)?;
        // println!("storing {} -> {:?}", name, outputs.iter().map(|m| m.shape()).collect::<Vec<_>>());
        self.outputs.insert(name.to_string(), outputs);
        Ok(())
    }

    pub fn eval(&mut self, name: &str) -> Result<&Vec<Matrix>> {
        self.compute(name)?;
        Ok(self.outputs.get(name).expect("node found but was not computed"))
    }

    pub fn take(&mut self, name: &str) -> Result<Vec<Matrix>> {
        self.compute(name)?;
        Ok(self.outputs.remove(name).ok_or(
            format!("{} does not exits", name),
        )?)
    }
}
