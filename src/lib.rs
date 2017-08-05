#[macro_use]
extern crate downcast_rs;
#[macro_use]
extern crate error_chain;
extern crate ndarray;
extern crate num_traits;
extern crate protobuf;

pub mod tfpb;
pub mod matrix;
pub mod ops;

use std::{ fs, path, rc, str };
use std::collections::{HashMap, HashSet};
use ops::Op;

pub use matrix::Matrix;

error_chain!{
    foreign_links {
        Io(::std::io::Error);
        NdarrayShape(::ndarray::ShapeError);
    }
}

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
    pub fn new(graph: tfpb::GraphDef) -> GraphAnalyser {
        GraphAnalyser {
            graph,
            op_builder: ops::OpBuilder::new(),
            nodes: HashMap::new(),
            outputs: HashMap::new(),
        }
    }

    pub fn from_file<P: AsRef<path::Path>>(p: P) -> GraphAnalyser {
        let mut model = fs::File::open(p).unwrap();
        let loaded = ::protobuf::core::parse_from_reader::<::tfpb::graph::GraphDef>(&mut model).unwrap();
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
        Ok(self.nodes.get(name).unwrap().clone())
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
            op: self.op_builder.build(&pbnode)?
        })
    }

    pub fn set_value(&mut self, name: &str, value: Matrix) -> Result<()> {
        use std::borrow::BorrowMut;
        let mut node = self.get_node(name)?;
        if let Some(mut ph) = node.op.downcast_ref::<ops::Placeholder>() {
            ph.set(value);
            Ok(())
        } else {
            Err(format!("node {} is not a placeholder", name))?
        }
    }

    fn compute(&mut self, name: &str) -> Result<()> {
        println!("computing {}", name);
        if self.outputs.contains_key(name) {
            return Ok(())
        }
        let node:rc::Rc<Node> = self.get_node(name)?;
        let inputs:Vec<Matrix> = node.inputs.iter().map(|i| {
            self.compute(&*i.0.name)?;
            Ok(self.outputs.get(&i.0.name).unwrap()[i.1].clone())
        }).collect::<Result<_>>()?;
        let outputs = node.op.eval(inputs)?;
        self.outputs.insert(name.to_string(), outputs);
        Ok(())
    }

    pub fn eval(&mut self, name: &str) -> Result<Option<&Vec<Matrix>>> {
        self.compute(name)?;
        Ok(self.outputs.get(name))
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, str};
    use super::*;

    #[test]
    fn eval_tree() {
        let mut model = fs::File::open("model.pb").unwrap();
        let graph = ::protobuf::core::parse_from_reader::<::tfpb::graph::GraphDef>(&mut model)
            .unwrap();
        let mut graph = GraphAnalyser::new(graph);
        let tree = graph.get_node("logits").unwrap();
        println!("{}", tree.dump_eval_tree());
    }

    #[test]
    fn inputs() {
        let mut model = fs::File::open("model.pb").unwrap();
        let graph = ::protobuf::core::parse_from_reader::<::tfpb::graph::GraphDef>(&mut model)
            .unwrap();
        let mut graph = GraphAnalyser::new(graph);
        let tree = graph.get_node("word_cnn/ExpandDims_2").unwrap();
        println!("{}", tree.dump_eval_tree());
    }

    #[test]
    fn it_works() {
        let mut model = fs::File::open("model.pb").unwrap();
        let graph = ::protobuf::core::parse_from_reader::<::tfpb::graph::GraphDef>(&mut model)
            .unwrap();
        for node in graph.get_node() {
            println!("node : {} {}", node.get_name(), node.get_op());
            for (key, rev) in node.get_attr() {
                if rev.has_tensor() {
                    println!(
                        "  m {} -> {:?} {:?}",
                        key,
                        rev.get_tensor().get_dtype(),
                        rev.get_tensor().get_tensor_shape()
                    );
                /*
                } else if rev.has_f() {
                    println!("  f {} -> {:?}", key, rev.get_f());
                } else if rev.has_s() {
                    println!("  s {} -> {:?}", key, str::from_utf8(rev.get_s()).unwrap());
                } else if rev.has_b() {
                    println!("  b {} -> {:?}", key, rev.get_b());
                } else if rev.has_i() {
                    println!("  i {} -> {:?}", key, rev.get_i());
                } else if rev.has_field_type() {
                    println!("  t {} -> {:?}", key, rev.get_field_type());
                } else if rev.has_shape() {
                    println!("  h {} -> {:?}", key, rev.get_shape());
                } else if rev.has_tensor() {
                    println!(
                        "  m {} -> {:?} {:?}",
                        key,
                        rev.get_tensor().get_dtype(),
                        rev.get_tensor().get_tensor_shape()
                    );
                } else if rev.has_list() {
                    println!("  l {} -> {:?}", key, rev.get_list());
                    */
                } else {
                    println!("  * {} -> {:?}", key, rev);
                }
            }
        }
    }
}
