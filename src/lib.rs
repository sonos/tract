extern crate protobuf;

pub mod tfpb;

use std::rc;
use std::collections::{HashMap, HashSet};

struct Node {
    name: String,
    inputs: Vec<(rc::Rc<Node>, usize)>,
}

impl Node {
    fn dump_eval_tree(&self) -> String {
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

struct GraphAnalyser {
    graph: tfpb::GraphDef,
    nodes: HashMap<String, rc::Rc<Node>>,
}

impl GraphAnalyser {
    pub fn new(graph: tfpb::GraphDef) -> GraphAnalyser {
        GraphAnalyser {
            graph,
            nodes: HashMap::new(),
        }
    }

    fn get_node(&mut self, name: &str) -> rc::Rc<Node> {
        if !self.nodes.contains_key(name) {
            let node = self.make_node(name);
            self.nodes.insert(name.to_string(), rc::Rc::new(node));
        }
        self.nodes.get(name).unwrap().clone()
    }

    fn make_node(&mut self, name: &str) -> Node {
        let pbnode = self.graph
            .get_node()
            .iter()
            .find(|n| n.get_name() == name)
            .unwrap()
            .clone();
        Node {
            name: name.to_string(),
            inputs: pbnode
                .get_input()
                .iter()
                .map(|s| (self.get_node(&s), 0))
                .collect(),
        }
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
        let tree = graph.get_node("logits");
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
