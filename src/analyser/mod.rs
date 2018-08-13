use std::collections::{ HashMap, BTreeSet };
use std::sync::Arc;

use errors::*;
use ops::Op;
use model:: {Model, RawModel };
use Node;
use model::eval_order_for_nodes;

mod constants;
mod types;

pub mod prelude {
    pub use super::types::*;
    pub use super::Analyser;
    use Result;

    /// Attempts to unify two tensor facts into a more specialized one.
    pub fn unify(x: &TensorFact, y: &TensorFact) -> Result<TensorFact> {
        x.unify(y)
    }

    /// Attempts to unify two datum_type facts.
    pub fn unify_datum_type(x: &TypeFact, y: &TypeFact) -> Result<TypeFact> {
        x.unify(y)
    }

    /// Attempts to unify two shape facts.
    pub fn unify_shape(x: &ShapeFact, y: &ShapeFact) -> Result<ShapeFact> {
        x.unify(y)
    }

    /// Attempts to unify two dimension facts.
    pub fn unify_dim(x: &DimFact, y: &DimFact) -> Result<DimFact> {
        x.unify(y)
    }

    /// Attempts to unify two value facts.
    pub fn unify_value(x: &ValueFact, y: &ValueFact) -> Result<ValueFact> {
        x.unify(y)
    }
}

pub use self::prelude::*;

#[macro_use]
pub mod macros;
#[macro_use]
pub mod helpers;
#[macro_use]
pub mod interface;

/// Tries to auto-detect the names of the input nodes.
pub fn detect_inputs(model: &Model) -> Result<Vec<&Node>> {
    let inputs: Vec<_> = model
        .nodes()
        .iter()
        .filter(|n| n.op_name == "Placeholder")
        .inspect(|n| info!("Autodetected input node: {} {:?}.", n.id, n.name))
        .collect();

    Ok(inputs)
}

/// Tries to auto-detect the name of the output node.
pub fn detect_output(model: &Model) -> Result<Option<&Node>> {
    // We search for the only node in the graph with no successor.
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); model.nodes().len()];

    for node in model.nodes() {
        for &link in &node.inputs {
            succs[link.0].push(node.id);
        }
    }

    for (i, s) in succs.iter().enumerate() {
        if s.len() == 0 {
            info!(
                "Autodetected output node: {} {:?}.",
                i,
                model.nodes()[i].name
            );
            return Ok(Some(&model.nodes[i]));
        }
    }

    Ok(None)
}

/// An edge of the analysed graph, annotated by a fact.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub id: usize,
    pub from_node: Option<usize>,
    pub from_out: usize,
    pub to_node: Option<usize>,
    pub to_input: usize,
    pub fact: TensorFact,
}

/// A graph analyser, along with its current state.
pub struct Analyser {
    model: Model,
    // The output.
    pub output: usize,

    pub nodes: Vec<Node>,

    // The graph being analysed.
    pub edges: Vec<Edge>,
    pub prev_edges: Vec<Vec<usize>>,
    pub next_edges: Vec<Vec<usize>>,

    // The execution plan
    plan: Vec<usize>,
}

impl Analyser {
    /// Constructs an analyser for the given graph.
    ///
    /// The output argument is used to infer an execution plan for the graph.
    /// Changing it won't alter the correctness of the analysis, but it might
    /// take much longer to complete.
    pub fn new(model: &Model, output: &str) -> Result<Analyser> {
        let nodes:Vec<Node> = model.nodes().iter().cloned().collect();
        let mut edges = vec![];
        let mut prev_edges = vec![Vec::new(); model.nodes().len() + 1];
        let mut next_edges = vec![Vec::new(); model.nodes().len() + 1];
        let output = model.node_by_name(output)?;

        for node in &nodes {
            for (ix, input) in node.inputs.iter().enumerate() {
                let id = edges.len();

                edges.push(Edge {
                    id,
                    from_node: Some(input.0),
                    from_out: input.1,
                    to_node: Some(node.id),
                    to_input: ix,
                    fact: TensorFact::new(),
                });

                prev_edges[node.id].push(id);
                next_edges[input.0].push(id);
            }
        }

        // Add a special output edge.
        let special_edge_id = edges.len();
        edges.push(Edge {
            id: special_edge_id,
            from_node: Some(output.id),
            from_out: 0,
            to_node: None,
            to_input: 0,
            fact: TensorFact::new(),
        });

        next_edges[output.id].push(special_edge_id);

        // Compute an execution plan for the graph.
        let plan = eval_order_for_nodes(model.nodes(), &[output.id])?;

        trace!("Using execution plan {:?}.", plan);

        Ok(Analyser {
            model: model.clone(),
            output: output.id,
            nodes,
            edges,
            prev_edges,
            next_edges,
            plan,
        })
    }

    /// Adds an user-provided tensor fact to the analyser.
    pub fn hint(&mut self, node: &str, fact: &TensorFact) -> Result<()> {
        let id = self.model.node_by_name(node)?.id;
        self.hint_by_id(id, fact)
    }

    /// Adds an user-provided tensor fact to the analyser.
    pub fn hint_by_id(&mut self, node: usize, fact: &TensorFact) -> Result<()> {
        debug!("Hint for node \"{}\": {:?}", self.model.nodes()[node].name, fact);
        if node >= self.next_edges.len() {
            bail!("There is no node with index {:?}.", node);
        }

        for &j in &self.next_edges[node] {
            self.edges[j].fact = unify(fact, &self.edges[j].fact)?;
        }

        Ok(())
    }

    /// Adds an user-provided tensor fact to the analyser.
    pub fn with_hint(mut self, node: &str, fact: &TensorFact) -> Result<Analyser> {
        let node = self.model.node_by_name(node)?.id;
        self.hint_by_id(node, fact)?;
        Ok(self)
    }

    /// Returns a model from the analyser.
    pub fn to_model(&self) -> Result<Model> {
        let mut nodes_by_name = HashMap::with_capacity(self.plan.len());
        let mut nodes_mapped = HashMap::with_capacity(self.plan.len());
        let mut nodes = Vec::with_capacity(self.plan.len());
        self.plan.iter().enumerate().for_each(|(ix,&n)| {
            let old_node = &self.nodes[n];
            nodes_by_name.insert(old_node.name.clone(), ix);
            nodes_mapped.insert(old_node.id, ix);
            nodes.push(Node {
                id: ix,
                name: old_node.name.clone(),
                op_name: old_node.op_name.clone(),
                inputs: old_node.inputs.iter().map(|&(input, port)| (nodes_mapped[&input], port)).collect(),
                op: old_node.op.clone(),
            });
        });

        Ok(Model(Arc::new(RawModel { nodes, nodes_by_name, })))
    }

    /// Returns a model from the analyser.
    pub fn to_optimized_model(&mut self) -> Result<Model> {
        self.analyse()?;
        constants::propagate_constants(self)?;
        self.to_model()
    }

    /// Computes a new execution plan for the graph.
    pub fn reset_plan(&mut self) -> Result<()> {
        self.plan = eval_order_for_nodes(&self.nodes, &[self.output])?;
        Ok(())
    }

    /// Detaches the constant nodes and edges from the given graph.
    pub fn propagate_constants(&mut self) -> Result<()> {
        constants::propagate_constants(self)
    }

    /// Runs the entire analysis at once.
    pub fn analyse(&mut self) -> Result<()> {
        let mut nodes_to_visit:BTreeSet<usize> = (0..self.nodes.len()).collect();
        loop {
            trace!("Remaining nodes {}", nodes_to_visit.len());
            let node = match nodes_to_visit.iter().next() {
                None => return Ok(()),
                Some(n) => *n,
            };
            let changed_edges = self.step(node)?;
            for edge in changed_edges {
                let edge = &self.edges[edge];
                trace!("Changed edge: {:?}", edge);
                if let Some(dst) = edge.to_node {
                    if dst != node {
                        trace!("Inserting node dn {}", dst);
                        nodes_to_visit.insert(dst);
                    }
                }
                if let Some(src) = edge.from_node {
                    if src != node {
                        trace!("Inserting node up {}", src);
                        nodes_to_visit.insert(src);
                    }
                }
            }
            nodes_to_visit.remove(&node);
        }
    }

    /// Tries to run a single step of the analysis, and returns whether
    /// there was any additional information gained during the step.
    fn step(&mut self, node: usize) -> Result<Vec<usize>> {
        let node = &self.nodes[node];
        debug!(
            "Starting step for {} {} ({})",
            node.id,
            node.name,
            node.op_name,
        );

        trace!("{:?}", node.op);

        let inputs: Vec<_> = self.prev_edges[node.id]
            .iter()
            .map(|&i| &self.edges[i])
            .inspect(|edge| {
                trace!(
                    " Input {} from {}/{}: {:?}",
                    edge.to_input,
                    edge.from_node.unwrap(),
                    edge.from_out,
                    edge.fact
                );
            })
            .map(|edge| edge.fact.clone())
            .collect();

        // FIXME(liautaud): We should handle multiple output ports in the future.
        let mut outputs = vec![TensorFact::new()];
        for &i in &self.next_edges[node.id] {
            outputs[0] = unify(&self.edges[i].fact, &outputs[0])?;
        }
        trace!("  Output 0: {:?}", &outputs[0]);

        let inferred = node.op
            .infer_and_propagate(inputs, outputs)
            .map_err(|e| format!("While inferring forward for {} {}: {}", node.id, node.name, e))?;

        let mut changed_edges = vec!();

        for (i, &j) in self.prev_edges[node.id].iter().enumerate() {
            let fact = &inferred.0[i];
            let unified = unify(fact, &self.edges[j].fact)
                .map_err(|e| format!("While unifying inputs of node {} {}: {}", node.id, node.name, e))?;

            if unified != self.edges[j].fact {
                debug!(" Refined {} input #{} to {:?}", node.name, i, unified);
                changed_edges.push(j);
                self.edges[j].fact = unified;
            }
        }

        for (i, &j) in self.next_edges[node.id].iter().enumerate() {
            // FIXME(liautaud): We should handle multiple output ports in the future.
            if inferred.1.len() != 1 {
                panic!("Inference only supports nodes with a single output port.");
            }

            let fact = &inferred.1[0];
            let unified = unify(fact, &self.edges[j].fact)
                .map_err(|e| format!("While unifying outputs of node {} {} {}", node.id, node.name, e))?;

            if unified != self.edges[j].fact {
                debug!(" Refined {} output {}/{} to {:?}", node.name, node.id, i, unified);
                changed_edges.push(j);
                self.edges[j].fact = unified;
            }
        }

        Ok(changed_edges)
    }
}

#[cfg(tests)]
mod tests {
    #[test]
    fn unify_same_datum_type() {
        let dt = TypeFact::Only(DatumType::DT_FLOAT);
        assert_eq!(unify_datum_type(&dt, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_datum_types_only() {
        let dt1 = TypeFact::Only(DatumType::DT_FLOAT);
        let dt2 = TypeFact::Only(DatumType::DT_DOUBLE);
        assert!(unify_datum_type(&dt1, &dt2).is_err());
    }

    #[test]
    fn unify_different_datum_types_any_left() {
        let dt = TypeFact::Only(DatumType::DT_FLOAT);
        assert_eq!(unify_datum_type(&TypeFact::Any, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_datum_types_any_right() {
        let dt = TypeFact::Only(DatumType::DT_FLOAT);
        assert_eq!(unify_datum_type(&dt, &TypeFact::Any).unwrap(), dt);
    }

    #[test]
    fn unify_same_shape_1() {
        let s = ShapeFact::closed(vec![]);
        assert_eq!(unify_shape(&s, &s).unwrap(), s);
    }

    #[test]
    fn unify_same_shape_2() {
        use super::DimFact::*;
        let s = ShapeFact::closed(vec![Any]);
        assert_eq!(unify_shape(&s, &s).unwrap(), s);
    }

    #[test]
    fn unify_same_shape_3() {
        use super::DimFact::*;
        let s = ShapeFact::closed(vec![Only(1), Only(2)]);
        assert_eq!(unify_shape(&s, &s).unwrap(), s);
    }

    #[test]
    fn unify_different_shapes_1() {
        use super::DimFact::*;
        let s1 = ShapeFact::closed(vec![Only(1), Only(2)]);
        let s2 = ShapeFact::closed(vec![Only(1)]);
        assert!(unify_shape(&s1, &s2).is_err());
    }

    #[test]
    fn unify_different_shapes_2() {
        use super::DimFact::*;
        let s1 = ShapeFact::closed(vec![Only(1), Only(2)]);
        let s2 = ShapeFact::closed(vec![Any]);
        assert!(unify_shape(&s1, &s2).is_err());
    }

    #[test]
    fn unify_different_shapes_3() {
        use super::DimFact::*;
        let s1 = ShapeFact::open(vec![Only(1), Only(2)]);
        let s2 = ShapeFact::closed(vec![Any]);
        assert!(unify_shape(&s1, &s2).is_err());
    }

    #[test]
    fn unify_different_shapes_4() {
        use super::DimFact::*;
        let s1 = ShapeFact::closed(vec![Any]);
        let s2 = ShapeFact::closed(vec![Any]);
        let sr = ShapeFact::closed(vec![Any]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_5() {
        use super::DimFact::*;
        let s1 = ShapeFact::closed(vec![Any]);
        let s2 = ShapeFact::closed(vec![Only(1)]);
        let sr = ShapeFact::closed(vec![Only(1)]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_6() {
        use super::DimFact::*;
        let s1 = ShapeFact::open(vec![]);
        let s2 = ShapeFact::closed(vec![Only(1)]);
        let sr = ShapeFact::closed(vec![Only(1)]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_different_shapes_7() {
        use super::DimFact::*;
        let s1 = ShapeFact::open(vec![Any, Only(2)]);
        let s2 = ShapeFact::closed(vec![Only(1), Any, Any]);
        let sr = ShapeFact::closed(vec![Only(1), Only(2), Any]);
        assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
    }

    #[test]
    fn unify_same_value() {
        use ndarray::prelude::*;
        let dt = ValueFact::Only(Tensor::F32(ArrayD::zeros(IxDyn(&[1]))));
        assert_eq!(unify_value(&dt, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_values_only() {
        use ndarray::prelude::*;
        let dt1 = ValueFact::Only(Tensor::F32(ArrayD::zeros(IxDyn(&[1]))));
        let dt2 = ValueFact::Only(Tensor::F32(ArrayD::zeros(IxDyn(&[2]))));
        assert!(unify_value(&dt1, &dt2).is_err());
    }

    #[test]
    fn unify_different_values_any_left() {
        use ndarray::prelude::*;
        let dt = ValueFact::Only(Tensor::F32(ArrayD::zeros(IxDyn(&[1]))));
        assert_eq!(unify_value(&ValueFact::Any, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_values_any_right() {
        use ndarray::prelude::*;
        let dt = ValueFact::Only(Tensor::F32(ArrayD::zeros(IxDyn(&[1]))));
        assert_eq!(unify_value(&dt, &ValueFact::Any).unwrap(), dt);
    }
}
