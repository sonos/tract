use std::collections::HashMap;
use std::sync::Arc;

use errors::*;
use ops::Op;
use model:: {Model, RawModel };
use Node;
use Plan;

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

    /// Attempts to unify two datatype facts.
    pub fn unify_datatype(x: &TypeFact, y: &TypeFact) -> Result<TypeFact> {
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
pub fn detect_inputs(model: &Model) -> Result<Option<Vec<usize>>> {
    let inputs: Vec<_> = model
        .nodes()
        .iter()
        .filter(|n| n.op_name == "Placeholder")
        .inspect(|n| info!("Autodetected input node: {} {:?}.", n.id, n.name))
        .map(|n| n.id)
        .collect();

    if inputs.len() > 0 {
        Ok(Some(inputs))
    } else {
        Ok(None)
    }
}

/// Tries to auto-detect the name of the output node.
pub fn detect_output(model: &Model) -> Result<Option<usize>> {
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
                model.get_node_by_id(i)?.name
            );
            return Ok(Some(i));
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

    // The current state of the algorithm.
    pub current_pass: usize,
    pub current_step: usize,
    pub current_direction: bool,
}

impl Analyser {
    /// Constructs an analyser for the given graph.
    ///
    /// The output argument is used to infer an execution plan for the graph.
    /// Changing it won't alter the correctness of the analysis, but it might
    /// take much longer to complete.
    pub fn new(model: &Model, output: usize) -> Result<Analyser> {
        let nodes:Vec<Node> = model.nodes().iter().cloned().collect();
        let mut edges = vec![];
        let mut prev_edges = vec![Vec::new(); model.nodes().len() + 1];
        let mut next_edges = vec![Vec::new(); model.nodes().len() + 1];

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
            from_node: Some(output),
            from_out: 0,
            to_node: None,
            to_input: 0,
            fact: TensorFact::new(),
        });

        next_edges[output].push(special_edge_id);

        // Compute an execution plan for the graph.
        let plan = Plan::for_model(model, &[output])?.order;
        let current_pass = 0;
        let current_step = 0;
        let current_direction = true;

        debug!("Using execution plan {:?}.", plan);

        Ok(Analyser {
            model: model.clone(),
            output,
            nodes,
            edges,
            prev_edges,
            next_edges,
            plan,
            current_pass,
            current_step,
            current_direction,
        })
    }

    /// Adds an user-provided tensor fact to the analyser.
    pub fn hint(&mut self, node: usize, fact: &TensorFact) -> Result<()> {
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
    pub fn with_hint(mut self, node: usize, fact: &TensorFact) -> Result<Analyser> {
        self.hint(node, fact)?;
        Ok(self)
    }

    /// Adds an user-provided tensor fact to the analyser.
    pub fn with_named_hint(mut self, node: &str, fact: &TensorFact) -> Result<Analyser> {
        let node = self.model.node_id_by_name(node)?;
        self.hint(node, fact)?;
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
        constants::propagate_constants(self)?;
        self.to_model()
    }

    /// Computes a new execution plan for the graph.
    pub fn reset_plan(&mut self) -> Result<()> {
        self.plan = Plan::for_nodes(&self.nodes, &vec!(self.output))?.order;
        Ok(())
    }

    /// Detaches the constant nodes and edges from the given graph.
    pub fn propagate_constants(&mut self) -> Result<()> {
        constants::propagate_constants(self)
    }

    /*
    /// Removes the nodes and edges which are not part of the execution plan.
    /// Returns the mapping between the old and new node indexes.
    pub fn prune_unused(&mut self) -> Vec<Option<usize>> {
        let mut node_used = vec![false; self.nodes.len()];
        let mut edge_used = vec![false; self.edges.len()];
        for &i in &self.plan {
            node_used[i] = true;
        }

        // Remove the nodes while keeping track of the new indices.
        let mut deleted = 0;
        let mut node_mapping = vec![None; self.nodes.len()];

        for i in 0..self.nodes.len() {
            if !node_used[i] {
                self.nodes.remove(i - deleted);

                self.prev_edges.remove(i - deleted);
                self.next_edges.remove(i - deleted);
                deleted += 1;
            } else {
                node_mapping[i] = Some(i - deleted);

                self.prev_edges[i - deleted]
                    .iter()
                    .for_each(|&j| edge_used[j] = true);
                self.next_edges[i - deleted]
                    .iter()
                    .for_each(|&j| edge_used[j] = true);
            }
        }

        info!("Nodes: before: {}, deleted: {}, kept: {}", self.nodes.len(), deleted, self.nodes.len() - deleted);

        // Update the nodes and edges to use the new indices.
        for node in &mut self.nodes {
            node.id = node_mapping[node.id].unwrap();
            node.inputs
                .iter_mut()
                .for_each(|i| i.0 = node_mapping[i.0].unwrap());
        }

        for edge in &mut self.edges {
            if let Some(i) = edge.from_node {
                edge.from_node = node_mapping[i];
            }

            if let Some(i) = edge.to_node {
                edge.to_node = node_mapping[i];
            }
        }

        // Remove the edges while keeping track of the new indices.
        let mut deleted = 0;
        let mut edge_mapping = vec![None; self.edges.len()];

        for i in 0..self.edges.len() {
            if !edge_used[i] {
                self.edges.remove(i - deleted);
                deleted += 1;
            } else {
                edge_mapping[i] = Some(i - deleted);
            }
        }

        info!("Edges: before: {}, deleted: {}, kept: {}", self.edges.len(), deleted, self.edges.len() - deleted);

        // Update the adjacency lists to use the new indices.
        for i in 0..self.nodes.len() {
            self.prev_edges[i]
                .iter_mut()
                .for_each(|j| *j = edge_mapping[*j].unwrap());
            self.next_edges[i]
                .iter_mut()
                .for_each(|j| *j = edge_mapping[*j].unwrap());
        }

        node_mapping
    }
    */

    /// Runs the entire analysis at once.
    pub fn analyse(&mut self) -> Result<()> {
        self.current_pass = 0;

        loop {
            if !self.run_two_passes()? {
                return Ok(());
            }
        }
    }

    /// Runs two passes of the analysis.
    pub fn run_two_passes(&mut self) -> Result<bool> {
        let mut changed = false;

        info!(
            "Starting pass [pass={:?}, direction={:?}].",
            self.current_pass, self.current_direction,
        );

        // We first run a forward pass.
        self.current_step = 0;
        for _ in 0..self.plan.len() {
            if self.analyse_step()? {
                changed = true;
            }
        }

        info!(
            "Starting pass [pass={:?}, direction={:?}].",
            self.current_pass, self.current_direction,
        );

        // We then run a backward pass.
        self.current_step = 0;
        for _ in 0..self.plan.len() {
            if self.analyse_step()? {
                changed = true;
            }
        }

        Ok(changed)
    }

    /// Runs a single step of the analysis.
    pub fn analyse_step(&mut self) -> Result<bool> {
        let changed = self.try_step()?;

        // Switch to the next step.
        self.current_step += 1;
        if self.current_step == self.plan.len() {
            self.current_pass += 1;
            self.current_direction = !self.current_direction;
            self.current_step = 0;
        }

        Ok(changed)
    }

    /// Tries to run a single step of the analysis, and returns whether
    /// there was any additional information gained during the step.
    fn try_step(&mut self) -> Result<bool> {
        let node_id = if self.current_direction {
            self.plan[self.current_step]
        } else {
            self.plan[self.plan.len() - 1 - self.current_step]
        };
        let node = &self.nodes[node_id];

        debug!(
            "Starting step for {} {} ({}) [pass={:?}, direction={:?}, step={:?}].",
            node.id,
            node.name,
            node.op_name,
            self.current_pass,
            self.current_direction,
            self.current_step,
        );

        trace!("{:?}", node.op);

        let inputs: Vec<_> = self.prev_edges[node_id]
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
        for &i in &self.next_edges[node_id] {
            outputs[0] = unify(&self.edges[i].fact, &outputs[0])?;
        }
        trace!("  Output 0: {:?}", &outputs[0]);

        let inferred = node.op
            .infer_and_propagate(inputs, outputs)
            .map_err(|e| format!("While inferring forward for {} {}: {}", node_id, node.name, e))?;

        let mut changed = false;

        for (i, &j) in self.prev_edges[node_id].iter().enumerate() {
            let fact = &inferred.0[i];
            let unified = unify(fact, &self.edges[j].fact)
                .map_err(|e| format!("While unifying inputs of node {} {}: {}", node_id, node.name, e))?;

            if unified != self.edges[j].fact {
                debug!(" Refined {} input #{} to {:?}", node.name, i, unified);
                changed = true;
                self.edges[j].fact = unified;
            }
        }

        for (i, &j) in self.next_edges[node_id].iter().enumerate() {
            // FIXME(liautaud): We should handle multiple output ports in the future.
            if inferred.1.len() != 1 {
                panic!("Inference only supports nodes with a single output port.");
            }

            let fact = &inferred.1[0];
            let unified = unify(fact, &self.edges[j].fact)
                .map_err(|e| format!("While unifying outputs of node {} {} {}", node_id, node.name, e))?;

            if unified != self.edges[j].fact {
                debug!(" Refined {} output {}/{} to {:?}", node.name, node_id, i, unified);
                changed = true;
                self.edges[j].fact = unified;
            }
        }

        Ok(changed)
    }
}

#[cfg(tests)]
mod tests {
    #[test]
    fn unify_same_datatype() {
        let dt = TypeFact::Only(DataType::DT_FLOAT);
        assert_eq!(unify_datatype(&dt, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_datatypes_only() {
        let dt1 = TypeFact::Only(DataType::DT_FLOAT);
        let dt2 = TypeFact::Only(DataType::DT_DOUBLE);
        assert!(unify_datatype(&dt1, &dt2).is_err());
    }

    #[test]
    fn unify_different_datatypes_any_left() {
        let dt = TypeFact::Only(DataType::DT_FLOAT);
        assert_eq!(unify_datatype(&TypeFact::Any, &dt).unwrap(), dt);
    }

    #[test]
    fn unify_different_datatypes_any_right() {
        let dt = TypeFact::Only(DataType::DT_FLOAT);
        assert_eq!(unify_datatype(&dt, &TypeFact::Any).unwrap(), dt);
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
