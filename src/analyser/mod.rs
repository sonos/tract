use errors::*;
use ops::Op;
use Model;
use Node;
use Plan;

mod types;
pub use self::types::*;

#[cfg(feature = "serialize")]
use serde_json;

#[macro_use]
pub mod macros;
#[macro_use]
pub mod helpers;
pub mod constants;

/// Attempts to unify two tensor facts into a more specialized one.
pub fn unify(x: &TensorFact, y: &TensorFact) -> Result<TensorFact> {
    let tensor = TensorFact {
        datatype: unify_datatype(&x.datatype, &y.datatype)?,
        shape: unify_shape(&x.shape, &y.shape)?,
        value: unify_value(&x.value, &y.value)?,
    };

    Ok(tensor)
}

/// Attempts to unify two datatype facts.
pub fn unify_datatype(x: &TypeFact, y: &TypeFact) -> Result<TypeFact> {
    use self::TypeFact::*;

    let datatype = match (x, y) {
        (_, Any) => x,
        (Any, _) => y,
        (Only(a), Only(b)) if a == b => x,
        _ => bail!("Impossible to unify datatypes {:?} and {:?}.", x, y),
    };

    Ok(*datatype)
}

/// Attempts to unify two shape facts.
pub fn unify_shape(x: &ShapeFact, y: &ShapeFact) -> Result<ShapeFact> {
    use self::DimFact::*;
    use itertools::EitherOrBoth::{Both, Left, Right};
    use itertools::Itertools;

    let xi = x.dims.iter();
    let yi = y.dims.iter();

    let dimensions: Vec<_> = xi.zip_longest(yi)
        .map(|r| match r {
            Both(Any, Any) => Ok(Any),

            Both(Only(i), Any) | Both(Any, Only(i)) => Ok(Only(*i)),

            Both(Only(i), Only(j)) if i == j => Ok(Only(*i)),
            Both(Only(i), Only(j)) => bail!("Impossible to unify dimensions {:?} and {:?}.", i, j),

            Left(d) if y.open => Ok(*d),
            Right(d) if x.open => Ok(*d),

            Left(_) | Right(_) => bail!(
                "Impossible to unify closed shapes of different rank (found {:?} and {:?}).",
                x,
                y
            ),
        })
        .collect::<Result<_>>()?;

    if x.open && y.open {
        Ok(ShapeFact::open(dimensions))
    } else {
        Ok(ShapeFact::closed(dimensions))
    }
}

/// Attempts to unify two value facts.
pub fn unify_value(x: &ValueFact, y: &ValueFact) -> Result<ValueFact> {
    use self::ValueFact::*;

    let value = match (x, y) {
        (_, Any) => x.clone(),
        (Any, _) => y.clone(),
        (Only(a), Only(b)) => if a == b {
            x.clone()
        } else {
            bail!("Impossible to unify values {:?} and {:?}.", x, y);
        },
    };

    Ok(value)
}

/// An edge of the analysed graph, annotated by a fact.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub id: usize,
    pub from_node: Option<usize>,
    pub from_out: usize,
    pub to_node: Option<usize>,
    pub fact: TensorFact,
}

/// A graph analyser, along with its current state.
pub struct Analyser {
    // The original output.
    pub output: usize,

    // The graph being analysed.
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    prev_edges: Vec<Vec<usize>>,
    next_edges: Vec<Vec<usize>>,

    // The execution plan and unused nodes.
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
    pub fn new(model: Model, output: usize) -> Result<Analyser> {
        let nodes = model.nodes;
        let mut edges = vec![];
        let mut prev_edges = vec![Vec::new(); nodes.len() + 1];
        let mut next_edges = vec![Vec::new(); nodes.len() + 1];

        for node in &nodes {
            for input in &node.inputs {
                let id = edges.len();

                edges.push(Edge {
                    id,
                    from_node: Some(input.0),
                    from_out: input.1.unwrap_or(0),
                    to_node: Some(node.id),
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
            fact: TensorFact::new(),
        });

        next_edges[output].push(special_edge_id);

        // Compute an execution plan for the graph.
        let plan = Plan::for_nodes(&nodes, &[output])?.order;
        let current_pass = 0;
        let current_step = 0;
        let current_direction = true;

        info!("Using execution plan {:?}.", plan);

        Ok(Analyser {
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

    /// Returns a model from the analyser.
    pub fn into_model(self) -> Model {
        unimplemented!()
    }

    /// Returns a JSON representation of the graph from the analyser.
    #[cfg(feature = "serialize")]
    pub fn as_json(&self) -> String {
        serde_json::to_string(&(&self.nodes, &self.edges)).unwrap()
    }

    /// Computes a new execution plan for the graph.
    pub fn reset_plan(&mut self) -> Result<()> {
        self.plan = Plan::for_nodes(&self.nodes, &[self.output])?.order;
        Ok(())
    }

    /// Removes the nodes and edges which are not part of the execution plan.
    pub fn remove_unused(&mut self) {
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

                self.prev_edges[i - deleted].iter().for_each(|&j| edge_used[j] = true);
                self.next_edges[i - deleted].iter().for_each(|&j| edge_used[j] = true);
            }
        }

        info!("Deleted {:?} unused nodes.", deleted);

        // Update the nodes and edges to use the new indices.
        for node in &mut self.nodes {
            node.id = node_mapping[node.id].unwrap();
            node.inputs.iter_mut().for_each(|i| i.0 = node_mapping[i.0].unwrap());
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

        info!("Deleted {:?} unused edges.", deleted);

        // Update the adjacency lists to use the new indices.
        for i in 0..self.nodes.len() {
            self.prev_edges[i].iter_mut().for_each(|j| *j = edge_mapping[*j].unwrap());
            self.next_edges[i].iter_mut().for_each(|j| *j = edge_mapping[*j].unwrap());
        }
    }

    /// Runs the entire analysis at once.
    pub fn run(&mut self) -> Result<()> {
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
            if self.run_step()? {
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
            if self.run_step()? {
                changed = true;
            }
        }

        Ok(changed)
    }

    /// Runs a single step of the analysis.
    pub fn run_step(&mut self) -> Result<bool> {
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
        let node = if self.current_direction {
            &self.nodes[self.plan[self.current_step]]
        } else {
            &self.nodes[self.plan[self.plan.len() - 1 - self.current_step]]
        };

        debug!(
            "Starting step for {} ({}) [pass={:?}, direction={:?}, step={:?}].",
            node.name, node.op_name, self.current_pass, self.current_direction, self.current_step,
        );

        let (source, target) = if self.current_direction {
            (&self.prev_edges, &self.next_edges)
        } else {
            (&self.next_edges, &self.prev_edges)
        };

        let inferred = {
            let sources: Vec<_> = source[node.id]
                .iter()
                .map(|&i| &self.edges[i].fact)
                .collect();

            let inferred = if self.current_direction {
                node.op.infer_forward(sources)?
            } else {
                node.op.infer_backward(sources)?
            };

            if inferred.is_none() {
                return Ok(false);
            }

            inferred.unwrap()
        };

        let mut changed = false;

        // TODO(liautaud): For now, we will assume that forward inference only
        // produces a single output. We need to know this because several nodes
        // might want to consume that single output, so we must copy it instead
        // of expecting the node to produce several copies itself.
        for (i, &j) in target[node.id].iter().enumerate() {
            let fact = if self.current_direction {
                if inferred.len() > 1 {
                    panic!("Forward inference should not produce more than one output.");
                }

                &inferred[0]
            } else {
                &inferred[i]
            };

            let unified = unify(fact, &self.edges[j].fact)?;
            if unified != self.edges[j].fact {
                self.edges[j].fact = unified;
                changed = true;
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
