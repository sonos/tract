use super::Analyser;
use super::Result;
use ops::OpBuilder;
use tfpb;
use tfpb::tensor::TensorProto;
use Node;

/// All constant tensors with an area lower than COPY_THRESHOLD will be
/// replaced with a constant node containing a copy of that tensor.
// const COPY_THRESHOLD: usize = 100;

#[derive(Debug)]
pub enum Element {
    Node(usize),
    Edge(usize),
}

#[derive(Debug)]
pub struct Component {
    pub elements: Vec<Element>,
    pub outputs: Vec<usize>,
}

/// Computes all the connected components of the constant underlying graph.
///
/// The constant underlying graph G is constructed using these rules:
/// - If an edge has a constant value according to the analyser, it is in G.
/// - If all the outgoing edges of a node are in G, that node is also in G.
/// - If an edge in G has no target, it is called an "output".
pub fn connected_components(analyser: &Analyser) -> Result<Vec<Component>> {
    let is_edge_const: Vec<bool> = analyser
        .edges
        .iter()
        .map(|e| e.fact.value.is_concrete())
        .collect();

    let is_node_const: Vec<bool> = analyser
        .next_edges
        .iter()
        .map(|next| next.len() > 0 && next.iter().all(|i| is_edge_const[*i]))
        .collect();

    let mut components = vec![];
    let mut is_node_colored = vec![false; analyser.nodes.len()];
    let mut is_edge_colored = vec![false; analyser.edges.len()];
    let mut stack = vec![];

    macro_rules! process_edges {
        ($from:ident, $field:ident, $component:expr, $node:expr) => {{
            for &edge in &analyser.$from[$node] {
                if !is_edge_const[edge] || is_edge_colored[edge] {
                    continue;
                }

                is_edge_colored[edge] = true;
                $component.elements.push(Element::Edge(edge));

                let target = analyser.edges[edge].$field;
                if target.is_none() {
                    continue;
                }

                if !is_node_const[target.unwrap()] {
                    $component.outputs.push(edge);
                } else {
                    stack.push(target.unwrap());
                }
            }
        }};
    };

    for (node, &is_const) in is_node_const.iter().enumerate() {
        if is_const && !is_node_colored[node] {
            let mut component = Component {
                elements: vec![],
                outputs: vec![],
            };

            stack.push(node);

            while let Some(node) = stack.pop() {
                if !is_node_const[node] || is_node_colored[node] {
                    continue;
                }

                is_node_colored[node] = true;
                component.elements.push(Element::Node(node));

                process_edges!(prev_edges, from_node, component, node);
                process_edges!(next_edges, to_node, component, node);
            }

            components.push(component);
        }
    }

    Ok(components)
}

/// Computes the lowest common ancestor of the sinks of a connected component.
#[allow(dead_code)]
fn lowest_common_ancestor() {
    unimplemented!()
}

/// Creates a new Const node with the given Tensor value.
fn build_const_node(id: usize, name: String, tensor: TensorProto) -> Node {
    let node_def = tfpb::node()
        .name(name.clone())
        .op("Const")
        .attr("dtype", tensor.get_dtype())
        .attr("value", tensor);

    Node {
        id,
        name,
        op_name: "Const".to_string(),
        inputs: vec![],
        op: OpBuilder::new().build(&node_def).unwrap(),
    }
}

/// Detaches the constant nodes and edges from the given graph.
///
/// The following algorithm is used:
/// 1. Compute the constant underlying graph of the given graph.
/// 2. Compute the undirected connected components of that underlying graph.
/// 3. Choose a pruning strategy and apply it to each connected component.
///
/// There are several pruning strategies to choose from:
/// - The simplest is to prune all nodes but the sinks of each component, and
///   to replace the latter with Const nodes. This might however increase the
///   size of the model dramatically in cases like the one below, where we'll
///   end up storing two large constants instead of one while only getting a
///   neglectible performance boost from the operation.
///
/// ```text
///                                     +---------------------+
///                                 +--^+ Simple operation 1  +-->
///             +---------------+   |   +---------------------+
///             | Const (large) +---+
///             +---------------+   |   +---------------------+
///                                 +--^+ Simple operation 2  +-->
///                                     +---------------------+
/// ```
///
/// - We could also search for the lowest common ancestor of all the sinks in
///   each connected component, and prune every node and edge that isn't part
///   of a path between that ancestor and a sink. If no such ancestor exists,
///   we don't do anything. This way we guarantee that we don't increase the
///   size of the model, but we might miss some optimisations.
///
/// - Ideally, we would use a heuristic to find a middle ground between the
///   two strategies. This would allow the duplication of constants if the
///   size or performance gained from pruning compensates the size loss.
pub fn propagate_constants(analyser: &mut Analyser) -> Result<()> {
    let components: Vec<Component> = connected_components(analyser)?;
    info!("Detected {:?} connected components.", components.len());

    for component in components {
        for i in component.outputs {
            let edge = &mut analyser.edges[i];
            let tensor = edge
                .fact
                .value
                .concretize()
                .unwrap()
                .to_pb()
                .unwrap();

            // TODO(liautaud): Implement the other strategy.
            // let area: usize = tensor.shape().iter().product();
            // if area <= COPY_THRESHOLD {
            let node_id = analyser.nodes.len();
            let node_name = format!("generated_{}", i).to_string();
            let node = build_const_node(node_id, node_name, tensor);
            let old_node_id = edge.from_node.unwrap();

            // Detach the edge from its previous source.
            {
                let successors = &mut analyser.next_edges[old_node_id];
                let position = successors.iter().position(|&i| i == edge.id).unwrap();
                successors.remove(position);
            };

            // Detach the target node from its previous source.
            {
                let predecessors = &mut analyser.nodes[edge.to_node.unwrap()].inputs;
                let position = predecessors.iter().position(|&(i, _)| i == old_node_id).unwrap();
                predecessors[position] = (node_id, None);
            }

            // Attach the edge to its new source.
            edge.from_node = Some(node_id);
            analyser.prev_edges.push(vec![]);
            analyser.next_edges.push(vec![edge.id]);
            analyser.nodes.push(node);
        }
    }

    analyser.reset_plan()?;

    Ok(())
}
