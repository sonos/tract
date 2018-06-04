use Node;
use Tensor;
use ops::OpBuilder;
use tfpb::node_def::NodeDef;
use tfpb::attr_value::AttrValue;
use super::Analyser;
use super::Result;

/// All constant tensors with an area lower than COPY_THRESHOLD will be
/// replaced with a constant node containing a copy of that tensor.
const COPY_THRESHOLD: usize = 100;

#[derive(Debug)]
pub enum Element {
    Node(usize),
    Edge(usize),
}

#[derive(Debug)]
pub struct Component {
    pub elements: Vec<Element>,
    pub outputs: Vec<usize>
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
    let mut colored = vec![false; is_node_const.len()];
    let mut stack = vec![];

    for (node, &is_const) in is_node_const.iter().enumerate() {
        if is_const && !colored[node] {
            let mut component = Component {
                elements: vec![],
                outputs: vec![]
            };

            stack.push(node);

            while let Some(node) = stack.pop() {
                if !is_node_const[node] {
                    continue;
                }

                colored[node] = true;
                component.elements.push(Element::Node(node));

                for &edge in &analyser.prev_edges[node] {
                    if !is_edge_const[edge] {
                        continue;
                    }

                    component.elements.push(Element::Edge(edge));
                    stack.push(analyser.edges[edge].from_node);
                }

                for &edge in &analyser.next_edges[node] {
                    if !is_edge_const[edge] {
                        continue;
                    }

                    component.elements.push(Element::Edge(edge));
                    stack.push(analyser.edges[edge].to_node);
                }
            }

            components.push(component);
        }
    }

    Ok(components)
}

/// Computes the lowest common ancestor of the sinks of a connected component.
fn lowest_common_ancestor() {
    unimplemented!()
}

/// Creates a new Const node with the given Tensor value.
fn build_const_node(id: usize, name: String, tensor: &Tensor) -> Node {
    let op_builder = OpBuilder::new();
	let mut node_def = NodeDef::new();

    node_def.set_name(name.clone());
    node_def.set_op("Const".to_string());

    let mut dtype = AttrValue::new();
    dtype.set_field_type(tensor.datatype());
    node_def.mut_attr().insert("dtype".to_string(), dtype);

    let mut value = AttrValue::new();
    value.set_tensor(tensor.to_pb().unwrap());
    node_def.mut_attr().insert("value".to_string(), value);

	Node {
	    id,
	    name,
	    op_name: "Const".to_string(),
	    inputs: vec![],
	    op: op_builder
	        .build(&node_def)
	        .unwrap()
	}
}


/// Prunes the constant nodes and edges from the given graph.
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
///                                     +---------------------+
///                                 +--^+ Simple operation 1  +-->
///             +---------------+   |   +---------------------+
///             | Const (large) +---+
///             +---------------+   |   +---------------------+
///                                 +--^+ Simple operation 2  +-->
///                                     +---------------------+
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
pub fn prune_constants(analyser: &mut Analyser) -> Result<()> {
    let components: Vec<Component> = connected_components(analyser)?;

    info!("Constant connected components: {:?}", components);

    // for component in components {
    //     // let preserve_edges = vec![];

    // 	for i in component.outputs {
    // 		let tensor = analyser.edges[i].fact.value.concretize().unwrap();
    // 		let area: usize = tensor.shape().iter().product();

    //         // TODO(liautaud): Implement the other strategy.
    // 		// if area <= COPY_THRESHOLD {
    // 			let id = analyser.nodes.len();
    // 			let name = format!("generated_{}", i).to_string();
    // 			let node = build_const_node(id, name, tensor);

    //             analyser.nodes.push(Some(&node));
    //             analyser.edges[i].from_node = id;
    //             analyser.edges[i].from_out = 0;
    // 		// }

    //         // TODO(liautaud): Delete unused nodes for each component.
    // 	}
    // }

    // // TODO(liautaud): Implement the other strategy.

    // // Prune the remaining nodes and edges.
    // analyser.clear_plan();

    // // let constant_nodes = is_node_const.into_iter()
    // //     .enumerate()
    // //     .filter(|&(_, is_const)| is_const)
    // //     .map(&|(i, _)| i);

    // // for i in constant_nodes {
    // //     analyser.remove_node(i);
    // // }

    // // let constant_edges = is_edge_const.into_iter()
    // //     .enumerate()
    // //     .filter(|&(_, is_const)| is_const)
    // //     .map(&|(i, _)| i);

    // // for i in constant_edges {
    // //     analyser.remove_edge(i);
    // // }

    Ok(())
}
