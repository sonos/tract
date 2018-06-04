/// Computes the constant underlying graph of the given graph.
///
/// The constant underlying graph G is constructed using these rules:
/// - If an edge has a constant value according to the analyser, it is in G.
/// - If all the outgoing edges of a node are in G, that node is also in G.
/// - If an edge in G has no target, a new node is added to G and becomes
///   the target of that edge. This new node is called a sink.
fn constant_subgraph(analyser: &Analyser) -> (Vec<bool>, Vec<bool>) {
	let edge_belongs = analyser.edges.iter()
		.map(|e| e.fact.value.is_concrete())
		.collect();

	let node_belongs = analyser.next_edges.iter()
		.map(|next| next.len() > 0 && next.all(|i| edge_belongs[i]))
		.collect();

	// TODO(liautaud): How do we represent sinks?

	(node_belongs, edge_belongs)
}


/// Computes all the (undirected) connected components of a graph.
fn connected_components() {
	unimplemented!()
}


/// Computes the lowest common ancestor of the sinks of a connected component.
fn lowest_common_ancestor() {
	unimplemented!()
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
pub fn prune_constants(analyser: &mut Analyser) {
	unimplemented!()
}