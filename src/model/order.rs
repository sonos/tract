use bit_set;
use model::Node;
use Result;

pub fn eval_order_for_nodes(nodes: &[Node], targets: &[usize]) -> Result<Vec<usize>> {
    let mut order: Vec<usize> = Vec::new();
    let mut done = bit_set::BitSet::with_capacity(nodes.len());
    let mut needed = bit_set::BitSet::with_capacity(nodes.len());
    for &t in targets {
        needed.insert(t);
    }
    loop {
        let mut done_something = false;
        let mut missing = needed.clone();
        missing.difference_with(&done);
        for node_id in missing.iter() {
            let mut computable = true;
            let node = &nodes[node_id];
            for i in node.inputs.iter() {
                if !done.contains(i.node) {
                    computable = false;
                    done_something = true;
                    needed.insert(i.node);
                }
            }
            if computable {
                done_something = true;
                order.push(node_id);
                done.insert(node_id);
            }
        }
        if !done_something {
            break;
        }
    }
    for &t in targets {
        if !done.contains(t) {
            let node = &nodes[t];
            Err(format!("Could not plan for node {}", node.name))?
        }
    }
    Ok(order)
}
