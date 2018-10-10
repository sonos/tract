use bit_set;
use model::Node;
use TfdResult;

pub fn eval_order(model: &super::Model) -> TfdResult<Vec<usize>> {
    let targets = model.outputs()?.iter().map(|n| n.node).collect::<Vec<usize>>();
    eval_order_for_nodes(model.nodes(), &targets)
}

pub fn eval_order_for_nodes(nodes: &[Node], targets: &[usize]) -> TfdResult<Vec<usize>> {
    let mut done = bit_set::BitSet::with_capacity(nodes.len());
    let mut needed:Vec<usize> = vec!();
    let mut order:Vec<usize> = vec!();
    for &t in targets {
        needed.push(t);
    }
    while let Some(&node) = needed.last() {
        if done.contains(node) {
            needed.pop();
            continue;
        }
        if nodes[node].inputs.iter().all(|i| done.contains(i.node)) {
            order.push(node);
            needed.pop();
            done.insert(node);
        } else {
            for input in nodes[node].inputs.iter().rev() {
                if !done.contains(input.node) {
                    needed.push(input.node);
                }
            }
        }
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use *;
    use model::*;
    use model::dsl::ModelDsl;

    #[test]
    fn test_simple() {
        let mut model = Model::default();
        model.add_source("a").unwrap();
        model.chain("add", ::ops::math::Add::default()).unwrap();
        model.add_const("b", Tensor::from(12.0f32)).unwrap();
        model.add_edge(OutletId::new(2,0), InletId::new(1,1)).unwrap();
        assert_eq!(model.eval_order().unwrap(), vec!(0, 2, 1));
    }

    #[test]
    fn test_diamond() {
        let mut model = Model::default();
        model.add_source("a").unwrap();
        model.chain("add", ::ops::math::Add::default()).unwrap();
        model.add_edge(OutletId::new(0,0), InletId::new(1,1)).unwrap();
        assert_eq!(model.eval_order().unwrap(), vec!(0, 1));
    }
}

/*
pub fn eval_order_for_nodes(nodes: &[Node], targets: &[usize]) -> TfdResult<Vec<usize>> {
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
*/
