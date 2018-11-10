use bit_set;
use model::Node;
use TractResult;

pub fn eval_order(model: &super::Model) -> TractResult<Vec<usize>> {
    let targets = model
        .outputs()?
        .iter()
        .map(|n| n.node)
        .collect::<Vec<usize>>();
    eval_order_for_nodes(model.nodes(), &targets)
}

pub fn eval_order_for_nodes(nodes: &[Node], targets: &[usize]) -> TractResult<Vec<usize>> {
    let mut done = bit_set::BitSet::with_capacity(nodes.len());
    let mut needed: Vec<usize> = vec![];
    let mut order: Vec<usize> = vec![];
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
    use model::dsl::ModelDsl;
    use model::*;
    use ops::math::Add;
    use *;

    #[test]
    fn test_simple() {
        let mut model = Model::default();
        model.add_source("a").unwrap();
        model.chain("add", Box::new(Add::default())).unwrap();
        model.add_const("b", DtArray::from(12.0f32).into()).unwrap();
        model
            .add_edge(OutletId::new(2, 0), InletId::new(1, 1))
            .unwrap();
        assert_eq!(model.eval_order().unwrap(), vec!(0, 2, 1));
    }

    #[test]
    fn test_diamond() {
        let mut model = Model::default();
        model.add_source("a").unwrap();
        model.chain("add", Box::new(Add::default())).unwrap();
        model
            .add_edge(OutletId::new(0, 0), InletId::new(1, 1))
            .unwrap();
        assert_eq!(model.eval_order().unwrap(), vec!(0, 1));
    }
}
