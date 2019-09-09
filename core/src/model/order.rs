//! Evaluation order for nodes.
use crate::internal::*;
use bit_set;
use std::fmt::{Debug, Display};

/// Find an evaluation order for a model, using its default inputs and outputs
/// as boundaries.
pub fn eval_order<
    TI: TensorInfo + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
>(
    model: &super::ModelImpl<TI, O>,
) -> TractResult<Vec<usize>> {
    let inputs = model.input_outlets()?.iter().map(|n| n.node).collect::<Vec<usize>>();
    let targets = model.output_outlets()?.iter().map(|n| n.node).collect::<Vec<usize>>();
    eval_order_for_nodes(model.nodes(), &inputs, &targets)
}

/// Find a working evaluation order for a list of nodes.
pub fn eval_order_for_nodes<TI: TensorInfo, O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op>>(
    nodes: &[BaseNode<TI, O>],
    inputs: &[usize],
    targets: &[usize],
) -> TractResult<Vec<usize>> {
    let mut done = bit_set::BitSet::with_capacity(nodes.len());
    let mut order: Vec<usize> = vec![];
    for &target in targets {
        if done.contains(target) {
            continue;
        }
        let mut current_stack: Vec<(usize, usize)> = vec![(target, 0)];
        let mut pending = bit_set::BitSet::with_capacity(nodes.len());
        while let Some((current_node, current_input)) = current_stack.pop() {
            if inputs.contains(&current_node)
                || current_input
                    == nodes[current_node].inputs.len() + nodes[current_node].control_inputs.len()
            {
                order.push(current_node);
                done.insert(current_node);
                pending.remove(current_node);
            } else {
                let precursor = if current_input < nodes[current_node].inputs.len() {
                    nodes[current_node].inputs[current_input].node
                } else {
                    nodes[current_node].control_inputs
                        [current_input - nodes[current_node].inputs.len()]
                };
                if done.contains(precursor) {
                    current_stack.push((current_node, current_input + 1));
                } else if pending.contains(precursor) {
                    if log_enabled!(log::Level::Debug) {
                        debug!("Loop detected:");
                        current_stack
                            .iter()
                            .skip_while(|s| s.0 != precursor)
                            .for_each(|n| debug!("  {}", nodes[n.0]));
                    }
                    bail!("Loop detected")
                } else {
                    pending.insert(precursor);
                    current_stack.push((current_node, current_input));
                    current_stack.push((precursor, 0));
                }
            }
        }
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use crate::internal::*;
    use crate::ops::math;

    #[test]
    fn simple() {
        let mut model = ModelImpl::default();
        model.add_source_default("a").unwrap();
        model.chain_default("add", math::add::bin()).unwrap();
        model.add_const("b", Tensor::from(12.0f32)).unwrap();
        model.add_edge(OutletId::new(2, 0), InletId::new(1, 1)).unwrap();
        model.auto_outputs().unwrap();
        assert_eq!(model.eval_order().unwrap(), vec!(0, 2, 1));
    }

    #[test]
    fn diamond() {
        let mut model = ModelImpl::default();
        model.add_source_default("a").unwrap();
        model.chain_default("add", math::add::bin()).unwrap();
        model.add_edge(OutletId::new(0, 0), InletId::new(1, 1)).unwrap();
        model.auto_outputs().unwrap();
        assert_eq!(model.eval_order().unwrap(), vec!(0, 1));
    }

    #[test]
    fn dodge_loop() {
        let mut model = ModelImpl::default();
        model.add_source_default("a").unwrap();
        let add = model.chain_default("add", math::add::bin()).unwrap();
        let neg = model.chain_default("neg", math::add::bin()).unwrap();
        model.add_edge(OutletId::new(neg, 0), InletId::new(add, 1)).unwrap();
        model.set_output_outlets(&tvec!(OutletId::new(neg, 0))).unwrap();
        let (rx, tx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            rx.send(model.eval_order()).unwrap();
        });
        assert!(tx.recv_timeout(std::time::Duration::from_secs(1)).unwrap().is_err());
    }
}
