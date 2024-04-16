//! Evaluation order for nodes.
use crate::internal::*;
use bit_set::BitSet;
use std::collections::VecDeque;
use std::fmt::{Debug, Display};
use tract_data::itertools::Itertools;

/// Find an evaluation order for a model, using its default inputs and outputs
/// as boundaries.
pub fn eval_order<F, O>(model: &super::Graph<F, O>) -> TractResult<Vec<usize>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    let inputs = model.input_outlets()?.iter().map(|n| n.node).collect::<Vec<usize>>();
    let targets = model.output_outlets()?.iter().map(|n| n.node).collect::<Vec<usize>>();
    eval_order_for_nodes(model.nodes(), &inputs, &targets, &[])
}

/// Find a working evaluation order for a list of nodes.
/// This algorithm starts from the outputs, so it will only compute what is necessary.
pub fn eval_order_for_nodes<F, O>(
    nodes: &[Node<F, O>],
    model_inputs: &[usize],
    model_outputs: &[usize],
    more_dependencies: &[(usize, usize)],
) -> TractResult<Vec<usize>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    let mut done = BitSet::with_capacity(nodes.len());
    let mut order: Vec<usize> = vec![];
    for &model_target in model_outputs {
        if done.contains(model_target) {
            continue;
        }
        let mut current_stack: Vec<(usize, usize)> = vec![(model_target, 0)];
        let mut pending = BitSet::with_capacity(nodes.len());
        while let Some((current_node, current_input)) = current_stack.pop() {
            let deps_from_inputs = nodes[current_node].inputs.len();
            let all_deps_count =
                deps_from_inputs + more_dependencies.iter().filter(|a| a.0 == current_node).count();
            if model_inputs.contains(&current_node) || current_input == all_deps_count {
                order.push(current_node);
                done.insert(current_node);
                pending.remove(current_node);
            } else {
                let precursor: usize = nodes[current_node]
                    .inputs
                    .iter()
                    .filter(|n| nodes[n.node].inputs.len() > 0)
                    .map(|n| n.node)
                    .chain(more_dependencies.iter().filter(|a| a.0 == current_node).map(|n| n.1))
                    .chain(
                        nodes[current_node]
                            .inputs
                            .iter()
                            .filter(|n| nodes[n.node].inputs.len() == 0)
                            .map(|n| n.node),
                    )
                    .nth(current_input)
                    .unwrap();
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

/// Find an evaluation order for a list of model trying to minimize memory occupation.
pub fn eval_order_opt_ram<F, O>(model: &super::Graph<F, O>) -> TractResult<Vec<usize>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    let inputs = model.input_outlets()?.iter().map(|n| n.node).collect::<Vec<usize>>();
    let targets = model.output_outlets()?.iter().map(|n| n.node).collect::<Vec<usize>>();
    eval_order_opt_ram_for_nodes(model.nodes(), &inputs, &targets, &[])
}

/// Find an evaluation order for a list of nodes trying to minimize memory occupation.
pub fn eval_order_opt_ram_for_nodes<F, O>(
    nodes: &[Node<F, O>],
    model_inputs: &[usize],
    model_outputs: &[usize],
    more_dependencies: &[(usize, usize)],
) -> TractResult<Vec<usize>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    let tocompute: BitSet =
        eval_order_for_nodes(nodes, model_inputs, model_outputs, more_dependencies)?
            .into_iter()
            .collect();

    let mut ups = vec![tvec!(); nodes.len()];
    let mut downs = vec![tvec!(); nodes.len()];
    for ix in tocompute.iter() {
        for input in &nodes[ix].inputs {
            if !ups[ix].contains(&input.node) {
                ups[ix].push(input.node);
                downs[input.node].push(ix);
            }
        }
    }
    for (down, up) in more_dependencies {
        if !ups[*down].contains(up) {
            ups[*down].push(*up);
            downs[*up].push(*down);
        }
    }

    struct Dfs {
        ups: Vec<TVec<usize>>,
        downs: Vec<TVec<usize>>,
    }

    let dfs = Dfs { ups, downs };

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    struct Path {
        order: Vec<usize>,
        done: BitSet,
        alive: Vec<usize>,
    }

    impl Path {
        fn follow_one(&mut self, dfs: &Dfs, next: usize) {
            assert!(!self.done.contains(next));
            self.order.push(next);
            self.done.insert(next);
            self.alive.push(next);
            self.alive.retain(|n| dfs.downs[*n].iter().any(|down| !self.done.contains(*down)))
        }

        fn missing_upstream_starters(&self, dfs: &Dfs, from: usize) -> Vec<usize> {
            let mut found = vec![];
            let mut done = self.done.clone();
            let mut todo = VecDeque::<usize>::new();
            todo.push_back(from);
            done.insert(from);
            while let Some(next) = todo.pop_front() {
                if dfs.ups[next].len() == 0 {
                    found.push(next);
                }
                for up in &dfs.ups[next] {
                    if done.insert(*up) {
                        todo.push_back(*up);
                    }
                }
            }
            assert!(found.len() > 0);
            found
        }
    }

    let mut done: Path = Path::default();
    for i in model_inputs {
        if tocompute.contains(*i) {
            done.follow_one(&dfs, *i);
        }
    }

    while !model_outputs.iter().all(|o| done.done.contains(*o)) {
        let candidates: Vec<usize> = done
            .alive
            .iter()
            .copied()
            .flat_map(|n| dfs.downs[n].iter())
            .copied()
            .filter(|n| !done.done.contains(*n))
            .sorted()
            .unique()
            .collect_vec();
        let next = if let Some(next) =
            candidates.iter().copied().find(|n| dfs.ups[*n].iter().all(|n| done.done.contains(*n)))
        {
            next
        } else if let Some(next) = candidates
            .iter()
            .map(|c| done.missing_upstream_starters(&dfs, *c))
            .min_by_key(|p| p.len())
            .map(|s| s[0])
        {
            next
        } else {
            tocompute
                .difference(&done.done)
                .find(|n| dfs.ups[*n].iter().all(|n| done.done.contains(*n)))
                .unwrap()
        };
        done.follow_one(&dfs, next);
    }

    Ok(done.order.clone())
}

#[cfg(test)]
mod tests {
    use crate::internal::*;
    use crate::ops::array::Gather;
    use crate::ops::math;

    #[test]
    fn simple() {
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact([1])).unwrap();
        let b = model.add_const("b", tensor1(&[12.0f32])).unwrap();
        let add = model.wire_node("add", math::add(), &[a, b]).unwrap()[0];
        model.auto_outputs().unwrap();
        assert_eq!(model.eval_order().unwrap(), vec!(a.node, b.node, add.node));
        assert_eq!(model.eval_order_opt_ram().unwrap(), vec!(a.node, b.node, add.node));
    }

    #[test]
    fn diamond() {
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact([1])).unwrap();
        let add = model.wire_node("add", math::add(), &[a, a]).unwrap()[0];
        model.auto_outputs().unwrap();
        assert_eq!(model.eval_order().unwrap(), vec!(a.node, add.node));
        assert_eq!(model.eval_order_opt_ram().unwrap(), vec!(a.node, add.node));
    }

    #[test]
    fn dodge_loop() {
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact([1])).unwrap();
        let add = model.wire_node("add", math::add(), &[a, a]).unwrap()[0];
        let neg = model.wire_node("neg", math::add(), &[add, a]).unwrap()[0];
        model.add_edge(neg, InletId::new(add.node, 1)).unwrap();
        model.set_output_outlets(&[neg]).unwrap();
        let cloned = model.clone();
        let (rx, tx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            rx.send(cloned.eval_order()).unwrap();
        });
        assert!(tx.recv_timeout(std::time::Duration::from_secs(1)).unwrap().is_err());
        let (rx, tx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            rx.send(model.eval_order_opt_ram()).unwrap();
        });
        assert!(tx.recv_timeout(std::time::Duration::from_secs(1)).unwrap().is_err());
    }

    #[test]
    fn opt_ram() -> TractResult<()> {
        let mut model = TypedModel::default();
        let b = model.add_const("b", tensor1(&[0i64; 1000]))?;
        let d = model.add_const("d", tensor1(&[0i64; 100]))?;
        let a = model.add_source("a", i32::fact([10]))?;
        let c = model.wire_node("c", Gather::new(0), &[a, b])?[0];
        let e = model.wire_node("e", Gather::new(0), &[c, d])?[0];
        model.set_output_outlets(&[e]).unwrap();
        eprintln!("{model}");
        assert!(&model.eval_order_opt_ram()?[2..] == &[c.node, d.node, e.node]);
        Ok(())
    }
}
