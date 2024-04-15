//! Evaluation order for nodes.
use crate::internal::*;
use bit_set::BitSet;
use std::collections::BinaryHeap;
use std::default;
use std::fmt::{Debug, Display};
use std::iter::once;
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
    _model_inputs: &[usize],
    model_outputs: &[usize],
    more_dependencies: &[(usize, usize)],
) -> TractResult<Vec<usize>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    let mut ups = vec![tvec!(); nodes.len()];
    let mut downs = vec![tvec!(); nodes.len()];
    for (ix, node) in nodes.iter().enumerate() {
        for input in &node.inputs {
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
    let costs: Vec<usize> = nodes
        .iter()
        .map(|node| {
            node.outputs
                .iter()
                .map(|o| {
                    o.fact
                        .to_typed_fact()
                        .map(|f| {
                            f.datum_type.size_of()
                                * f.shape
                                    .as_concrete()
                                    .map(|dims| dims.iter().product())
                                    .unwrap_or(1)
                        })
                        .unwrap_or(1)
                })
                .sum()
        })
        .collect_vec();

    struct DFS {
        nodes: usize,
        costs: Vec<usize>,
        ups: Vec<TVec<usize>>,
        downs: Vec<TVec<usize>>,
        outputs: Vec<usize>,
    }

    let dfs = DFS { nodes: nodes.len(), costs, ups, downs, outputs: model_outputs.to_vec() };

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    struct Path {
        order: Vec<usize>,
        alive: Vec<usize>,
        space_now: usize,
        space_max: usize,
        score: usize,
    }

    impl Path {
        /*
        fn candidates(&self, dfs: &DFS) -> Vec<usize> {
        (0..dfs.nodes)
        .filter(|node| {
        !self.order.contains(node)
        && dfs.ups[*node].iter().all(|up| self.alive.contains(up))
        })
        .collect()
        }

        fn follow(&self, dfs: &DFS, next: usize) -> Path {
        let it = self.follow_one(dfs, next);
        if let [one] = it.candidates(dfs)[..] {
        it.follow(dfs, one)
        } else {
        it
        }
        }
        */
        fn follow_one(&self, dfs: &DFS, next: usize) -> Path {
            let order = self.order.iter().copied().chain(once(next)).collect_vec();
            let alive = self
                .alive
                .iter()
                .copied()
                .chain(once(next))
                .filter(|n| dfs.downs[*n].iter().any(|down| !order.contains(down)))
                .sorted()
                .collect_vec();
            let space_now = alive.iter().map(|a| dfs.costs[*a]).sum::<usize>();
            let space_max = space_now.max(self.space_max);
            let score = self.score + space_now;
            Path { order, alive, space_now, space_max, score }
        }
    }

    impl PartialOrd for Path {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Path {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.space_max
                .cmp(&other.space_max)
                .then(self.score.cmp(&other.score))
                .then(self.order.cmp(&other.order))
        }
    }

    type Cache = HashMap<BitSet, Path>;

    /*
    impl DFS {
    fn reach<'c>(&self, cache: &'c mut Cache, to: &BitSet) -> &'c Path {
    if !cache.contains_key(to) {
    let path = if to.len() == 0 {
    Path::default()
    } else {
    to.iter()
    .map(|last| {
    let mut previous = to.clone();
    previous.remove(last);
    previous.extend(self.ups[last].iter().copied());
    let mut path = self.reach(cache, &previous).clone();
    if !path.order.contains(&last) {
    path = path.follow_one(self, last);
    }
    path
    })
    .min()
    .unwrap()ow
    };
    cache.insert(to.clone(), path);
    }
    &cache[to]
    }
    }
    Ok(dfs.reach(&mut Default::default(), &dfs.outputs.iter().copied().collect()).order.clone())
    */

    let mut done = HashMap::<BitSet, Path>::default();
    let mut todo: Vec<BitSet> = vec![];
    let target:BitSet = dfs.outputs.iter().copied().collect();
    done.insert(BitSet::default(), Path::default());
    todo.push(target.clone());
    while let Some(current) = todo.pop() {
  //      println!("Stack: {todo:?}");
        if done.contains_key(&current) {
            continue;
        }
 //       println!("Computing: {current:?}");
        let mut best: Option<Path> = None;
        let mut incomplete = false;
        for last in current.iter() {
            let mut previous = current.clone();
            previous.remove(last);
            previous.extend(dfs.ups[last].iter().copied());
            if let Some(prec) = done.get(&previous) {
                let path = prec.follow_one(&dfs, last);
                if let Some(b) = best {
                    best = Some(b.min(path));
                } else {
                    best = Some(path)
                }
            } else {
                if !incomplete {
//                    println!("  Pushing {current:?} again");
                    todo.push(current.clone());
                }
 //               println!("  Pushing {previous:?}");
                todo.push(previous);
                incomplete = true;
            }
        }
        if !incomplete {
//            println!("######## {current:?} ====> {best:?}");
            done.insert(current, best.unwrap());
        }
    }

    Ok(done[&target].order.clone())

    /*
    let mut to_explore = BinaryHeap::<Path>::new();
    to_explore.push(Path::default());
    let mut best = eval_order_for_nodes(nodes, _model_inputs, model_outputs, more_dependencies)?
    .iter()
    .fold(Path::default(), |path, next| path.follow_one(&dfs, *next));
    while let Some(from) = to_explore.pop() {
    println!("best {} ::: [{}] ::: {:?}", best.space_max, to_explore.len(), from);
    if dfs.outputs.iter().all(|t| from.order.contains(t)) {
    best = from;
    continue;
    }
    for c in from.candidates(&dfs) {
    let next = from.follow(&dfs, c);
    if next.space_max < best.space_max {
    to_explore.push(next);
    }
    }
    }
    Ok(best.order)
    */
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
