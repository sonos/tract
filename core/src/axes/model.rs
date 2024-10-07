use crate::internal::*;

#[derive(Debug, Clone, Default)]
pub struct OutletMap<T>(Vec<TVec<Option<T>>>);

impl<T: Clone> OutletMap<T> {
    fn insert(&mut self, outlet: OutletId, t: T) {
        if outlet.node >= self.0.len() {
            self.0.resize_with(outlet.node + 1, || tvec!());
        }
        let node = &mut self.0[outlet.node];
        if outlet.slot >= node.len() {
            node.resize(outlet.slot + 1, None);
        }
        node[outlet.slot] = Some(t)
    }
}

impl<T> OutletMap<T> {
    fn remove(&mut self, outlet: &OutletId) -> Option<T> {
        if let Some(node) = self.0.get_mut(outlet.node) {
            if let Some(slot) = node.get_mut(outlet.slot) {
                return slot.take();
            }
        }
        None
    }

    pub fn get(&self, outlet: &OutletId) -> Option<&T> {
        if let Some(node) = self.0.get(outlet.node) {
            if let Some(slot) = node.get(outlet.slot) {
                return slot.as_ref();
            }
        }
        None
    }

    pub fn keys(&self) -> OutletMapKeysIter<T> {
        OutletMapKeysIter(self, (0, 0).into())
    }
}

impl<'a, T: Clone> std::ops::Index<&'a OutletId> for OutletMap<T> {
    type Output = T;
    fn index(&self, index: &'a OutletId) -> &Self::Output {
        self.get(index).unwrap()
    }
}

pub struct OutletMapKeysIter<'a, T>(&'a OutletMap<T>, OutletId);

impl<T> std::iter::Iterator for OutletMapKeysIter<'_, T> {
    type Item = OutletId;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.1.node >= (self.0).0.len() {
                return None;
            }
            if self.1.slot >= (self.0).0[self.1.node].len() {
                self.1.slot = 0;
                self.1.node += 1;
                continue;
            }
            let current = self.1;
            self.1.slot += 1;
            if self.0.get(&current).is_some() {
                return Some(current);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct AxisTracking {
    pub creators: TVec<OutletId>,
    pub destructors: TVec<InletId>,
    pub outlets: OutletMap<usize>,
}

impl AxisTracking {
    pub fn for_outlet_and_axis(
        model: &TypedModel,
        outlet: OutletId,
        axis: usize,
    ) -> TractResult<Option<AxisTracking>> {
        let mut mapped_outlets = OutletMap::default();
        let mut todo = OutletMap::default();
        let mut creators = tvec!();
        let mut destructors = tvec!();
        mapped_outlets.insert(outlet, axis);
        todo.insert(outlet, ());
        while let Some(wire) = todo.keys().next() {
            todo.remove(&wire);
            let axis = mapped_outlets[&wire];
            let emiter_node = model.node(wire.node);
            let mut nodes = vec![];
            let (input_facts, output_facts) = model.node_facts(emiter_node.id)?;
            let map = emiter_node
                .op
                .axes_mapping(&input_facts, &output_facts)
                .with_context(|| format!("Computing axes mapping for {emiter_node}"))?;
            let info = map.axis((InOut::Out(wire.slot), axis)).with_context(|| {
                format!(
                    "Axes mapping for {} is {map}, need output axis {:?} from slot {}",
                    emiter_node, axis, wire.slot,
                )
            })?;

            if info.inputs.iter().any(|i| i.len() > 0) {
                nodes.push((wire.node, info.clone()));
            } else {
                creators.push(wire);
            };
            for succ in &emiter_node.outputs[wire.slot].successors {
                let succ_node = model.node(succ.node);
                let (input_facts, output_facts) = model.node_facts(succ_node.id)?;
                let map = succ_node.op.axes_mapping(&input_facts, &output_facts)?;
                let info = map.axis((InOut::In(succ.slot), axis)).with_context(|| {
                    format!(
                        "Axes mapping for {succ_node} is {map}, need input axis {:?} from slot {}",
                        axis, succ.slot,
                    )
                })?;
                if info.outputs.iter().any(|o| o.len() > 0) {
                    nodes.push((succ_node.id, info.clone()));
                } else {
                    destructors.push(*succ);
                };
            }
            let mut new_outlets = vec![];
            for (n, axes) in nodes {
                let node = model.node(n);
                for slot in 0..node.outputs.len() {
                    if let &[axis] = &*axes.outputs[slot] {
                        new_outlets.push((OutletId::new(n, slot), axis));
                    }
                }
                for slot in 0..node.inputs.len() {
                    if let &[axis] = &*axes.inputs[slot] {
                        new_outlets.push((node.inputs[slot], axis));
                    }
                }
            }
            for (outlet, axis) in new_outlets {
                if let Some(prev) = mapped_outlets.get(&outlet) {
                    if *prev != axis {
                        return Ok(None);
                    }
                } else {
                    mapped_outlets.insert(outlet, axis);
                    todo.insert(outlet, ());
                }
            }
        }
        Ok(Some(AxisTracking { creators, destructors, outlets: mapped_outlets }))
    }
}

pub fn full_axis_tracking(model: &TypedModel) -> TractResult<Vec<AxisTracking>> {
    let mut axes: Vec<AxisTracking> = vec![];
    for node in model.eval_order()? {
        for slot in 0..model.node(node).outputs.len() {
            let outlet = OutletId::new(node, slot);
            let input_fact = model.outlet_fact(outlet)?;
            'axis: for axis in 0..input_fact.rank() {
                if axes.iter().any(|tracking| tracking.outlets.get(&outlet) == Some(&axis)) {
                    continue 'axis;
                }
                if let Some(tracker) = AxisTracking::for_outlet_and_axis(model, outlet, axis)? {
                    axes.push(tracker);
                }
            }
        }
    }
    Ok(axes)
}

pub fn for_model(model: &TypedModel) -> TractResult<AxesMapping> {
    let input_ranks = model
        .input_outlets()?
        .iter()
        .map(|io| model.outlet_fact(*io).map(|f| f.rank()))
        .collect::<TractResult<TVec<usize>>>()?;
    let output_ranks = model
        .output_outlets()?
        .iter()
        .map(|io| model.outlet_fact(*io).map(|f| f.rank()))
        .collect::<TractResult<TVec<usize>>>()?;
    let mut result = AxesMapping::disconnected_for_ranks(&input_ranks, &output_ranks)?;
    for tracking in full_axis_tracking(model)? {
        let mut reprs: Vec<char> = vec![];
        for (ix, outlet) in model.input_outlets()?.iter().enumerate() {
            if let Some(appearance) = tracking.outlets.get(outlet) {
                reprs.push(result.axis((InOut::In(ix), *appearance)).unwrap().repr);
            }
        }
        for (ix, outlet) in model.output_outlets()?.iter().enumerate() {
            if let Some(appearance) = tracking.outlets.get(outlet) {
                reprs.push(result.axis((InOut::Out(ix), *appearance)).unwrap().repr);
            }
        }
        if reprs.len() > 1 {
            for other in &reprs[1..] {
                result = result.linking(reprs[0], *other)?;
            }
        }
    }
    result.relabel()
}
