use std::fmt::Display;
use std::iter::FromIterator;
use std::str::FromStr;

use tract_ndarray::{ArrayViewD, ArrayViewMutD};

use crate::internal::*;
use crate::prelude::tract_itertools::Itertools;

use super::Axis;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AxesMapping {
    input_count: usize,
    output_count: usize,
    axes: TVec<Axis>,
}

impl AxesMapping {
    pub fn new(it: impl AsRef<[Axis]>) -> TractResult<AxesMapping> {
        let mut axes: TVec<_> = it.as_ref().into();
        axes.sort_by_key(|ax| ax.repr);
        let input_count = axes[0].inputs.len();
        let output_count = axes[0].outputs.len();
        AxesMapping { axes, output_count, input_count }.check()
    }

    pub fn new_no_inputs(it: impl AsRef<[Axis]>) -> TractResult<AxesMapping> {
        let axes: TVec<_> = it.as_ref().into();
        let output_count = axes[0].outputs.len();
        AxesMapping { axes, input_count: 0, output_count }.check()
    }

    pub fn for_numpy_matmul(
        rank: usize,
        transposing_a: bool,
        transposing_b: bool,
        transposing_c: bool,
    ) -> TractResult<AxesMapping> {
        let mut axes: TVec<Axis> = ('a'..)
            .take(rank - 2)
            .enumerate()
            .map(|(ix, repr)| Axis {
                repr,
                inputs: tvec!(tvec!(ix), tvec!(ix)),
                outputs: tvec!(tvec!(ix)),
            })
            .collect();
        axes.push(Axis {
            repr: 'm',
            inputs: tvec!(tvec!(rank - 2 + transposing_a as usize), tvec!()),
            outputs: tvec!(tvec!(rank - 2 + transposing_c as usize)),
        });
        axes.push(Axis {
            repr: 'k',
            inputs: tvec!(
                tvec!(rank - 1 - transposing_a as usize),
                tvec!(rank - 2 + transposing_b as usize)
            ),
            outputs: tvec!(tvec!()),
        });
        axes.push(Axis {
            repr: 'n',
            inputs: tvec!(tvec!(), tvec!(rank - 1 - transposing_b as usize),),
            outputs: tvec!(tvec!(rank - 1 - transposing_c as usize)),
        });
        AxesMapping::new(axes)
    }

    pub fn disconnected(inputs: &[&TypedFact], outputs: &[&TypedFact]) -> TractResult<AxesMapping> {
        let input_ranks: TVec<usize> = inputs.iter().map(|i| i.rank()).collect();
        let output_ranks: TVec<usize> = outputs.iter().map(|i| i.rank()).collect();
        Self::disconnected_for_ranks(&input_ranks, &output_ranks)
    }

    pub fn disconnected_for_ranks(inputs: &[usize], outputs: &[usize]) -> TractResult<AxesMapping> {
        let mut axes = tvec!();
        let mut alphabet = 'a'..;
        for (ix, &rank) in inputs.iter().enumerate() {
            for a in 0..rank {
                axes.push(
                    Axis::new(alphabet.next().unwrap(), inputs.len(), outputs.len()).input(ix, a),
                );
            }
        }
        for (ix, &rank) in outputs.iter().enumerate() {
            for a in 0..rank {
                axes.push(
                    Axis::new(alphabet.next().unwrap(), inputs.len(), outputs.len()).output(ix, a),
                );
            }
        }
        AxesMapping { axes, input_count: inputs.len(), output_count: outputs.len() }.check()
    }

    pub fn natural(inputs: &[&TypedFact], outputs: &[&TypedFact]) -> TractResult<AxesMapping> {
        let rank = inputs[0].rank();
        let axes = (0..rank)
            .zip('a'..)
            .map(|(axis_id, repr)| Axis::natural(inputs.len(), outputs.len(), repr, axis_id))
            .collect::<TVec<_>>();
        AxesMapping { axes, output_count: outputs.len(), input_count: inputs.len() }.check()
    }

    pub fn natural_for_rank(
        inputs: usize,
        outputs: usize,
        rank: usize,
    ) -> TractResult<AxesMapping> {
        let axes = (0..rank)
            .zip('a'..)
            .map(|(axis_id, repr)| Axis::natural(inputs, outputs, repr, axis_id))
            .collect::<TVec<_>>();
        AxesMapping { axes, output_count: outputs, input_count: inputs }.check()
    }

    pub fn iter_all_axes(&self) -> impl Iterator<Item = &Axis> {
        self.axes.iter()
    }

    pub fn input_count(&self) -> usize {
        self.input_count
    }

    pub fn output_count(&self) -> usize {
        self.output_count
    }

    pub fn axis_by_repr(&self, c: char) -> Option<&Axis> {
        self.iter_all_axes().find(|axis| axis.repr == c)
    }

    pub fn axis_positions_in_input(&self, input: usize, c: char) -> Option<&[usize]> {
        self.axis_by_repr(c).map(|axis| &*axis.inputs[input])
    }

    pub fn input_axis(&self, input: usize, position: usize) -> TractResult<&Axis> {
        self.iter_all_axes().find(|axis| axis.inputs[input].contains(&position)).with_context(
            || format!("Failed to find axis {position} in input {input} for \"{self}\""),
        )
    }

    fn input_axis_mut(&mut self, input: usize, position: usize) -> TractResult<&mut Axis> {
        let repr = self.input_axis(input, position)?.repr;
        Ok(self.axes.iter_mut().find(|axis| axis.repr == repr).unwrap())
    }

    pub fn interface_rank(&self, io: InOut) -> usize {
        match io {
            InOut::In(i) => self.input_rank(i),
            InOut::Out(o) => self.output_rank(o),
        }
    }

    pub fn input_rank(&self, input: usize) -> usize {
        self.iter_all_axes().map(|axis| axis.inputs[input].len()).sum()
    }

    pub fn input_axes(&self, input: usize) -> impl Iterator<Item = &Axis> {
        (0..self.input_rank(input)).map(move |ix| self.input_axis(input, ix).unwrap())
    }

    pub fn output_rank(&self, output: usize) -> usize {
        self.iter_all_axes().map(|axis| axis.outputs[output].len()).sum()
    }

    pub fn output_axis(&self, output: usize, position: usize) -> TractResult<&Axis> {
        self.iter_all_axes().find(|axis| axis.outputs[output].contains(&position)).with_context(
            || format!("Failed to find axis {position} in output {output} for \"{self}\""),
        )
    }

    pub fn output_axes(&self, output: usize) -> impl Iterator<Item = &Axis> {
        (0..self.output_rank(output)).map(move |ix| self.output_axis(output, ix).unwrap())
    }

    fn output_axis_mut(&mut self, output: usize, position: usize) -> TractResult<&mut Axis> {
        let repr = self.output_axis(output, position)?.repr;
        Ok(self.axes.iter_mut().find(|axis| axis.repr == repr).unwrap())
    }

    pub fn with_input_axis_named(
        mut self,
        input_id: usize,
        axis_pos: usize,
        name: char,
    ) -> TractResult<AxesMapping> {
        let old_label = self.input_axis(input_id, axis_pos)?.repr;
        if let Some(conflict) = self.axes.iter_mut().find(|axis| axis.repr == name) {
            conflict.repr = old_label
        }
        self.input_axis_mut(input_id, axis_pos)?.repr = name;
        self.sort();
        self.check()
    }

    pub fn with_output_axis_named(
        mut self,
        output_id: usize,
        axis_pos: usize,
        name: char,
    ) -> TractResult<AxesMapping> {
        let old_label = self.output_axis(output_id, axis_pos)?.repr;
        if let Some(conflict) = self.axes.iter_mut().find(|axis| axis.repr == name) {
            conflict.repr = old_label
        }
        self.output_axis_mut(output_id, axis_pos)?.repr = name;
        self.sort();
        self.check()
    }

    pub fn linking(mut self, a: char, b: char) -> TractResult<AxesMapping> {
        let b = self
            .axes
            .iter()
            .position(|axis| axis.repr == b)
            .with_context(|| format!("No axis called {b} in {self}"))?;
        let b = self.axes.remove(b);
        let a = self
            .axes
            .iter()
            .position(|axis| axis.repr == a)
            .with_context(|| format!("No axis called {a} in {self}"))?;
        let a = &mut self.axes[a];
        for (ia, ib) in a.inputs.iter_mut().zip(b.inputs.iter()) {
            ia.extend(ib.into_iter().cloned())
        }
        for (ia, ib) in a.outputs.iter_mut().zip(b.outputs.iter()) {
            ia.extend(ib.into_iter().cloned())
        }
        self.sort();
        self.check()
    }

    pub fn with_input_axis_linked_to(
        self,
        slot: usize,
        axis: usize,
        to: char,
    ) -> TractResult<AxesMapping> {
        let from = self.input_axis(slot, axis)?.repr;
        self.linking(to, from)
    }

    pub fn with_output_axis_linked_to(
        self,
        slot: usize,
        axis: usize,
        to: char,
    ) -> TractResult<AxesMapping> {
        let from = self.output_axis(slot, axis)?.repr;
        self.linking(to, from)
    }

    // FIXME: fuse with with_extra_input ?
    pub fn add_input(mut self, rank: usize) -> TractResult<AxesMapping> {
        self.input_count += 1;
        for axis in &mut self.axes {
            axis.inputs.push(tvec!());
        }
        for ix in 0..rank {
            let repr = self.available_label();
            let mut inputs = tvec!(tvec!(); self.input_count);
            inputs[self.input_count - 1].push(ix);
            let outputs = tvec!(tvec!(); self.output_count);
            self.axes.push(Axis { repr, inputs, outputs });
        }
        self.sort();
        self.check()
    }

    fn sort(&mut self) {
        self.axes.sort_by_key(|axis| axis.repr);
    }

    fn do_check(&self) -> TractResult<()> {
        for axis in &self.axes {
            ensure!(axis.inputs.len() == self.input_count);
            ensure!(axis.outputs.len() == self.output_count);
        }
        for input_ix in 0..self.input_count() {
            for axis in 0..self.input_rank(input_ix) {
                ensure!(self.input_axis(input_ix, axis).is_ok());
            }
        }
        for output_ix in 0..self.output_count() {
            for axis in 0..self.output_rank(output_ix) {
                ensure!(self.output_axis(output_ix, axis).is_ok());
            }
        }
        ensure!(self.axes.iter().map(|ax| ax.repr).duplicates().count() == 0);
        for (a, b) in self.axes.iter().tuple_windows() {
            ensure!(a.repr < b.repr, "{self}");
        }
        Ok(())
    }

    pub fn check(self) -> TractResult<AxesMapping> {
        self.do_check().with_context(|| format!("Checking {:?}", self.axes))?;
        Ok(self)
    }

    pub fn available_label(&self) -> char {
        ('a'..).find(|c| self.iter_all_axes().all(|axis| axis.repr != *c)).unwrap()
    }

    pub fn is_element_wise_unary(&self) -> bool {
        self.input_count == 1
            && self.output_count == 1
            && self
                .iter_all_axes()
                .all(|axis| axis.inputs[0].len() == 1 && axis.outputs[0] == axis.inputs[0])
    }

    pub fn extract_sub_mapping(
        &self,
        inputs: &[usize],
        outputs: &[usize],
    ) -> TractResult<AxesMapping> {
        let axes: Vec<_> = self
            .iter_all_axes()
            .filter(|axis| {
                inputs.iter().any(|i| axis.inputs[*i].len() > 0)
                    || outputs.iter().any(|o| axis.outputs[*o].len() > 0)
            })
            .map(|axis| Axis {
                inputs: axis
                    .inputs
                    .iter()
                    .enumerate()
                    .filter(|(ix, _)| inputs.contains(ix))
                    .map(|(_, it)| it.clone())
                    .collect(),
                outputs: axis
                    .outputs
                    .iter()
                    .enumerate()
                    .filter(|(ix, _)| outputs.contains(ix))
                    .map(|(_, it)| it.clone())
                    .collect(),
                repr: axis.repr,
            })
            .collect();
        AxesMapping::new(axes)
    }

    pub fn relabel(mut self) -> TractResult<AxesMapping> {
        for (ax, repr) in self.axes.iter_mut().zip('a'..) {
            ax.repr = repr;
        }
        Ok(self)
    }

    pub fn remove_axis(&self, repr: char) -> TractResult<AxesMapping> {
        let mut axes: TVec<Axis> =
            self.axes.iter().filter(|axis| axis.repr != repr).cloned().collect();
        let removed = self.axis_by_repr(repr).context("Axis not found")?;
        for input in 0..self.input_count {
            for &position in &removed.inputs[input] {
                for other in &mut axes {
                    other.inputs[input]
                        .iter_mut()
                        .for_each(|other_pos| *other_pos -= (*other_pos > position) as usize);
                }
            }
        }
        for output in 0..self.output_count {
            for &position in &removed.outputs[output] {
                for other in &mut axes {
                    other.outputs[output]
                        .iter_mut()
                        .for_each(|other_pos| *other_pos -= (*other_pos > position) as usize);
                }
            }
        }
        AxesMapping { axes, ..self.clone() }.check()
    }

    pub fn remove_input_axis(&self, slot: usize, position: usize) -> TractResult<AxesMapping> {
        let mut axes = self.axes.clone();
        for axis in &mut axes {
            axis.inputs[slot].retain(|pos| *pos != position);
            axis.inputs[slot].iter_mut().for_each(|pos| *pos -= (*pos > position) as usize);
        }
        AxesMapping { axes, ..self.clone() }.check()
    }

    pub fn with_extra_input(&self, slot: usize) -> TractResult<AxesMapping> {
        let mut axes: TVec<Axis> = self
            .iter_all_axes()
            .map(|axis| {
                let mut axis = axis.clone();
                axis.inputs.insert(slot, tvec!());
                axis
            })
            .collect();
        axes.sort_by_key(|ax| ax.repr);
        AxesMapping { axes, input_count: self.input_count + 1, ..self.clone() }.check()
    }

    pub fn with_extra_input_axis(
        &self,
        repr: char,
        slot: usize,
        position: usize,
    ) -> TractResult<AxesMapping> {
        let mut axes: TVec<Axis> = self.axes.clone();
        axes.iter_mut().for_each(|axis| {
            axis.inputs[slot].iter_mut().for_each(|pos| *pos += (*pos >= position) as usize)
        });
        let mut axis = Axis::new(repr, self.input_count, self.output_count);
        axis.inputs[slot].push(position);
        axes.push(axis);
        axes.sort_by_key(|ax| ax.repr);
        AxesMapping { axes, ..self.clone() }.check()
    }

    pub fn with_extra_output_axis(
        &self,
        repr: char,
        slot: usize,
        position: usize,
    ) -> TractResult<AxesMapping> {
        let mut axes: TVec<Axis> = self.axes.clone();
        axes.iter_mut().for_each(|axis| {
            axis.outputs[slot].iter_mut().for_each(|pos| *pos += (*pos >= position) as usize)
        });
        let mut axis = Axis::new(repr, self.input_count, self.output_count);
        axis.outputs[slot].push(position);
        axes.push(axis);
        axes.sort_by_key(|ax| ax.repr);
        AxesMapping { axes, ..self.clone() }.check()
    }

    pub fn translate_to_axis_ops(&self) -> TractResult<Vec<AxisOp>> {
        ensure!(self.input_count() == 1);
        ensure!(self.output_count() == 1);
        ensure!(self.iter_all_axes().all(|axis| axis.inputs[0].len() <= 1));
        let rms = self
            .iter_all_axes()
            .filter(|a| a.outputs[0].len() == 0)
            .sorted_by_key(|axis| -(axis.inputs[0][0] as isize))
            .collect_vec();
        let adds = self
            .iter_all_axes()
            .filter(|a| a.inputs[0].len() == 0)
            .sorted_by_key(|axis| axis.outputs[0][0] as isize)
            .collect_vec();
        let permutation = rms
            .iter()
            .chain(adds.iter())
            .try_fold(self.clone(), |mapping, axis| mapping.remove_axis(axis.repr))?;
        let permutation = permutation
            .iter_all_axes()
            .sorted_by_key(|axis| axis.inputs[0][0])
            .map(|a| a.outputs[0][0])
            .collect_vec();
        let permutation = perm_to_ops(&permutation);
        let rms = rms.iter().map(|axis| AxisOp::Rm(axis.inputs[0][0]));
        let adds = adds.iter().map(|axis| AxisOp::Add(axis.outputs[0][0]));
        Ok(rms.chain(permutation).chain(adds).collect())
    }

    pub fn from_strs(
        inputs: &[impl AsRef<str>],
        outputs: &[impl AsRef<str>],
    ) -> TractResult<AxesMapping> {
        let mut axes = HashMap::<char, Axis>::default();
        for (input_ix, input) in inputs.iter().enumerate() {
            for (ix, axis) in input.as_ref().chars().enumerate() {
                axes.entry(axis)
                    .or_insert_with(|| Axis::new(axis, inputs.len(), outputs.len().max(1)))
                    .add_input(input_ix, ix);
            }
        }
        for (output_ix, output) in outputs.iter().enumerate() {
            for (ix, axis) in output.as_ref().chars().enumerate() {
                axes.entry(axis)
                    .or_insert_with(|| Axis::new(axis, inputs.len(), outputs.len().max(1)))
                    .add_output(output_ix, ix);
            }
        }
        if outputs.len() == 0 {
            axes.iter_mut()
                .sorted_by_key(|(k, _)| *k)
                .filter(|(_, v)| v.inputs.iter().map(|input| input.len()).sum::<usize>() == 1)
                .enumerate()
                .for_each(|(ix, (_, v))| v.add_output(0, ix))
        }
        axes.into_iter().sorted_by_key(|(k, _)| *k).map(|(_, v)| v).collect()
    }

    pub fn to_strs(&self) -> (TVec<String>, TVec<String>) {
        let mut inputs = tvec![];
        let mut outputs = tvec![];
        for input in 0..self.input_count() {
            let s = self
                .iter_all_axes()
                .flat_map(|axis| {
                    axis.inputs[input].iter().map(move |position| (position, axis.repr))
                })
                .sorted()
                .map(|(_, r)| r)
                .collect();
            inputs.push(s);
        }
        for output in 0..self.output_count() {
            let s = self
                .iter_all_axes()
                .flat_map(|axis| {
                    axis.outputs[output].iter().map(move |position| (position, axis.repr))
                })
                .sorted()
                .map(|(_, r)| r)
                .collect();
            outputs.push(s);
        }
        (inputs, outputs)
    }

    pub fn change_axis_sink(&self, io: InOut, change: &AxisOp) -> TractResult<Option<AxesMapping>> {
        let (mut inputs, mut outputs) = self.to_strs();
        let interface: &mut String = match io {
            InOut::In(i) => &mut inputs[i],
            InOut::Out(o) => &mut outputs[o],
        };
        let mut axes: Vec<char> = interface.chars().collect();
        match change {
            AxisOp::Rm(rm) => {
                axes.remove(*rm);
            }
            AxisOp::Add(add) => axes.insert(*add, self.available_label()),
            AxisOp::Move(from, to) => {
                let c = axes.remove(*from);
                axes.insert(*to, c);
            }
            _ => return Ok(None),
        };
        *interface = axes.into_iter().collect();
        Ok(Some(AxesMapping::from_strs(&inputs, &outputs)?))
    }

    pub fn axis_ops_to_canonical(&self, io: InOut) -> TractResult<Vec<AxisOp>> {
        let rank = self.interface_rank(io);
        let target_rank = self.axes.len();
        let mut next_insert_axis = 0;
        let mut permutation = tvec!();
        for axis in &self.axes {
            let spec = match io {
                InOut::In(i) => axis.inputs[i].get(0),
                InOut::Out(o) => axis.outputs[o].get(0),
            };
            if let Some(pos_in_a) = spec {
                permutation.push(pos_in_a + target_rank - rank)
            } else {
                permutation.push(next_insert_axis);
                next_insert_axis += 1;
            }
        }
        let mut ops = vec![AxisOp::Add(0); target_rank - rank];
        ops.extend(crate::ops::change_axes::perm_to_ops(&permutation).into_iter());
        dbg!(&ops);
        Ok(ops)
    }

    pub fn view_to_canonical<D>(&self, io: InOut, view: &mut ArrayViewD<D>) -> TractResult<()> {
        dbg!(&view.shape());
        for op in self.axis_ops_to_canonical(io)? {
            op.change_view(view)?;
            dbg!(op, &view.shape());
        }
        Ok(())
    }

    pub fn view_to_canonical_mut<D>(
        &self,
        io: InOut,
        view: &mut ArrayViewMutD<D>,
    ) -> TractResult<()> {
        dbg!(&self.axes);
        for op in dbg!(self.axis_ops_to_canonical(io)?) {
            op.change_view_mut(view)?;
        }
        Ok(())
    }
}

impl FromIterator<Axis> for TractResult<AxesMapping> {
    fn from_iter<T: IntoIterator<Item = Axis>>(iter: T) -> TractResult<AxesMapping> {
        let axes = iter.into_iter().collect::<TVec<_>>();
        if axes.len() == 0 {
            bail!("Can not build axes mapping by collecting 0 axes");
        }
        AxesMapping::new(axes)
    }
}

impl FromStr for AxesMapping {
    type Err = TractError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        assert!(!s.contains("..."));
        let s = s.replace(' ', "");
        let (inputs, outputs) =
            if let Some((i, r)) = s.split_once("->") { (i, r) } else { (&*s, "") };
        let inputs: TVec<&str> = inputs.split(',').collect();
        let outputs: TVec<&str> = outputs.split(',').filter(|s| s.len() > 0).collect();
        AxesMapping::from_strs(&inputs, &outputs)
    }
}

impl Display for AxesMapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (inputs, outputs) = self.to_strs();
        write!(f, "{}->{}", inputs.iter().join(","), outputs.iter().join(","))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn m(s: &str) -> AxesMapping {
        s.parse().unwrap()
    }

    #[test]
    fn test_parse_transpose() {
        assert_eq!(
            m("ij->ji"),
            AxesMapping::new(tvec![
                Axis::new('i', 1, 1).output(0, 1).input(0, 0),
                Axis::new('j', 1, 1).output(0, 0).input(0, 1)
            ])
            .unwrap(),
        )
    }

    #[test]
    fn test_parse_diag() {
        assert_eq!(
            m("ii->i"),
            AxesMapping::new(tvec![Axis::new('i', 1, 1).output(0, 0).input(0, 0).input(0, 1)])
                .unwrap(),
        )
    }

    #[test]
    fn test_parse_adamar_product_explicit() {
        assert_eq!(
            m("i,i->i"),
            AxesMapping::new(tvec![Axis::new('i', 2, 1).output(0, 0).input(0, 0).input(1, 0)])
                .unwrap(),
        )
    }

    #[test]
    fn test_parse_inner_product_implicit() {
        assert_eq!(m("i,i"), m("i,i->"))
    }

    #[test]
    fn test_parse_batch_matmul() {
        assert_eq!(
            m("bij , bjk -> bik "),
            AxesMapping::new(tvec![
                Axis::new('b', 2, 1).output(0, 0).input(0, 0).input(1, 0),
                Axis::new('i', 2, 1).output(0, 1).input(0, 1),
                Axis::new('j', 2, 1).input(0, 2).input(1, 1),
                Axis::new('k', 2, 1).output(0, 2).input(1, 2)
            ])
            .unwrap()
        )
    }

    #[test]
    fn test_parse_outer_product() {
        assert_eq!(
            m("i,j->ij"),
            AxesMapping::new(tvec![
                Axis::new('i', 2, 1).output(0, 0).input(0, 0),
                Axis::new('j', 2, 1).output(0, 1).input(1, 0)
            ])
            .unwrap(),
        )
    }

    #[test]
    fn test_parse_bilinear() {
        assert_eq!(
            m("ik,jkl,il->ij"),
            AxesMapping::new(tvec![
                Axis::new('i', 3, 1).output(0, 0).input(0, 0).input(2, 0),
                Axis::new('j', 3, 1).output(0, 1).input(1, 0),
                Axis::new('k', 3, 1).input(0, 1).input(1, 1),
                Axis::new('l', 3, 1).input(1, 2).input(2, 1)
            ])
            .unwrap(),
        )
    }

    #[test]
    fn test_parse_complex_tensor_contraction() {
        assert_eq!(
            m("pqrs,tuqvr->pstuv"),
            AxesMapping::new(tvec![
                Axis::new('p', 2, 1).output(0, 0).input(0, 0),
                Axis::new('q', 2, 1).input(0, 1).input(1, 2),
                Axis::new('r', 2, 1).input(0, 2).input(1, 4),
                Axis::new('s', 2, 1).output(0, 1).input(0, 3),
                Axis::new('t', 2, 1).output(0, 2).input(1, 0),
                Axis::new('u', 2, 1).output(0, 3).input(1, 1),
                Axis::new('v', 2, 1).output(0, 4).input(1, 3),
            ])
            .unwrap(),
        )
    }

    #[test]
    fn test_parse_complex_tensor_contraction_implicit() {
        assert_eq!(m("pqrs,tuqvr"), m("pqrs,tuqvr->pstuv"))
    }

    #[test]
    fn test_display_expr() {
        assert_eq!(m("pqrs,tuqvr->pstuv").to_string(), "pqrs,tuqvr->pstuv");
    }

    #[test]
    fn test_parse_pulsed_matmul() {
        assert_eq!(
            m("sij,ijk->sik"),
            AxesMapping::new(tvec![
                Axis::new('i', 2, 1).output(0, 1).input(0, 1).input(1, 0),
                Axis::new('j', 2, 1).input(0, 2).input(1, 1),
                Axis::new('k', 2, 1).output(0, 2).input(1, 2),
                Axis::new('s', 2, 1).output(0, 0).input(0, 0),
            ])
            .unwrap()
        )
    }

    #[test]
    fn test_parse_pulsed_batch_matmul() {
        assert_eq!(
            m("bsij,ijk->bsik"),
            AxesMapping::new(tvec![
                Axis::new('b', 2, 1).output(0, 0).input(0, 0),
                Axis::new('i', 2, 1).output(0, 2).input(0, 2).input(1, 0),
                Axis::new('j', 2, 1).input(0, 3).input(1, 1),
                Axis::new('k', 2, 1).output(0, 3).input(1, 2),
                Axis::new('s', 2, 1).output(0, 1).input(0, 1),
            ])
            .unwrap()
        )
    }

    #[test]
    fn test_extract_sub_mapping() {
        assert_eq!(m("bsij,ijk->bsik").extract_sub_mapping(&[0], &[0]).unwrap(), m("bsij->bsik"));
        assert_eq!(m("bsij,ijk->bsik").extract_sub_mapping(&[1], &[0]).unwrap(), m("ijk->bsik"));
        assert_eq!(m("bsij,ijk->ij").extract_sub_mapping(&[1], &[0]).unwrap(), m("ijk->ij"));
    }

    #[test]
    fn test_remove_axis_0() {
        assert_eq!(m("ab->a").remove_axis('b').unwrap(), m("a->a"));
        assert_eq!(m("ba->a").remove_axis('b').unwrap(), m("a->a"));
        assert_eq!(m("a->ba").remove_axis('b').unwrap(), m("a->a"));
        assert_eq!(m("a->ab").remove_axis('b').unwrap(), m("a->a"));
        assert_eq!(m("ab,a->a").remove_axis('b').unwrap(), m("a,a->a"));
        assert_eq!(m("ba,a->a").remove_axis('b').unwrap(), m("a,a->a"));
        assert_eq!(m("a,ab->a").remove_axis('b').unwrap(), m("a,a->a"));
        assert_eq!(m("a,ba->a").remove_axis('b').unwrap(), m("a,a->a"));
        assert_eq!(m("a,a->ab").remove_axis('b').unwrap(), m("a,a->a"));
        assert_eq!(m("a,a->ba").remove_axis('b').unwrap(), m("a,a->a"));
        assert_eq!(m("bsij,ijk->bsik").remove_axis('i').unwrap(), m("bsj,jk->bsk"),);
    }

    #[test]
    fn test_translate_to_ops_rm_add() {
        assert_eq!(m("ab->a").translate_to_axis_ops().unwrap(), vec!(AxisOp::Rm(1)));
        assert_eq!(m("ba->a").translate_to_axis_ops().unwrap(), vec!(AxisOp::Rm(0)));
        assert_eq!(
            m("ab->c").translate_to_axis_ops().unwrap(),
            vec!(AxisOp::Rm(1), AxisOp::Rm(0), AxisOp::Add(0))
        );
    }

    #[test]
    fn test_translate_to_ops_move() {
        assert_eq!(m("ab->ba").translate_to_axis_ops().unwrap(), vec!(AxisOp::Move(1, 0)));
    }
}
