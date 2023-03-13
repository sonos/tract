use std::fmt::Display;
use std::iter::FromIterator;
use std::str::FromStr;

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
        let axes: TVec<_> = it.as_ref().into();
        let input_count = axes[0].inputs.len();
        let output_count = axes[0].outputs.len();
        AxesMapping { axes, output_count, input_count }.check()
    }

    pub fn new_no_inputs(it: impl AsRef<[Axis]>) -> TractResult<AxesMapping> {
        let axes: TVec<_> = it.as_ref().into();
        let output_count = axes[0].outputs.len();
        AxesMapping { axes, input_count: 0, output_count }.check()
    }

    pub fn disconnected(inputs: &[&TypedFact], outputs: &[&TypedFact]) -> TractResult<AxesMapping> {
        let input_ranks: TVec<usize> = inputs.iter().map(|i| i.rank()).collect();
        let output_ranks: TVec<usize> = outputs.iter().map(|i| i.rank()).collect();
        Self::disconnected_for_ranks(&input_ranks, &output_ranks)
    }

    pub fn disconnected_for_ranks(inputs: &[usize], outputs: &[usize]) -> TractResult<AxesMapping> {
        let mut axes = tvec!();
        let mut alphabet = ('a'..).into_iter();
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
            .into_iter()
            .zip('a'..)
            .map(|(axis_id, repr)| Axis::natural(inputs, outputs, repr, axis_id))
            .collect::<TVec<_>>();
        AxesMapping { axes, output_count: outputs.len(), input_count: inputs.len() }.check()
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
        self.check()
    }

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
            self.axes.push(Axis{ repr, inputs, outputs });
        }
        self.check()
    }

    fn do_check(&self) -> TractResult<()> {
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
        Ok(())
    }

    pub fn check(self) -> TractResult<AxesMapping> {
        self.do_check().with_context(|| format!("Checking {:?}", self.axes))?;
        Ok(self)
    }

    pub fn available_label(&self) -> char {
        ('a'..).find(|c| self.iter_all_axes().all(|axis| axis.repr != *c)).unwrap()
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

    #[test]
    fn test_parse_transpose() {
        assert_eq!(
            "ij->ji".parse::<AxesMapping>().unwrap(),
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
            "ii->i".parse::<AxesMapping>().unwrap(),
            AxesMapping::new(tvec![Axis::new('i', 1, 1).output(0, 0).input(0, 0).input(0, 1)])
                .unwrap(),
        )
    }

    #[test]
    fn test_parse_adamar_product_explicit() {
        assert_eq!(
            "i,i->i".parse::<AxesMapping>().unwrap(),
            AxesMapping::new(tvec![Axis::new('i', 2, 1).output(0, 0).input(0, 0).input(1, 0)])
                .unwrap(),
        )
    }

    #[test]
    fn test_parse_inner_product_implicit() {
        assert_eq!("i,i".parse::<AxesMapping>().unwrap(), "i,i->".parse::<AxesMapping>().unwrap(),)
    }

    #[test]
    fn test_parse_batch_matmul() {
        assert_eq!(
            "bij , bjk -> bik ".parse::<AxesMapping>().unwrap(),
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
            "i,j->ij".parse::<AxesMapping>().unwrap(),
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
            "ik,jkl,il->ij".parse::<AxesMapping>().unwrap(),
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
            "pqrs,tuqvr->pstuv".parse::<AxesMapping>().unwrap(),
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
        assert_eq!(
            "pqrs,tuqvr".parse::<AxesMapping>().unwrap(),
            "pqrs,tuqvr->pstuv".parse::<AxesMapping>().unwrap(),
        )
    }

    #[test]
    fn test_display_expr() {
        assert_eq!(
            "pqrs,tuqvr->pstuv".parse::<AxesMapping>().unwrap().to_string(),
            "pqrs,tuqvr->pstuv"
        );
    }

    #[test]
    fn test_parse_pulsed_matmul() {
        assert_eq!(
            "sij,ijk->sik".parse::<AxesMapping>().unwrap(),
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
            "bsij,ijk->bsik".parse::<AxesMapping>().unwrap(),
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
}
