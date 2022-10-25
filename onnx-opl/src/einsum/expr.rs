use std::fmt::Display;
use std::iter::FromIterator;
use std::str::FromStr;

use tract_nnef::internal::*;
use tract_nnef::prelude::tract_itertools::Itertools;

#[derive(Debug, Clone, PartialEq, Eq, Default, Hash)]
pub struct AxisSym {
    pub result: Option<usize>,
    pub inputs: TVec<TVec<usize>>,
    pub repr: char,
}

impl AxisSym {
    fn new(repr: char) -> AxisSym {
        AxisSym { repr, result: None, inputs: tvec!() }
    }

    fn result(self, axis: usize) -> AxisSym {
        AxisSym { result: Some(axis), ..self }
    }

    fn set_result(&mut self, axis: usize) {
        self.result = Some(axis)
    }

    #[allow(dead_code)]
    fn input(mut self, input_id: usize, axis: usize) -> AxisSym {
        self.add_input(input_id, axis);
        self
    }

    fn ensure_inputs_count(&mut self, inputs: usize) {
        if self.inputs.len() < inputs {
            self.inputs.resize(inputs, tvec!())
        }
    }

    fn add_input(&mut self, input_id: usize, axis: usize) {
        self.ensure_inputs_count(input_id + 1);
        self.inputs[input_id].push(axis);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Expr {
    pub index: TVec<AxisSym>,
    pub sum: TVec<AxisSym>,
}

impl Expr {
    pub fn new(index: TVec<AxisSym>, sum: TVec<AxisSym>) -> Expr {
        let mut e = Expr { index, sum };
        e.canonicalize();
        e
    }

    pub fn index(&self) -> &[AxisSym] {
        &self.index
    }

    pub fn sum(&self) -> &[AxisSym] {
        &self.sum
    }

    pub fn iter_all_axes(&self) -> impl Iterator<Item = &AxisSym> {
        self.index.iter().chain(self.sum.iter())
    }

    pub fn n_inputs(&self) -> usize {
        self.iter_all_axes().map(|axis| axis.inputs.len()).max().unwrap()
    }

    pub fn axis_by_repr(&self, c: char) -> Option<&AxisSym> {
        self.iter_all_axes().find(|axis| axis.repr == c)
    }

    pub fn axis_positions_in_input(&self, input: usize, c: char) -> Option<&[usize]> {
        self.axis_by_repr(c).map(|axis| &*axis.inputs[input])
    }

    pub fn input_axis(&self, input: usize, position: usize) -> Option<&AxisSym> {
        self.iter_all_axes().find(|axis| axis.inputs[input].contains(&position))
    }

    pub fn input_rank(&self, input: usize) -> usize {
        self.iter_all_axes().map(|axis| axis.inputs[input].len()).sum()
    }

    pub fn input_axes(&self, input: usize) -> impl Iterator<Item = &AxisSym> {
        (0..self.input_rank(input)).map(move |ix| self.input_axis(input, ix).unwrap())
    }

    pub fn output_rank(&self) -> usize {
        self.index.len()
    }

    pub fn output_axes(&self) -> impl Iterator<Item = &AxisSym> {
        self.index.iter()
    }

    pub fn insert_input_axis(&mut self, axis: char, input: usize, position: usize) {
        self.index.iter_mut().for_each(|ax| {
            ax.inputs[input].iter_mut().for_each(|pos| *pos += (*pos >= position) as usize)
        });
        self.sum.iter_mut().for_each(|ax| {
            ax.inputs[input].iter_mut().for_each(|pos| *pos += (*pos >= position) as usize)
        });
        self.index.iter_mut().chain(self.sum.iter_mut()).find(|x| x.repr == axis).unwrap().inputs[input].push(position)
    }

    pub fn canonicalize(&mut self) {
        let n_inputs = self.n_inputs();
        for axis in &mut self.index {
            axis.ensure_inputs_count(n_inputs);
        }
        for axis in &mut self.sum {
            axis.ensure_inputs_count(n_inputs);
        }
    }
}

impl FromIterator<AxisSym> for Expr {
    fn from_iter<T: IntoIterator<Item = AxisSym>>(iter: T) -> Self {
        let (index, sum) = iter.into_iter().partition(|ax| ax.result.is_some());
        Expr::new(index, sum)
    }
}

impl<I: IntoIterator<Item = AxisSym>> From<I> for Expr {
    fn from(it: I) -> Self {
        it.into_iter().collect()
    }
}

impl FromStr for Expr {
    type Err = TractError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        assert!(!s.contains("..."));
        let s = s.replace(' ', "");
        let (inputs, result) =
            if let Some((i, r)) = s.split_once("->") { (i, Some(r)) } else { (&*s, None) };
        let inputs: TVec<&str> = inputs.split(',').collect();
        let mut axes = HashMap::<char, AxisSym>::default();
        if let Some(result) = result {
            for (ix, axis) in result.chars().enumerate() {
                axes.insert(axis, AxisSym::new(axis).result(ix));
            }
        }
        for (input_ix, input) in inputs.iter().enumerate() {
            for (ix, axis) in input.chars().enumerate() {
                axes.entry(axis).or_insert_with(|| AxisSym::new(axis)).add_input(input_ix, ix);
            }
        }
        if result.is_none() {
            axes.iter_mut()
                .sorted_by_key(|(k, _)| *k)
                .filter(|(_, v)| v.inputs.iter().map(|input| input.len()).sum::<usize>() == 1)
                .enumerate()
                .for_each(|(ix, (_, v))| v.set_result(ix))
        }
        Ok(axes.into_iter().sorted_by_key(|(k, _)| *k).map(|(_, v)| v).collect::<Expr>())
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for input in 0..self.n_inputs() {
            if input > 0 {
                write!(f, ",")?;
            }
            for axis in self
                .iter_all_axes()
                .flat_map(|axis| {
                    axis.inputs[input].iter().map(move |position| (position, axis.repr))
                })
                .sorted()
                .map(|(_, r)| r)
            {
                write!(f, "{}", axis)?;
            }
        }
        write!(f, "->")?;
        for axis in self
            .index
            .iter()
            .flat_map(|axis| axis.result.iter().map(move |position| (position, axis.repr)))
            .sorted()
            .map(|(_, r)| r)
        {
            write!(f, "{}", axis)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_expr_builder() {
        assert_eq!(
            Expr::from(tvec![
                AxisSym::new('a').result(0).input(0, 1),
                AxisSym::new('b').result(1).input(0, 0)
            ]),
            Expr {
                index: tvec!(
                    AxisSym::new('a').result(0).input(0, 1),
                    AxisSym::new('b').result(1).input(0, 0)
                ),
                sum: tvec!(),
            }
        )
    }

    #[test]
    fn test_parse_transpose() {
        assert_eq!(
            "ij->ji".parse::<Expr>().unwrap(),
            Expr::from(tvec![
                AxisSym::new('i').result(1).input(0, 0),
                AxisSym::new('j').result(0).input(0, 1)
            ]),
        )
    }

    #[test]
    fn test_parse_diag() {
        assert_eq!(
            "ii->i".parse::<Expr>().unwrap(),
            Expr::from(tvec![AxisSym::new('i').result(0).input(0, 0).input(0, 1)]),
        )
    }

    #[test]
    fn test_parse_adamar_product_explicit() {
        assert_eq!(
            "i,i->i".parse::<Expr>().unwrap(),
            Expr::from(tvec![AxisSym::new('i').result(0).input(0, 0).input(1, 0)]),
        )
    }

    #[test]
    fn test_parse_inner_product_implicit() {
        assert_eq!("i,i".parse::<Expr>().unwrap(), "i,i->".parse::<Expr>().unwrap(),)
    }

    #[test]
    fn test_parse_batch_matmul() {
        assert_eq!(
            "bij , bjk -> bik ".parse::<Expr>().unwrap(),
            Expr::from(tvec![
                AxisSym::new('b').result(0).input(0, 0).input(1, 0),
                AxisSym::new('i').result(1).input(0, 1),
                AxisSym::new('j').input(0, 2).input(1, 1),
                AxisSym::new('k').result(2).input(1, 2)
            ])
        )
    }

    #[test]
    fn test_parse_outer_product() {
        assert_eq!(
            "i,j->ij".parse::<Expr>().unwrap(),
            Expr::from(tvec![
                AxisSym::new('i').result(0).input(0, 0),
                AxisSym::new('j').result(1).input(1, 0)
            ]),
        )
    }

    #[test]
    fn test_parse_bilinear() {
        assert_eq!(
            "ik,jkl,il->ij".parse::<Expr>().unwrap(),
            Expr::from(tvec![
                AxisSym::new('i').result(0).input(0, 0).input(2, 0),
                AxisSym::new('j').result(1).input(1, 0),
                AxisSym::new('k').input(0, 1).input(1, 1),
                AxisSym::new('l').input(1, 2).input(2, 1)
            ]),
        )
    }

    #[test]
    fn test_parse_complex_tensor_contraction() {
        assert_eq!(
            "pqrs,tuqvr->pstuv".parse::<Expr>().unwrap(),
            Expr::from(tvec![
                AxisSym::new('p').result(0).input(0, 0),
                AxisSym::new('q').input(0, 1).input(1, 2),
                AxisSym::new('r').input(0, 2).input(1, 4),
                AxisSym::new('s').result(1).input(0, 3),
                AxisSym::new('t').result(2).input(1, 0),
                AxisSym::new('u').result(3).input(1, 1),
                AxisSym::new('v').result(4).input(1, 3),
            ]),
        )
    }

    #[test]
    fn test_parse_complex_tensor_contraction_implicit() {
        assert_eq!(
            "pqrs,tuqvr".parse::<Expr>().unwrap(),
            "pqrs,tuqvr->pstuv".parse::<Expr>().unwrap(),
        )
    }

    #[test]
    fn test_display_expr() {
        assert_eq!("pqrs,tuqvr->pstuv".parse::<Expr>().unwrap().to_string(), "pqrs,tuqvr->pstuv");
    }

    #[test]
    fn test_parse_pulsed_matmul() {
        assert_eq!(
            "sij,ijk->sik".parse::<Expr>().unwrap(),
            Expr::from(tvec![
                AxisSym::new('i').result(1).input(0, 1).input(1, 0),
                AxisSym::new('k').result(2).input(1, 2),
                AxisSym::new('s').result(0).input(0, 0),
                AxisSym::new('j').input(0, 2).input(1, 1),
            ])
        )
    }

    #[test]
    fn test_parse_pulsed_batch_matmul() {
        assert_eq!(
            "bsij,ijk->bsik".parse::<Expr>().unwrap(),
            Expr::from(tvec![
                AxisSym::new('b').result(0).input(0, 0),
                AxisSym::new('i').result(2).input(0, 2).input(1, 0),
                AxisSym::new('k').result(3).input(1, 2),
                AxisSym::new('s').result(1).input(0, 1),
                AxisSym::new('j').input(0, 3).input(1, 1),
            ])
        )
    }
}
