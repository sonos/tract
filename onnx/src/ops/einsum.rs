use std::iter::FromIterator;
use std::str::FromStr;

use crate::model::ParsingContext;
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops::unimpl;
use tract_hir::prelude::tract_itertools::Itertools;

// ij->ji => [{0, [[1]]}, {1, [[0]]}]
// ij->i => [{0, [[0]]}, {, [[1]]}]
// mk,bkn => bmn [{0, [,[0]]}, {, [[1],[1]]}, {1, [[0],]}, {2, [,[2]]}]

/*
struct A {
axes: Vec<AxisSym>
}
*/

// ij->ji => [[1],[0]]
// ij->i => [[0]]

// Batched DNN: A=W=1,m,k X=B=b,k,n => mk,bkn -> bmn
//
// abij,abji -> ab

/*
struct Expr {
index: TVec<TVec<usize>>,
sum: TVec<TVec<TVec<usize>>>, // [axis_id][input_id][..]
}
*/

#[derive(Debug, Clone, PartialEq, Default)]
struct AxisSym {
    result: Option<usize>,
    inputs: TVec<TVec<usize>>,
}

impl AxisSym {
    fn result(axis: usize) -> AxisSym {
        AxisSym { result: Some(axis), inputs: tvec!() }
    }

    fn no_result() -> AxisSym {
        AxisSym { result: None, inputs: tvec!() }
    }

    fn set_result(&mut self, axis: usize) {
        self.result = Some(axis)
    }

    fn input(mut self, input_id: usize, axis: usize) -> AxisSym {
        self.add_input(input_id, axis);
        self
    }

    fn add_input(&mut self, input_id: usize, axis: usize) {
        if self.inputs.len() <= input_id {
            self.inputs.resize(input_id + 1, tvec!())
        }
        self.inputs[input_id].push(axis);
    }
}

pub fn einsum(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let mut equation = node.get_attr::<String>("equation")?;
    dbg!(equation);
    unimplemented!();
}

#[derive(Debug, Clone, PartialEq)]
struct Expr {
    index: TVec<AxisSym>,
    sum: TVec<AxisSym>,
}

impl FromIterator<AxisSym> for Expr {
    fn from_iter<T: IntoIterator<Item = AxisSym>>(iter: T) -> Self {
        let (index, sum) = iter.into_iter().partition(|ax| ax.result.is_some());
        Expr { index, sum }
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
        let s = s.replace(" ", "");
        let (inputs, result) =
            if let Some((i, r)) = s.split_once("->") { (i, Some(r)) } else { (&*s, None) };
        let inputs: TVec<&str> = inputs.split(",").collect();
        let mut axes = HashMap::<char, AxisSym>::default();
        if let Some(result) = result {
            for (ix, axis) in result.chars().enumerate() {
                axes.insert(axis, AxisSym::result(ix));
            }
        }
        for (input_ix, input) in inputs.iter().enumerate() {
            for (ix, axis) in input.chars().enumerate() {
                axes.entry(axis).or_insert(AxisSym::no_result()).add_input(input_ix, ix);
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

struct EinSum {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_expr_builder() {
        assert_eq!(
            Expr::from(tvec![AxisSym::result(0).input(0, 1), AxisSym::result(1).input(0, 0)]),
            Expr {
                index: tvec!(AxisSym::result(0).input(0, 1), AxisSym::result(1).input(0, 0)),
                sum: tvec!(),
            }
        )
    }

    #[test]
    fn test_parse_transpose() {
        assert_eq!(
            "ij->ji".parse::<Expr>().unwrap(),
            Expr::from(tvec![AxisSym::result(1).input(0, 0), AxisSym::result(0).input(0, 1)]),
        )
    }

    #[test]
    fn test_parse_diag() {
        assert_eq!(
            "ii->i".parse::<Expr>().unwrap(),
            Expr::from(tvec![AxisSym::result(0).input(0, 0).input(0, 1)]),
        )
    }

    #[test]
    fn test_parse_adamar_product_explicit() {
        assert_eq!(
            "i,i->i".parse::<Expr>().unwrap(),
            Expr::from(tvec![AxisSym::result(0).input(0, 0).input(1, 0)]),
        )
    }

    #[test]
    fn test_parse_inner_product_implicit() {
        assert_eq!(
            "i,i".parse::<Expr>().unwrap(),
            "i,i->".parse::<Expr>().unwrap(),
        )
    }


    #[test]
    fn test_parse_batch_matmul() {
        assert_eq!(
            "bij , bjk -> bik ".parse::<Expr>().unwrap(),
            Expr::from(tvec![
                AxisSym::result(0).input(0, 0).input(1, 0),
                AxisSym::result(1).input(0, 1),
                AxisSym::no_result().input(0, 2).input(1, 1),
                AxisSym::result(2).input(1, 2)
            ])
        )
    }

    #[test]
    fn test_parse_outer_product() {
        assert_eq!(
            "i,j->ij".parse::<Expr>().unwrap(),
            Expr::from(tvec![AxisSym::result(0).input(0, 0), AxisSym::result(1).input(1, 0)]),
        )
    }

    #[test]
    fn test_parse_bilinear() {
        assert_eq!(
            "ik,jkl,il->ij".parse::<Expr>().unwrap(),
            Expr::from(tvec![
                AxisSym::result(0).input(0, 0).input(2, 0),
                AxisSym::result(1).input(1, 0),
                AxisSym::no_result().input(0, 1).input(1, 1),
                AxisSym::no_result().input(1, 2).input(2, 1)
            ]),
        )
    }

    #[test]
    fn test_parse_complex_tensor_contraction() {
        assert_eq!(
            "pqrs,tuqvr->pstuv".parse::<Expr>().unwrap(),
            Expr::from(tvec![
                AxisSym::result(0).input(0, 0),
                AxisSym::no_result().input(0, 1).input(1, 2),
                AxisSym::no_result().input(0, 2).input(1, 4),
                AxisSym::result(1).input(0, 3),
                AxisSym::result(2).input(1, 0),
                AxisSym::result(3).input(1, 1),
                AxisSym::result(4).input(1, 3),
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
}
