use egglog::ast::{Expr, GenericExpr, Literal};
use egglog::{EGraph, TermDag};

use prelude::{SymbolScope, TDim, ToDim};
use tract_data::*;

fn main() {}

const ENGINE: &str = include_str!("../tdim.egglog");

fn egraph() -> TractResult<EGraph> {
    let mut graph = EGraph::default();
    let output = graph
        .parse_and_run_program(Some("tdim.egglog".into()), ENGINE)
        .map_err(|e| anyhow::format_err!("{e}"))?;
    output.iter().for_each(|line| println!("{line}"));
    Ok(graph)
}

fn tdim_to_expr(tdim: &TDim) -> Expr {
    match tdim {
        TDim::Val(x) => Expr::call_no_span("Num", [Expr::lit_no_span(Literal::Int(*x))]),
        TDim::Sym(s) => Expr::call_no_span("Var", [Expr::lit_no_span(Literal::String(s.to_string().into()))]),
        TDim::Add(terms) => {
            terms.iter().map(tdim_to_expr).reduce(|a, b| Expr::call_no_span("Add", [a, b])).unwrap()
        }
        _ => todo!(),
    }
}

fn expr_to_tdim(scope: &SymbolScope, expr: &Expr) -> TDim {
    match expr {
        GenericExpr::Call(_, s, children) if s.as_str() == "Add" => {
            let left = expr_to_tdim(scope, &children[0]);
            let right = expr_to_tdim(scope, &children[1]);
            left + right
        }
        GenericExpr::Call(_, s, children) if s.as_str() == "Num" => {
            expr_to_tdim(scope, &children[0])
        }
        GenericExpr::Call(_, s, children) if s.as_str() == "Var" => {
            expr_to_tdim(scope, &children[0])
        }
        GenericExpr::Lit(_, Literal::Int(i)) => i.to_dim(),
        GenericExpr::Lit(_, Literal::String(i)) => scope.sym(i.as_str()).to_dim(),
        _ => todo!("{expr}")
    }
}

#[test]
fn test_0() {
    let mut egraph = egraph().unwrap();
    let scope = SymbolScope::default();
    let s = scope.parse_tdim("s").unwrap();
    let it = TDim::Add(vec![TDim::Val(3), s, TDim::Val(2)]);
    let expr = tdim_to_expr(&it);
    let (sort, value) = egraph.eval_expr(&expr).unwrap();
    let mut termdag = TermDag::default();
    let (_, extracted) = egraph.extract(value, &mut termdag, &sort);
    let simplified = expr_to_tdim(&scope, &termdag.term_to_expr(&extracted));
    println!("{simplified}");
}
