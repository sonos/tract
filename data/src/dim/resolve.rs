use tract_num_traits::Zero;

use crate::internal::*;

pub fn solve_for(sym: &Symbol, left: &TDim, right: &TDim) -> Option<TDim> {
    if !left.symbols().contains(sym) && !right.symbols().contains(sym) {
        return None;
    }
    if right.symbols().contains(sym) {
        return solve_for(sym, &(left.clone() - right), &0.to_dim());
    }
    match left {
        TDim::Sym(s) => {
            if s == sym {
                Some(right.clone())
            } else {
                None
            }
        }
        TDim::Add(terms) => {
            let consts: TDim = terms.iter().filter(|t| !t.symbols().contains(sym)).sum();
            if consts.is_zero() {
                None
            } else {
                solve_for(sym, &(left.clone() - &consts), &(right.clone() - &consts))
            }
        }
        TDim::MulInt(z, a) => {
            let gcd = right.gcd();
            if gcd % z.unsigned_abs() == 0 {
                solve_for(sym, a, &(right.clone() / *z))
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{parse_tdim, SymbolScope};

    lazy_static::lazy_static!(
        static ref TABLE:SymbolScope = SymbolScope::default();
        static ref A:Symbol = TABLE.sym("a");
    );

    fn p(s: &str) -> TDim {
        parse_tdim(&TABLE, s).unwrap()
    }

    #[test]
    fn trivial() {
        assert_eq!(solve_for(&A, &p("a"), &p("3")), Some(3i32.to_dim()));
    }

    #[test]
    fn negative() {
        assert_eq!(solve_for(&A, &p("a + 3"), &p("0")), Some(-(3i32.to_dim())));
    }

    #[test]
    fn swap() {
        assert_eq!(solve_for(&A, &p("3"), &p("a")), Some(3i32.to_dim()));
    }

    #[test]
    fn scale() {
        assert_eq!(solve_for(&A, &p("3 * a"), &p("6")), Some(2.to_dim()));
    }

    #[test]
    fn ax_plus_b() {
        assert_eq!(solve_for(&A, &p("3 * a + 1"), &p("7")), Some(2.to_dim()));
    }

    #[test]
    fn both_sides() {
        assert_eq!(solve_for(&A, &p("3 * a + 1"), &p("2 * a")), Some((-1).to_dim()));
    }

    #[test]
    fn x_over_n() {
        assert_eq!(solve_for(&A, &p("a/4"), &p("2")), None);
    }

    #[test]
    fn with_symbols() {
        assert_eq!(solve_for(&A, &p("a + 1"), &p("b")), Some(p("b-1")));
    }
}
