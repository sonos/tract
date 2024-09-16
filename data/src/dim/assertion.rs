use fmt::Display;

use super::*;

#[derive(Debug, PartialEq, Clone, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum Assertion {
    Eq(TDim, TDim),
    LT(TDim, TDim),
    GT(TDim, TDim),
    LTE(TDim, TDim),
    GTE(TDim, TDim),
}

impl Display for Assertion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Assertion::*;
        match self {
            Eq(l, r) => write!(f, "{l} == {r}"),
            LT(l, r) => write!(f, "{l} < {r}"),
            GT(l, r) => write!(f, "{l} > {r}"),
            LTE(l, r) => write!(f, "{l} <= {r}"),
            GTE(l, r) => write!(f, "{l} >= {r}"),
        }
    }
}

impl Assertion {
    pub fn as_known_positive(&self) -> Option<TDim> {
        use Assertion::*;
        match self {
            Eq(left, right) => Some(left.clone() - right),
            GTE(left, right) => Some(left.clone() - right),
            GT(left, right) => Some(left.clone() - 1 - right),
            LTE(left, right) => Some(right.clone() - left),
            LT(left, right) => Some(right.clone() - 1 - left),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn use_equalities() {
        let s = SymbolScope::default();
        s.add_assertion("s==0").unwrap();
        assert!(s.parse_tdim("s").unwrap().simplify().is_zero());
    }

    #[test]
    fn prove_positive_with_axiom() {
        let s = SymbolScope::default();
        s.add_assertion("s>=0").unwrap();
        assert!(s.parse_tdim("s").unwrap().prove_positive_or_zero());
    }

    #[test]
    fn prove_positive_with_axiom_2() {
        let s = SymbolScope::default();
        s.add_assertion("s>=0").unwrap();
        s.add_assertion("p>=0").unwrap();
        s.add_assertion("p+s<4096").unwrap();
        assert!(s.parse_tdim("4096-p").unwrap().prove_positive_or_zero());
    }

    #[test]
    fn min_max_with_axiom() {
        let symbols = SymbolScope::default();
        symbols.add_assertion("a>=0").unwrap();
        assert_eq!(symbols.parse_tdim("min(a,0)").unwrap().simplify(), 0.into());
        assert_eq!(
            symbols.parse_tdim("max(a,0)").unwrap().simplify(),
            symbols.parse_tdim("a").unwrap()
        );
    }

    #[test]
    fn low_bound_0() -> TractResult<()> {
        let symbols = SymbolScope::default().with_assertion("S>=0")?;
        let s = symbols.parse_tdim("S").unwrap();
        assert_eq!(s.low_inclusive_bound(), Some(0));
        Ok(())
    }

    #[test]
    fn low_bound_1() -> TractResult<()> {
        let symbols = SymbolScope::default().with_assertion("S>0")?;
        assert_eq!(symbols.parse_tdim("S").unwrap().low_inclusive_bound(), Some(1));
        Ok(())
    }

    #[test]
    fn low_bound_2() -> TractResult<()> {
        let symbols = SymbolScope::default().with_assertion("S>0")?;
        assert_eq!(symbols.parse_tdim("S + 1").unwrap().low_inclusive_bound(), Some(2));
        Ok(())
    }

    #[test]
    fn low_bound_3() -> TractResult<()> {
        let symbols = SymbolScope::default().with_assertion("S>0")?;
        assert_eq!(symbols.parse_tdim("4*S").unwrap().low_inclusive_bound(), Some(4));
        Ok(())
    }

    #[test]
    fn low_bound_4() -> TractResult<()> {
        let symbols = SymbolScope::default().with_assertion("S>0")?.with_assertion("S>5")?;
        assert_eq!(symbols.parse_tdim("S + 3").unwrap().low_inclusive_bound(), Some(9));
        Ok(())
    }

    #[test]
    fn max_bug_1() {
        let symbols = SymbolScope::default();
        symbols.add_assertion("S>8").unwrap();
        assert_eq!(
            symbols.parse_tdim("max(1,-1+(S+1)/4)").unwrap().simplify(),
            symbols.parse_tdim("-1+(S+1)/4").unwrap(),
        );
    }

    #[test]
    fn min_bug_1() {
        let symbols = SymbolScope::default();
        symbols.add_assertion("S>8").unwrap();
        assert_eq!(
            symbols.parse_tdim("min(1,-1+(S+1)/4)").unwrap().simplify(),
            symbols.parse_tdim("1").unwrap()
        );
    }

    #[test]
    fn min_bug_2() {
        let symbols = SymbolScope::default();
        symbols.add_assertion("S>50").unwrap();
        assert_eq!(
            symbols.parse_tdim("min(-3+2*(S+1)/4,-1+(S+1)/4)").unwrap().simplify(),
            symbols.parse_tdim("-1+(S+1)/4").unwrap()
        );
    }

    #[test]
    fn min_bug_3() {
        let symbols = SymbolScope::default();
        symbols.add_assertion("S>=0").unwrap();
        symbols.add_assertion("P>=0").unwrap();
        assert_eq!(
            symbols.parse_tdim("min(0,(S)#(P+S))").unwrap().simplify(),
            symbols.parse_tdim("0").unwrap()
        );
    }

    #[test]
    fn min_llm_0() -> TractResult<()> {
        let symbols = SymbolScope::default()
            .with_assertion("S>=0")?
            .with_assertion("P>=0")?
            .with_scenario_assertion("tg", "S==1")?
            .with_scenario_assertion("pp", "P==0")?;
        assert_eq!(
            symbols.parse_tdim("min(P,(S)#(P+S))").unwrap().simplify(),
            symbols.parse_tdim("P").unwrap()
        );
        Ok(())
    }
}
