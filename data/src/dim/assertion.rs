use fmt::Display;

use super::*;

#[derive(Debug, PartialEq, Clone, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum Assertion {
    LT(TDim, TDim),
    GT(TDim, TDim),
    LTE(TDim, TDim),
    GTE(TDim, TDim),
}

impl Display for Assertion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Assertion::*;
        match self {
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
}
