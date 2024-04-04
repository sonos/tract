use super::*;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{alpha1, alphanumeric1, digit1, one_of};
use nom::combinator::{all_consuming, map, map_res, recognize};
use nom::multi::many0;
use nom::sequence::{delimited, pair, separated_pair};
use nom::IResult;

pub fn parse_tdim(symbol_table: &SymbolTable, input: &str) -> TractResult<TDim> {
    match all_consuming(|i| expr(symbol_table, i))(input) {
        Ok(pair) => Ok(pair.1),
        Err(e) => anyhow::bail!("Failed to parse {:?}, {:?}", input, e),
    }
}

fn expr<'i>(symbol_table: &SymbolTable, i: &'i str) -> IResult<&'i str, TDim> {
    add(symbol_table, i)
}

macro_rules! bin {
    ($name: ident, $next: ident, $op: expr, $builder: expr) => {
        fn $name<'i>(symbol_table: &SymbolTable, input: &'i str) -> IResult<&'i str, TDim> {
            let s = symbol_table;
            alt((map(separated_pair(|i| $next(s, i), stag($op), |i| $next(s, i)), $builder), |i| {
                $next(s, i)
            }))(input)
        }
    };
}

bin!(add, sub, "+", |(a, b)| a + b);
bin!(sub, mul, "-", |(a, b)| a - b);
bin!(mul, div, "*", |(a, b)| a * b);

fn div<'i>(symbol_table: &SymbolTable, input: &'i str) -> IResult<&'i str, TDim> {
    let s = symbol_table;
    alt((map(separated_pair(|i| atom(s, i), stag("/"), numeric), |(a, b)| a / b), |i| atom(s, i)))(
        input,
    )
}

fn atom<'i>(symbol_table: &SymbolTable, i: &'i str) -> IResult<&'i str, TDim> {
    alt((
        map(numeric, TDim::Val),
        map(|i| identifier(symbol_table, i), TDim::Sym),
        map(pair(recognize(stag("-")), |i| atom(symbol_table, i)), |(_, dim)| dim * -1),
        delimited(stag("("), |i| expr(symbol_table, i), stag(")")),
    ))(i)
}

fn identifier<'i>(symbol_table: &SymbolTable, i: &'i str) -> IResult<&'i str, Symbol> {
    map(recognize(pair(alt((alpha1, tag("_"))), many0(alt((alphanumeric1, tag("_")))))), |s| {
        symbol_table.sym(s)
    })(i)
}

fn numeric(i: &str) -> IResult<&str, i64> {
    map_res(digit1, std::str::FromStr::from_str)(i)
}

fn spaces(i: &str) -> IResult<&str, ()> {
    map(many0(one_of(" \t\n\r")), |_| ())(i)
}

fn spaced<'s, O, F>(it: F) -> impl FnMut(&'s str) -> IResult<&'s str, O>
where
    F: FnMut(&'s str) -> IResult<&'s str, O>,
{
    delimited(spaces, it, spaces)
}

pub(super) fn stag<'s>(t: &'static str) -> impl FnMut(&'s str) -> IResult<&'s str, &'s str> {
    spaced(tag(t))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse_int() {
        let table = SymbolTable::default();
        assert_eq!(parse_tdim(&table, "12").unwrap(), TDim::Val(12));
        assert_eq!(parse_tdim(&table, "-12").unwrap(), TDim::Val(-12));
    }

    #[test]
    fn parse_sym() {
        let table = SymbolTable::default();
        assert_eq!(parse_tdim(&table, "x").unwrap(), TDim::Sym(table.sym("x")));
        assert_eq!(
            parse_tdim(&table, "-y").unwrap(),
            TDim::MulInt(-1, Box::new(table.sym("y").into()))
        );
    }

    #[test]
    fn parse_bin() {
        let table = SymbolTable::default();
        assert_eq!(parse_tdim(&table, "1+2").unwrap(), 3.into());
        assert_eq!(parse_tdim(&table, "1-2").unwrap(), (-1).into());
        assert_eq!(parse_tdim(&table, "1*2").unwrap(), 2.into());
        assert_eq!(parse_tdim(&table, "1/2").unwrap(), 0.into());
    }

    #[test]
    fn parse_prio() {
        let table = SymbolTable::default();
        assert_eq!(parse_tdim(&table, "1+2*3").unwrap(), 7.into());
        assert_eq!(parse_tdim(&table, "1*2+3").unwrap(), 5.into());
    }
}
