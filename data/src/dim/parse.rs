use super::*;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{alpha1, alphanumeric1, digit1, one_of};
use nom::combinator::{all_consuming, map, map_res, recognize};
use nom::multi::{many0, separated_list0};
use nom::sequence::{delimited, pair, preceded, separated_pair};
use nom::{IResult, Parser};
use nom_language::error::VerboseError;

type R<'i, O> = IResult<&'i str, O, VerboseError<&'i str>>;

pub fn parse_tdim(symbol_table: &SymbolScope, input: &str) -> TractResult<TDim> {
    match all_consuming(|i| expr(symbol_table, i)).parse(input) {
        Ok(pair) => Ok(pair.1),
        Err(e) => bail!("Failed to parse {:?}, {:?}", input, e),
    }
}

pub fn parse_assertion(symbol_table: &SymbolScope, input: &str) -> TractResult<Assertion> {
    match all_consuming(|i| assertion(symbol_table, i)).parse(input) {
        Ok(pair) => Ok(pair.1),
        Err(e) => bail!("Failed to parse {:?}, {:?}", input, e),
    }
}

fn assertion<'i>(s: &SymbolScope, i: &'i str) -> R<'i, Assertion> {
    delimited(
        spaces,
        alt((
            map(separated_pair(|i| expr(s, i), stag("=="), |i| expr(s, i)), |(a, b)| {
                Assertion::Eq(a, b)
            }),
            map(separated_pair(|i| expr(s, i), stag("<="), |i| expr(s, i)), |(a, b)| {
                Assertion::LTE(a, b)
            }),
            map(separated_pair(|i| expr(s, i), stag(">="), |i| expr(s, i)), |(a, b)| {
                Assertion::GTE(a, b)
            }),
            map(separated_pair(|i| expr(s, i), stag("<"), |i| expr(s, i)), |(a, b)| {
                Assertion::LT(a, b)
            }),
            map(separated_pair(|i| expr(s, i), stag(">"), |i| expr(s, i)), |(a, b)| {
                Assertion::GT(a, b)
            }),
        )),
        spaces,
    )
    .parse(i)
}

fn expr<'i>(symbol_table: &SymbolScope, i: &'i str) -> R<'i, TDim> {
    broadcast(symbol_table, i)
}

macro_rules! bin {
    ($name: ident, $next: ident, $op: expr, $builder: expr) => {
        fn $name<'i>(symbol_table: &SymbolScope, input: &'i str) -> R<'i, TDim> {
            let s = symbol_table;
            alt((map(separated_pair(|i| $next(s, i), stag($op), |i| $next(s, i)), $builder), |i| {
                $next(s, i)
            }))
            .parse(input)
        }
    };
}
bin!(add, sub, "+", |(a, b)| a + b);
bin!(sub, mul, "-", |(a, b)| a - b);
bin!(mul, div, "*", |(a, b)| a * b);

fn broadcast<'i>(symbol_table: &SymbolScope, input: &'i str) -> R<'i, TDim> {
    let s = symbol_table;
    alt((
        map_res(separated_pair(|i| add(s, i), stag("#"), |i| add(s, i)), |(a, b)| a.broadcast(b)),
        |i| add(s, i),
    ))
    .parse(input)
}

fn div<'i>(symbol_table: &SymbolScope, input: &'i str) -> R<'i, TDim> {
    let s = symbol_table;
    alt((map(separated_pair(|i| atom(s, i), stag("/"), numeric), |(a, b)| a / b), |i| atom(s, i)))
        .parse(input)
}

fn atom<'i>(symbol_table: &SymbolScope, i: &'i str) -> R<'i, TDim> {
    alt((
        map(numeric, TDim::Val),
        map(|i| func(symbol_table, "min", i), TDim::Min),
        map(|i| func(symbol_table, "max", i), TDim::Max),
        map(|i| identifier(symbol_table, i), TDim::Sym),
        map(pair(recognize(stag("-")), |i| atom(symbol_table, i)), |(_, dim)| dim * -1),
        delimited(stag("("), |i| expr(symbol_table, i), stag(")")),
    ))
    .parse(i)
}

fn func<'i>(symbol_table: &SymbolScope, name: &'static str, i: &'i str) -> R<'i, Vec<TDim>> {
    preceded(
        stag(name),
        delimited(stag("("), separated_list0(stag(","), |i| expr(symbol_table, i)), stag(")")),
    )
    .parse(i)
}

fn identifier<'i>(symbol_table: &SymbolScope, i: &'i str) -> R<'i, Symbol> {
    map(recognize(pair(alt((alpha1, tag("_"))), many0(alt((alphanumeric1, tag("_")))))), |s| {
        symbol_table.sym(s)
    })
    .parse(i)
}

fn numeric(i: &str) -> R<'_, i64> {
    map_res(digit1, std::str::FromStr::from_str).parse(i)
}

fn spaces(i: &str) -> R<'_, ()> {
    map(many0(one_of(" \t\n\r")), |_| ()).parse(i)
}

fn spaced<'s, O, P>(it: P) -> impl Parser<&'s str, Output = O, Error = VerboseError<&'s str>>
where
    P: Parser<&'s str, Output = O, Error = VerboseError<&'s str>>,
{
    delimited(spaces, it, spaces)
}

pub(super) fn stag<'s>(
    t: &'static str,
) -> impl Parser<&'s str, Output = &'s str, Error = VerboseError<&'s str>> {
    spaced(tag(t))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse_int() {
        let table = SymbolScope::default();
        assert_eq!(parse_tdim(&table, "12").unwrap(), TDim::Val(12));
        assert_eq!(parse_tdim(&table, "-12").unwrap(), TDim::Val(-12));
    }

    #[test]
    fn parse_sym() {
        let table = SymbolScope::default();
        assert_eq!(parse_tdim(&table, "x").unwrap(), TDim::Sym(table.sym("x")));
        assert_eq!(
            parse_tdim(&table, "-y").unwrap(),
            TDim::MulInt(-1, Box::new(table.sym("y").into()))
        );
    }

    #[test]
    fn parse_bin() {
        let table = SymbolScope::default();
        assert_eq!(parse_tdim(&table, "1+2").unwrap(), 3.into());
        assert_eq!(parse_tdim(&table, "1-2").unwrap(), (-1).into());
        assert_eq!(parse_tdim(&table, "1*2").unwrap(), 2.into());
        assert_eq!(parse_tdim(&table, "1/2").unwrap(), 0.into());
    }

    #[test]
    fn parse_prio() {
        let table = SymbolScope::default();
        assert_eq!(parse_tdim(&table, "1+2*3").unwrap(), 7.into());
        assert_eq!(parse_tdim(&table, "1*2+3").unwrap(), 5.into());
    }

    #[test]
    fn parse_min() {
        let table = SymbolScope::default();
        assert_eq!(
            parse_tdim(&table, "min(P,S)").unwrap(),
            TDim::Min(vec!(table.sym("P").into(), table.sym("S").into()))
        );
    }

    #[test]
    fn parse_inequality_0() {
        let table = SymbolScope::default();
        assert_eq!(
            parse_assertion(&table, "P+S<4096").unwrap(),
            Assertion::LT(parse_tdim(&table, "P+S").unwrap(), 4096.to_dim())
        );
    }
}
