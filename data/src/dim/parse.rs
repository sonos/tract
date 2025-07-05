use super::*;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{alpha1, alphanumeric1, digit1, one_of};
use nom::combinator::{all_consuming, map, map_res, recognize};
use nom::error::{Error, FromExternalError, ParseError};
use nom::multi::{many0, separated_list0};
use nom::sequence::{delimited, pair, preceded, separated_pair};
use nom::{IResult, Parser};

pub fn parse_tdim(symbol_table: &SymbolScope, input: &str) -> TractResult<TDim> {
    match all_consuming(|i| expr::<Error<_>>(symbol_table, i)).parse(input) {
        Ok(pair) => Ok(pair.1),
        Err(e) => bail!("Failed to parse {:?}, {:?}", input, e),
    }
}

pub fn parse_assertion(symbol_table: &SymbolScope, input: &str) -> TractResult<Assertion> {
    match all_consuming(|i| assertion::<Error<_>>(symbol_table, i)).parse(input) {
        Ok(pair) => Ok(pair.1),
        Err(e) => bail!("Failed to parse {:?}, {:?}", input, e),
    }
}

fn assertion<'i, E: ParseError<&'i str>>(
    s: &SymbolScope,
    i: &'i str,
) -> IResult<&'i str, Assertion, E> {
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

fn expr<'i, E: ParseError<&'i str>>(
    symbol_table: &SymbolScope,
    i: &'i str,
) -> IResult<&'i str, TDim, E> {
    broadcast(symbol_table, i)
}

macro_rules! bin {
    ($name: ident, $next: ident, $op: expr, $builder: expr) => {
        fn $name<'i, E: ParseError<&'i str>>(
            symbol_table: &SymbolScope,
            input: &'i str,
        ) -> IResult<&'i str, TDim, E> {
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

fn broadcast<'i, E: ParseError<&'i str>>(
    symbol_table: &SymbolScope,
    input: &'i str,
) -> IResult<&'i str, TDim, E> {
    let s = symbol_table;
    alt((
        map(separated_pair(|i| add(s, i), stag("#"), |i| add(s, i)), |(a, b)| {
            a.broadcast(b).unwrap() // FIXME
        }),
        |i| add(s, i),
    ))
    .parse(input)
}

fn div<'i, E: ParseError<&'i str>>(
    symbol_table: &SymbolScope,
    input: &'i str,
) -> IResult<&'i str, TDim, E> {
    let s = symbol_table;
    alt((map(separated_pair(|i| atom(s, i), stag("/"), numeric), |(a, b)| a / b), |i| atom(s, i)))
        .parse(input)
}

fn atom<'i, E: ParseError<&'i str>>(
    symbol_table: &SymbolScope,
    i: &'i str,
) -> IResult<&'i str, TDim, E> {
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

fn func<'i, E: ParseError<&'i str>>(
    symbol_table: &SymbolScope,
    name: &'static str,
    i: &'i str,
) -> IResult<&'i str, Vec<TDim>, E> {
    preceded(
        stag(name),
        delimited(stag("("), separated_list0(stag(","), |i| expr(symbol_table, i)), stag(")")),
    )
    .parse(i)
}

fn identifier<'i, E: ParseError<&'i str>>(
    symbol_table: &SymbolScope,
    i: &'i str,
) -> IResult<&'i str, Symbol, E> {
    map(recognize(pair(alt((alpha1, tag("_"))), many0(alt((alphanumeric1, tag("_")))))), |s| {
        symbol_table.sym(s)
    })
    .parse(i)
}

fn numeric<'i, E: ParseError<&'i str>>(i: &'i str) -> IResult<&'i str, i64, E> {
    map(digit1, |d| std::str::FromStr::from_str(d).unwrap()).parse(i)
}

fn spaces<'s, E: ParseError<&'s str>>(i: &'s str) -> IResult<&'s str, (), E> {
    map(many0(one_of(" \t\n\r")), |_| ()).parse(i)
}

fn spaced<'s, O, E: ParseError<&'s str>, P>(it: P) -> impl Parser<&'s str, Output = O, Error = E>
where
    P: Parser<&'s str, Output = O, Error = E>,
{
    delimited(spaces, it, spaces)
}

pub(super) fn stag<'s, E: ParseError<&'s str>>(
    t: &'static str,
) -> impl Parser<&'s str, Output = &'s str, Error = E> {
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
