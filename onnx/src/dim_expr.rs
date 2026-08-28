//! Parses an ONNX `dim_param` as a rational expression and translates it to `TDim`.
//!
//! ONNX shape expressions are rational: in `floor(W/2 - 1/2) + 1`, `1/2` is one
//! half. `TDim` is integral, so the two are different algebras and the
//! translation has to be explicit. Parsing a `dim_param` with `TDim`'s own
//! parser instead folds `1/2` to `0`.
//!
//! The expression is normalised so that every `floor` applies to a single
//! fraction, at which point the `floor` can be dropped: `TDim`'s truncating
//! division agrees with it over the non-negative values a dimension takes.

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{alpha1, alphanumeric1, digit1, one_of};
use nom::combinator::{all_consuming, map, map_opt, map_res, recognize};
use nom::multi::{fold, many0, separated_list0};
use nom::sequence::{delimited, pair, preceded};
use nom::{IResult, Parser};
use nom_language::error::VerboseError;
use num_rational::Ratio;
use num_traits::{CheckedAdd, CheckedDiv, CheckedMul, One, Zero};
use tract_hir::internal::*;

type R<'i, O> = IResult<&'i str, O, VerboseError<&'i str>>;
type Rat = Ratio<i64>;

/// A `dim_param` expression, before translation to `TDim`.
///
/// Kept distinct from `TDim` so that rational division cannot reach tract's
/// integer division except through [`parse_onnx_dim`].
#[derive(Clone, PartialEq, Debug)]
enum Expr {
    Rat(Rat),
    Sym(String),
    Add(Vec<Expr>),
    Mul(Vec<Expr>),
    /// Rational division.
    Div(Box<Expr>, Box<Expr>),
    Floor(Box<Expr>),
    Min(Vec<Expr>),
    Max(Vec<Expr>),
    Broadcast(Vec<Expr>),
}

/// A rational linear form, `sum(coeff * atom) + konst`, where an atom is a
/// symbol or a sub-expression that is not linear over the symbols. Equal atoms
/// combine.
#[derive(Clone, Debug, Default)]
struct Lin {
    terms: Vec<(Rat, Expr)>,
    konst: Rat,
}

fn checked(r: Option<Rat>) -> TractResult<Rat> {
    r.context("overflow in a shape expression")
}

impl Lin {
    fn konst(k: Rat) -> Lin {
        Lin { terms: vec![], konst: k }
    }

    fn atom(e: Expr) -> Lin {
        Lin { terms: vec![(Rat::one(), e)], konst: Rat::zero() }
    }

    fn add(mut self, o: Lin) -> TractResult<Lin> {
        self.konst = checked(self.konst.checked_add(&o.konst))?;
        for (c, a) in o.terms {
            match self.terms.iter_mut().find(|(_, b)| *b == a) {
                Some(slot) => slot.0 = checked(slot.0.checked_add(&c))?,
                None => self.terms.push((c, a)),
            }
        }
        self.terms.retain(|(c, _)| !c.is_zero());
        Ok(self)
    }

    fn scale(mut self, k: Rat) -> TractResult<Lin> {
        self.konst = checked(self.konst.checked_mul(&k))?;
        for t in self.terms.iter_mut() {
            t.0 = checked(t.0.checked_mul(&k))?;
        }
        self.terms.retain(|(c, _)| !c.is_zero());
        Ok(self)
    }

    /// Scales the form until every coefficient and the constant are integers,
    /// returning it with the strictly positive denominator divided out.
    fn as_single_fraction(&self) -> TractResult<(Lin, i64)> {
        let mut den: i64 = *self.konst.denom();
        for (c, _) in &self.terms {
            den = checked_lcm(den, *c.denom())?;
        }
        Ok((self.clone().scale(Rat::from_integer(den))?, den))
    }
}

fn checked_lcm(a: i64, b: i64) -> TractResult<i64> {
    (a / num_integer::gcd(a, b))
        .checked_mul(b)
        .filter(|d| *d > 0)
        .context("overflow reducing a shape expression to a single fraction")
}

macro_rules! bin {
    ($name: ident, $next: ident, $op: expr, $builder: expr) => {
        fn $name(input: &str) -> R<'_, Expr> {
            let (input, first) = $next(input)?;
            fold(0.., preceded(stag($op), |i| $next(i)), move || first.clone(), $builder)
                .parse(input)
        }
    };
}

fn expr(i: &str) -> R<'_, Expr> {
    add(i)
}

bin!(add, sub, "+", |a, b| Expr::Add(vec![a, b]));
bin!(sub, mul, "-", |a, b| Expr::Add(vec![a, negate(b)]));
bin!(mul, div, "*", |a, b| Expr::Mul(vec![a, b]));
bin!(div, atom, "/", |a, b| Expr::Div(Box::new(a), Box::new(b)));

fn negate(e: Expr) -> Expr {
    Expr::Mul(vec![Expr::Rat(Rat::from_integer(-1)), e])
}

fn atom(i: &str) -> R<'_, Expr> {
    alt((
        map(numeric, |n| Expr::Rat(Rat::from_integer(n))),
        map_opt(
            |i| func("floor", i),
            |xs: Vec<Expr>| {
                let [x] = <[Expr; 1]>::try_from(xs).ok()?;
                Some(Expr::Floor(Box::new(x)))
            },
        ),
        map(|i| func("min", i), Expr::Min),
        map(|i| func("max", i), Expr::Max),
        map(|i| func("broadcast", i), Expr::Broadcast),
        map(identifier, Expr::Sym),
        map(pair(recognize(stag("-")), atom), |(_, e)| negate(e)),
        delimited(stag("("), expr, stag(")")),
    ))
    .parse(i)
}

fn func<'i>(name: &'static str, i: &'i str) -> R<'i, Vec<Expr>> {
    preceded(stag(name), delimited(stag("("), separated_list0(stag(","), expr), stag(")"))).parse(i)
}

fn identifier(i: &str) -> R<'_, String> {
    map(
        recognize(pair(alt((alpha1, tag("_"))), many0(alt((alphanumeric1, tag("_"), tag(".")))))),
        String::from,
    )
    .parse(i)
}

fn numeric(i: &str) -> R<'_, i64> {
    map_res(digit1, std::str::FromStr::from_str).parse(i)
}

fn spaces(i: &str) -> R<'_, ()> {
    map(many0(one_of(" \t\n\r")), |_| ()).parse(i)
}

fn stag<'s>(
    t: &'static str,
) -> impl Parser<&'s str, Output = &'s str, Error = VerboseError<&'s str>> {
    delimited(spaces, tag(t), spaces)
}

/// Rewrites `floor(floor(x)/n)` to `floor(x/n)` for integer `n > 0`, so that
/// [`linearise`] sees a single fraction rather than a floor over a divisor.
fn flatten_nested_floor(e: Expr) -> Expr {
    match e {
        Expr::Floor(inner) => {
            let inner = flatten_nested_floor(*inner);
            if let Expr::Div(num, den) = &inner
                && let (Expr::Floor(x), Expr::Rat(d)) = (&**num, &**den)
                && d.is_integer()
                && *d.numer() > 0
            {
                return Expr::Floor(Box::new(Expr::Div(x.clone(), den.clone())));
            }
            Expr::Floor(Box::new(inner))
        }
        Expr::Add(v) => Expr::Add(v.into_iter().map(flatten_nested_floor).collect()),
        Expr::Mul(v) => Expr::Mul(v.into_iter().map(flatten_nested_floor).collect()),
        Expr::Min(v) => Expr::Min(v.into_iter().map(flatten_nested_floor).collect()),
        Expr::Max(v) => Expr::Max(v.into_iter().map(flatten_nested_floor).collect()),
        Expr::Broadcast(v) => Expr::Broadcast(v.into_iter().map(flatten_nested_floor).collect()),
        Expr::Div(a, b) => {
            Expr::Div(Box::new(flatten_nested_floor(*a)), Box::new(flatten_nested_floor(*b)))
        }
        other => other,
    }
}

/// Rewrites to a rational linear form, treating anything not linear over the
/// symbols as an opaque atom.
fn linearise(e: &Expr) -> TractResult<Lin> {
    Ok(match e {
        Expr::Rat(r) => Lin::konst(*r),
        Expr::Sym(_) => Lin::atom(e.clone()),
        Expr::Add(v) => {
            ensure!(!v.is_empty(), "empty sum in a shape expression");
            let mut acc = Lin::default();
            for x in v {
                acc = acc.add(linearise(x)?)?;
            }
            acc
        }
        Expr::Mul(v) => {
            let mut scalar = Rat::one();
            let mut non_const: Option<Lin> = None;
            for x in v {
                let l = linearise(x)?;
                if l.terms.is_empty() {
                    scalar = checked(scalar.checked_mul(&l.konst))?;
                } else if non_const.is_none() {
                    non_const = Some(l);
                } else {
                    return Ok(Lin::atom(e.clone()));
                }
            }
            match non_const {
                Some(l) => l.scale(scalar)?,
                None => Lin::konst(scalar),
            }
        }
        Expr::Div(a, b) => {
            let rhs = linearise(b)?;
            ensure!(rhs.terms.is_empty(), "cannot divide by a symbol: {e:?}");
            linearise(a)?.scale(checked(Rat::one().checked_div(&rhs.konst))?)?
        }
        Expr::Floor(_) | Expr::Min(_) | Expr::Max(_) | Expr::Broadcast(_) => Lin::atom(e.clone()),
    })
}

fn lin_to_tdim(lin: &Lin, scope: &SymbolScope) -> TractResult<TDim> {
    let (scaled, den) = lin.as_single_fraction()?;
    let mut sum: TDim = scaled.konst.to_integer().to_dim();
    for (c, atom) in &scaled.terms {
        sum += atom_to_tdim(atom, scope)? * c.to_integer();
    }
    if den == 1 { Ok(sum) } else { Ok(sum / den) }
}

fn atom_to_tdim(e: &Expr, scope: &SymbolScope) -> TractResult<TDim> {
    Ok(match e {
        Expr::Sym(name) => TDim::Sym(scope.sym(name)),
        Expr::Floor(inner) => to_tdim(inner, scope)?,
        Expr::Min(v) => TDim::Min(v.iter().map(|x| to_tdim(x, scope)).collect::<TractResult<_>>()?),
        Expr::Max(v) => TDim::Max(v.iter().map(|x| to_tdim(x, scope)).collect::<TractResult<_>>()?),
        Expr::Broadcast(v) => {
            TDim::Broadcast(v.iter().map(|x| to_tdim(x, scope)).collect::<TractResult<_>>()?)
        }
        other => bail!("cannot translate shape sub-expression {other:?}"),
    })
}

fn to_tdim(e: &Expr, scope: &SymbolScope) -> TractResult<TDim> {
    lin_to_tdim(&linearise(e)?, scope)
}

/// Parses an ONNX `dim_param` and translates it to a `TDim`.
///
/// Accepts integers, identifiers, `+ - * /`, parentheses and `floor`, `min`,
/// `max` and `broadcast` calls, with `/` read as rational division. Fails,
/// naming the offending sub-expression, rather than returning an approximation.
pub fn parse_onnx_dim(scope: &SymbolScope, s: &str) -> TractResult<TDim> {
    let e = match all_consuming(expr).parse(s) {
        Ok((_, e)) => e,
        Err(e) => bail!("Failed to parse dim_param {s:?}, {e:?}"),
    };
    to_tdim(&flatten_nested_floor(e), scope).with_context(|| format!("dim_param {s:?}"))
}

#[cfg(test)]
mod test {
    use super::*;
    use std::collections::HashMap;

    /// Evaluates the ONNX reading of an expression in exact rational
    /// arithmetic, without touching tract's integer division.
    fn eval_rational(e: &Expr, vals: &HashMap<String, i64>) -> TractResult<Rat> {
        Ok(match e {
            Expr::Rat(r) => *r,
            Expr::Sym(n) => {
                Rat::from_integer(*vals.get(n).with_context(|| format!("no value for {n}"))?)
            }
            Expr::Add(v) => {
                let mut acc = Rat::zero();
                for x in v {
                    acc = checked(acc.checked_add(&eval_rational(x, vals)?))?;
                }
                acc
            }
            Expr::Mul(v) => {
                let mut acc = Rat::one();
                for x in v {
                    acc = checked(acc.checked_mul(&eval_rational(x, vals)?))?;
                }
                acc
            }
            Expr::Div(a, b) => {
                checked(eval_rational(a, vals)?.checked_div(&eval_rational(b, vals)?))?
            }
            Expr::Floor(x) => Rat::from_integer(eval_rational(x, vals)?.floor().to_integer()),
            Expr::Min(v) => {
                let mut acc = eval_rational(v.first().context("min()")?, vals)?;
                for x in &v[1..] {
                    acc = acc.min(eval_rational(x, vals)?);
                }
                acc
            }
            Expr::Max(v) => {
                let mut acc = eval_rational(v.first().context("max()")?, vals)?;
                for x in &v[1..] {
                    acc = acc.max(eval_rational(x, vals)?);
                }
                acc
            }
            Expr::Broadcast(_) => bail!("broadcast has no rational reading"),
        })
    }

    fn show(s: &str) -> String {
        let scope = SymbolScope::default();
        parse_onnx_dim(&scope, s).unwrap().to_string()
    }

    fn tdim(s: &str) -> TractResult<TDim> {
        parse_onnx_dim(&SymbolScope::default(), s)
    }

    /// Asserts the translated `TDim` and the exact rational reading agree at
    /// every listed value of `sym`.
    fn agree_at(src: &str, sym: &str, values: &[i64]) {
        let scope = SymbolScope::default();
        let dim = parse_onnx_dim(&scope, src).unwrap();
        let parsed = flatten_nested_floor(all_consuming(expr).parse(src).unwrap().1);
        for &v in values {
            let vals = HashMap::from([(sym.to_string(), v)]);
            let want = eval_rational(&parsed, &vals).unwrap();
            assert!(want.is_integer(), "{src} at {sym}={v} is not integral: {want}");
            let got = dim.eval(&SymbolValues::default().with(&scope.sym(sym), v)).to_i64().unwrap();
            assert_eq!(got, want.to_integer(), "{src} at {sym}={v}: {dim} gave {got}");
        }
    }

    #[test]
    fn fractional_literal_is_not_integer_division() {
        agree_at("floor(W/2 - 1/2) + 1", "W", &[1, 2, 3, 4, 5, 6, 7, 8, 9, 318, 319, 320, 321]);
    }

    #[test]
    fn even_and_odd_widths() {
        let scope = SymbolScope::default();
        let d = parse_onnx_dim(&scope, "floor(W/2 - 1/2) + 1").unwrap();
        let at =
            |v: i64| d.eval(&SymbolValues::default().with(&scope.sym("W"), v)).to_i64().unwrap();
        assert_eq!(at(4), 2);
        assert_eq!(at(5), 3);
        assert_eq!(at(6), 3);
        assert_eq!(at(320), 160);
        assert_eq!(at(321), 161);
    }

    #[test]
    fn nested_floor() {
        agree_at(
            "floor(floor(W/2 - 1/2)/2) + 1",
            "W",
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 318, 319, 320, 321],
        );
    }

    #[test]
    fn reduces_to_a_single_division() {
        assert_eq!(show("floor(W/2 - 1/2) + 1"), "(W+1)/2");
        assert_eq!(show("floor(floor(W/2 - 1/2)/2) + 1"), "(W+3)/4");
    }

    #[test]
    fn symbols_and_integers() {
        assert_eq!(show("N"), "N");
        assert_eq!(show("42"), "42");
        assert_eq!(show("DynamicDimension.1"), "DynamicDimension.1");
    }

    #[test]
    fn arithmetic_without_floor() {
        agree_at("2*W + 3", "W", &[0, 1, 7, 1000]);
        agree_at("(W - 1)*2", "W", &[1, 2, 50]);
        agree_at("W/2", "W", &[0, 2, 4, 100]);
    }

    #[test]
    fn equal_atoms_combine() {
        agree_at("floor(W/2) + floor(W/2)", "W", &[0, 1, 2, 3, 4, 5, 17]);
    }

    #[test]
    fn min_and_max() {
        assert!(tdim("min(N, 4)").is_ok());
        assert!(tdim("max(N, 4)").is_ok());
    }

    #[test]
    fn division_by_a_symbol_is_refused() {
        assert!(format!("{:?}", tdim("W/N").unwrap_err()).contains("divide by a symbol"));
    }

    #[test]
    fn unknown_function_is_refused() {
        assert!(tdim("ceil(W/2)").is_err());
    }

    #[test]
    fn malformed_input_is_refused() {
        for s in ["W +", "(W", "W)", "", "$", "W/0", "floor()", "floor(W,2)"] {
            assert!(tdim(s).is_err(), "{s:?} should not parse");
        }
    }

    #[test]
    fn overflow_is_reported() {
        assert!(checked(Rat::from_integer(i64::MAX).checked_mul(&Rat::from_integer(2))).is_err());
    }

    fn import_dim_param(expr: &str) -> String {
        use crate::pb::*;
        use prost::Message;
        use tract_hir::internal::Framework;

        let dim = tensor_shape_proto::Dimension {
            denotation: String::new(),
            value: Some(tensor_shape_proto::dimension::Value::DimParam(expr.to_string())),
        };
        let tensor = type_proto::Tensor {
            elem_type: tensor_proto::DataType::Float as i32,
            shape: Some(TensorShapeProto { dim: vec![dim] }),
        };
        let value_info = ValueInfoProto {
            name: "x".to_string(),
            r#type: Some(TypeProto {
                denotation: String::new(),
                value: Some(type_proto::Value::TensorType(tensor)),
            }),
            doc_string: String::new(),
        };
        let mut out_info = value_info.clone();
        out_info.name = "y".to_string();
        // An empty graph never applies the declared input fact.
        let node = NodeProto {
            name: "id".to_string(),
            op_type: "Identity".to_string(),
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "g".to_string(),
            input: vec![value_info],
            output: vec![out_info],
            ..GraphProto::default()
        };
        let model = ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto { domain: String::new(), version: 13 }],
            graph: Some(graph),
            ..ModelProto::default()
        };
        let mut buf = vec![];
        model.encode(&mut buf).unwrap();
        let loaded = crate::onnx().model_for_read(&mut &*buf).unwrap();
        format!("{:?}", loaded.input_fact(0).unwrap().shape)
    }

    #[test]
    fn importer_round_trip() {
        assert!(import_dim_param("W").contains('W'));
        assert!(import_dim_param("floor(W/2 - 1/2) + 1").contains("(W+1)/2"));
        assert!(import_dim_param("floor(floor(W/2 - 1/2)/2) + 1").contains("(W+3)/4"));
    }

    #[test]
    fn untranslatable_dim_param_imports_as_unknown() {
        assert!(!import_dim_param("Mod(s0, 8)").contains("Mod"));
    }
}
