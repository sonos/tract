//! ONNX shape expressions, parsed as rational and translated to `TDim`.
//!
//! # Why this exists
//!
//! ONNX writes a dynamic dimension as a string in `dim_param`, and exporters
//! write real arithmetic into it. paddle2onnx, for one, emits
//!
//! ```text
//! floor(DynamicDimension.1/2 - 1/2) + 1
//! ```
//!
//! **That expression is rational.** `1/2` means one half. `TDim`, on the other
//! hand, adopted integer arithmetic years ago and deliberately keeps it: the
//! convolution shape rules are written against integer division, and moving
//! `TDim` to rationals would put dozens of subtle rules — and business-critical
//! models — at risk. Both algebras are correct; they are simply not the same
//! algebra.
//!
//! The bug was the **implicit coercion between them**. `dim_param` strings were
//! handed straight to `tract_data`'s `parse_tdim`, which evaluated `1/2` with
//! integer division. It silently became `0`, so the expression above reduced to
//! `W/2 + 1` — off by one for every even `W`, with no error anywhere. See
//! <https://github.com/sonos/tract/issues/2724>.
//!
//! # What this module does
//!
//! It makes the translation an explicit step instead of a parser accident:
//!
//! 1. parse the `dim_param` as a **rational** expression;
//! 2. normalise it so every `floor` applies to a **single fraction**;
//! 3. drop those `floor`s, because a floored single fraction is exactly what
//!    `TDim`'s truncating division already computes over the non-negative
//!    values a shape can take;
//! 4. hand back a `TDim`, or fail with a message naming what could not be
//!    translated.
//!
//! Step 2 is the whole trick. `floor` of a *sum* cannot be dropped —
//! `floor(W/2 - 1/2)` is not `W/2 - 0` — but `floor` of a *single fraction*
//! can, so the reduction is what earns the right to ignore it:
//!
//! ```text
//! floor(W/2 - 1/2) + 1   -->   floor((W-1)/2) + 1   -->   (W-1)/2 + 1
//! ```
//!
//! # The nested case
//!
//! Two stacked stride-2 convolutions produce
//!
//! ```text
//! floor(floor(DynamicDimension.1/2 - 1/2)/2) + 1
//! ```
//!
//! which turns up in essentially any real CNN export rather than being a quirk
//! of one model. It is not a single fraction as written, and reducing the inner
//! floor alone does not make it one. It needs the identity
//!
//! ```text
//! floor(floor(x)/n) == floor(x/n)     for any real x and integer n > 0
//! ```
//!
//! applied first, after which the same reduction runs and the whole thing
//! collapses to `(W+3)/4`. Without that step the translation still produces a
//! *correct* `TDim` (nested truncating divisions agree with nested floors on
//! non-negative values), but a far uglier one that tract then has to unify
//! against the shape it computes itself.

use tract_hir::internal::*;

/// A rational constant, always in lowest terms with a strictly positive
/// denominator.
///
/// Deliberately `i64` rather than a bignum: shape expressions are small, and a
/// dependency is a poor trade for a case that does not occur. Every operation
/// is checked, and an overflow is reported rather than wrapped — a silently
/// wrong shape is the exact failure this module exists to end.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Rat {
    num: i64,
    den: i64,
}

/// ⚠️ Written out rather than derived. A derived `Default` gives `den: 0`,
/// which is not a rational at all, and every operation below assumes a positive
/// denominator.
impl Default for Rat {
    fn default() -> Rat {
        Rat::ZERO
    }
}

impl Rat {
    const ZERO: Rat = Rat { num: 0, den: 1 };
    const ONE: Rat = Rat { num: 1, den: 1 };

    fn int(n: i64) -> Rat {
        Rat { num: n, den: 1 }
    }

    fn new(num: i64, den: i64) -> TractResult<Rat> {
        ensure!(den != 0, "division by zero in a shape expression");
        let (num, den) = if den < 0 {
            (num.checked_neg().context("overflow negating a shape expression")?, -den)
        } else {
            (num, den)
        };
        let g = num_integer::gcd(num.abs(), den).max(1);
        Ok(Rat { num: num / g, den: den / g })
    }

    fn is_int(&self) -> bool {
        self.den == 1
    }

    fn add(self, o: Rat) -> TractResult<Rat> {
        let a = self.num.checked_mul(o.den).context("overflow adding a shape expression")?;
        let b = o.num.checked_mul(self.den).context("overflow adding a shape expression")?;
        let n = a.checked_add(b).context("overflow adding a shape expression")?;
        let d = self.den.checked_mul(o.den).context("overflow adding a shape expression")?;
        Rat::new(n, d)
    }

    fn mul(self, o: Rat) -> TractResult<Rat> {
        let n = self.num.checked_mul(o.num).context("overflow multiplying a shape expression")?;
        let d = self.den.checked_mul(o.den).context("overflow multiplying a shape expression")?;
        Rat::new(n, d)
    }

    fn div(self, o: Rat) -> TractResult<Rat> {
        ensure!(o.num != 0, "division by zero in a shape expression");
        self.mul(Rat::new(o.den, o.num)?)
    }
}

/// The parse tree of a `dim_param`, before any rational reasoning.
///
/// This is intentionally a *different* type from `TDim`. Keeping them separate
/// is what stops the two algebras being confused again: nothing here can reach
/// tract's integer division without going through [`to_tdim`].
#[derive(Clone, PartialEq, Debug)]
enum Expr {
    Rat(Rat),
    Sym(String),
    Add(Vec<Expr>),
    Mul(Vec<Expr>),
    /// True rational division, NOT tract's truncating division.
    Div(Box<Expr>, Box<Expr>),
    Floor(Box<Expr>),
    Min(Vec<Expr>),
    Max(Vec<Expr>),
    Broadcast(Vec<Expr>),
}

/// A rational linear form: `sum(coeff * atom) + konst`.
///
/// An "atom" is a symbol, or an expression we could not linearise (a `floor`,
/// `min`, `max`, or a product of symbols). Opaque atoms are carried along by
/// their key so that identical ones combine — `floor(x) + floor(x)` becomes
/// `2*floor(x)` rather than two separate terms.
#[derive(Clone, Debug, Default)]
struct Lin {
    terms: Vec<(Rat, Expr)>,
    konst: Rat,
}

impl Lin {
    fn konst(k: Rat) -> Lin {
        Lin { terms: vec![], konst: k }
    }

    fn atom(e: Expr) -> Lin {
        Lin { terms: vec![(Rat::ONE, e)], konst: Rat::ZERO }
    }

    fn add(mut self, o: Lin) -> TractResult<Lin> {
        self.konst = self.konst.add(o.konst)?;
        for (c, a) in o.terms {
            match self.terms.iter_mut().find(|(_, b)| *b == a) {
                Some(slot) => slot.0 = slot.0.add(c)?,
                None => self.terms.push((c, a)),
            }
        }
        self.terms.retain(|(c, _)| c.num != 0);
        Ok(self)
    }

    fn scale(mut self, k: Rat) -> TractResult<Lin> {
        self.konst = self.konst.mul(k)?;
        for t in self.terms.iter_mut() {
            t.0 = t.0.mul(k)?;
        }
        self.terms.retain(|(c, _)| c.num != 0);
        Ok(self)
    }

    /// The single fraction this form reduces to: `(integer_terms, denominator)`
    /// such that the value is `sum(coeff * atom) + konst` over `denominator`,
    /// with every coefficient an integer and the denominator strictly positive.
    ///
    /// This is the step that earns the right to drop a `floor`.
    fn as_single_fraction(&self) -> TractResult<(Lin, i64)> {
        let mut den: i64 = self.konst.den;
        for (c, _) in &self.terms {
            den = num_integer::lcm(den, c.den);
            ensure!(den > 0, "overflow reducing a shape expression to one fraction");
        }
        let scaled = self.clone().scale(Rat::int(den))?;
        debug_assert!(scaled.konst.is_int() && scaled.terms.iter().all(|(c, _)| c.is_int()));
        Ok((scaled, den))
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// A hand-written recursive-descent parser.
///
/// `tract_data`'s `TDim` parser uses nom; this one does not, on purpose. The
/// grammar is a dozen productions, `tract-onnx` carries no parser dependency
/// today, and adding one to express *this* would be the larger change. The
/// grammar it accepts is a superset of what the `TDim` parser accepts for the
/// same strings, minus the `#` broadcast operator, which no exporter emits.
struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Parser<'a> {
        Parser { s: s.as_bytes(), i: 0 }
    }

    fn ws(&mut self) {
        while self.i < self.s.len() && (self.s[self.i] as char).is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.ws();
        self.s.get(self.i).map(|c| *c as char)
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn rest(&self) -> String {
        String::from_utf8_lossy(&self.s[self.i.min(self.s.len())..]).into_owned()
    }

    /// `expr := term (('+' | '-') term)*`
    fn expr(&mut self) -> TractResult<Expr> {
        let mut acc = self.term()?;
        loop {
            if self.eat('+') {
                acc = Expr::Add(vec![acc, self.term()?]);
            } else if self.eat('-') {
                let rhs = self.term()?;
                acc = Expr::Add(vec![acc, Expr::Mul(vec![Expr::Rat(Rat::int(-1)), rhs])]);
            } else {
                return Ok(acc);
            }
        }
    }

    /// `term := unary (('*' | '/') unary)*`
    fn term(&mut self) -> TractResult<Expr> {
        let mut acc = self.unary()?;
        loop {
            if self.eat('*') {
                acc = Expr::Mul(vec![acc, self.unary()?]);
            } else if self.eat('/') {
                acc = Expr::Div(Box::new(acc), Box::new(self.unary()?));
            } else {
                return Ok(acc);
            }
        }
    }

    fn unary(&mut self) -> TractResult<Expr> {
        if self.eat('-') {
            Ok(Expr::Mul(vec![Expr::Rat(Rat::int(-1)), self.unary()?]))
        } else {
            self.atom()
        }
    }

    fn atom(&mut self) -> TractResult<Expr> {
        match self.peek() {
            None => bail!("unexpected end of shape expression"),
            Some('(') => {
                self.i += 1;
                let e = self.expr()?;
                ensure!(self.eat(')'), "expected ')' in a shape expression, at {:?}", self.rest());
                Ok(e)
            }
            Some(c) if c.is_ascii_digit() => {
                let start = self.i;
                while self.i < self.s.len() && (self.s[self.i] as char).is_ascii_digit() {
                    self.i += 1;
                }
                let text = std::str::from_utf8(&self.s[start..self.i])?;
                // A shape expression has no floating point literals: exporters
                // write `1/2`, never `0.5`. A '.' here belongs to an identifier
                // like `DynamicDimension.1`, and reaching this arm with one
                // would mean the identifier rule already had its chance.
                Ok(Expr::Rat(Rat::int(
                    text.parse::<i64>().context("shape expression integer out of range")?,
                )))
            }
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                let start = self.i;
                while self.i < self.s.len() {
                    let c = self.s[self.i] as char;
                    if c.is_ascii_alphanumeric() || c == '_' || c == '.' {
                        self.i += 1;
                    } else {
                        break;
                    }
                }
                let name = std::str::from_utf8(&self.s[start..self.i])?.to_string();
                // A name followed by '(' is a call. Only the functions ONNX
                // exporters actually emit are accepted; anything else is
                // reported rather than guessed at.
                if self.peek() == Some('(') {
                    self.i += 1;
                    let mut args = vec![];
                    if self.peek() != Some(')') {
                        loop {
                            args.push(self.expr()?);
                            if !self.eat(',') {
                                break;
                            }
                        }
                    }
                    ensure!(self.eat(')'), "expected ')' closing {name}(, at {:?}", self.rest());
                    return match name.as_str() {
                        "floor" => {
                            ensure!(
                                args.len() == 1,
                                "floor() takes one argument, got {}",
                                args.len()
                            );
                            Ok(Expr::Floor(Box::new(args.into_iter().next().unwrap())))
                        }
                        "min" => Ok(Expr::Min(args)),
                        "max" => Ok(Expr::Max(args)),
                        "broadcast" => Ok(Expr::Broadcast(args)),
                        _ => bail!("unsupported function {name}() in a shape expression"),
                    };
                }
                Ok(Expr::Sym(name))
            }
            Some(c) => bail!("unexpected {c:?} in a shape expression, at {:?}", self.rest()),
        }
    }
}

// ---------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------

/// `floor(floor(x)/n) == floor(x/n)` for integer `n > 0`.
///
/// Applied before linearisation, so the reduction below sees one fraction
/// rather than a floor divided by something. This is what reaches the nested
/// case that two stacked stride-2 convolutions produce; without it the result
/// is still correct but is a tower of divisions.
fn flatten_nested_floor(e: Expr) -> Expr {
    match e {
        Expr::Floor(inner) => {
            let inner = flatten_nested_floor(*inner);
            // floor( floor(x) / n )  ->  floor( x / n )
            if let Expr::Div(num, den) = &inner
                && let (Expr::Floor(x), Expr::Rat(d)) = (&**num, &**den)
                && d.is_int()
                && d.num > 0
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

/// Rewrite to a rational linear form.
///
/// Anything that is not linear over the symbols — a `floor`, a `min`, a product
/// of two symbols — becomes an opaque atom, normalised first so that equal
/// atoms compare equal and combine.
fn linearise(e: &Expr) -> TractResult<Lin> {
    Ok(match e {
        Expr::Rat(r) => Lin::konst(*r),
        Expr::Sym(_) => Lin::atom(e.clone()),
        Expr::Add(v) => {
            let mut acc = Lin::default();
            for x in v {
                acc = acc.add(linearise(x)?)?;
            }
            acc
        }
        Expr::Mul(v) => {
            // Linear only while at most one factor is non-constant.
            let mut scalar = Rat::ONE;
            let mut non_const: Option<Lin> = None;
            for x in v {
                let l = linearise(x)?;
                if l.terms.is_empty() {
                    scalar = scalar.mul(l.konst)?;
                } else if non_const.is_none() {
                    non_const = Some(l);
                } else {
                    // symbol * symbol: not linear, keep it whole.
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
            ensure!(
                rhs.terms.is_empty(),
                "cannot translate a shape expression that divides by a symbol: {e:?}"
            );
            linearise(a)?.scale(Rat::ONE.div(rhs.konst)?)?
        }
        // Opaque: correct, just not linear. It still combines with itself.
        Expr::Floor(_) | Expr::Min(_) | Expr::Max(_) | Expr::Broadcast(_) => Lin::atom(e.clone()),
    })
}

// ---------------------------------------------------------------------------
// Translation to TDim
// ---------------------------------------------------------------------------

/// Turn a rational linear form into a `TDim`, given a translation for each atom.
///
/// The form is reduced to a single fraction first, so the result is
/// `Div(integer_expression, denominator)` — and when the denominator is 1 there
/// is no division at all.
fn lin_to_tdim(lin: &Lin, scope: &SymbolScope) -> TractResult<TDim> {
    let (scaled, den) = lin.as_single_fraction()?;
    let mut sum: TDim = scaled.konst.num.to_dim();
    for (c, atom) in &scaled.terms {
        let a = atom_to_tdim(atom, scope)?;
        sum += a * c.num;
    }
    ensure!(den > 0, "non-positive denominator in a shape expression");
    if den == 1 {
        Ok(sum)
    } else {
        // ⚠️ THIS is where the two algebras meet, and it is sound only because
        // the floor has already been reduced onto a single fraction. tract's
        // `/` truncates toward zero rather than flooring; the two agree for
        // non-negative values, and a tensor dimension cannot be negative.
        Ok(sum / den)
    }
}

fn atom_to_tdim(e: &Expr, scope: &SymbolScope) -> TractResult<TDim> {
    Ok(match e {
        Expr::Sym(name) => TDim::Sym(scope.sym(name)),
        // The reduction has already put this floor over a single fraction, so
        // translating the argument and letting `/` truncate IS the floor.
        Expr::Floor(inner) => to_tdim_inner(inner, scope)?,
        Expr::Min(v) => {
            TDim::Min(v.iter().map(|x| to_tdim_inner(x, scope)).collect::<TractResult<_>>()?)
        }
        Expr::Max(v) => {
            TDim::Max(v.iter().map(|x| to_tdim_inner(x, scope)).collect::<TractResult<_>>()?)
        }
        Expr::Broadcast(v) => {
            TDim::Broadcast(v.iter().map(|x| to_tdim_inner(x, scope)).collect::<TractResult<_>>()?)
        }
        other => bail!("cannot translate shape sub-expression {other:?}"),
    })
}

fn to_tdim_inner(e: &Expr, scope: &SymbolScope) -> TractResult<TDim> {
    lin_to_tdim(&linearise(e)?, scope)
}

/// Parse an ONNX `dim_param` and translate it into a `TDim`.
///
/// Returns `Err` with a message naming what could not be translated, rather
/// than a shape that is quietly wrong. Callers decide whether an untranslatable
/// dimension is fatal or should fall back to "unknown, let tract infer it".
pub fn parse_onnx_dim(scope: &SymbolScope, s: &str) -> TractResult<TDim> {
    let mut p = Parser::new(s);
    let e = p.expr().with_context(|| format!("parsing ONNX dim_param {s:?}"))?;
    p.ws();
    ensure!(p.i == p.s.len(), "trailing {:?} in ONNX dim_param {s:?}", p.rest());
    let e = flatten_nested_floor(e);
    to_tdim_inner(&e, scope).with_context(|| format!("translating ONNX dim_param {s:?}"))
}

/// Evaluate a parsed expression over concrete symbol values, in EXACT rational
/// arithmetic with real flooring.
///
/// This is what the tests check the `TDim` translation against: it is the ONNX
/// reading of the expression, computed without ever touching tract's integer
/// division, so agreement between the two is evidence and not a tautology.
#[cfg(test)]
use std::collections::HashMap;

#[cfg(test)]
fn eval_rational(e: &Expr, vals: &HashMap<String, i64>) -> TractResult<Rat> {
    Ok(match e {
        Expr::Rat(r) => *r,
        Expr::Sym(n) => Rat::int(*vals.get(n).with_context(|| format!("no value for {n}"))?),
        Expr::Add(v) => {
            let mut acc = Rat::ZERO;
            for x in v {
                acc = acc.add(eval_rational(x, vals)?)?;
            }
            acc
        }
        Expr::Mul(v) => {
            let mut acc = Rat::ONE;
            for x in v {
                acc = acc.mul(eval_rational(x, vals)?)?;
            }
            acc
        }
        Expr::Div(a, b) => eval_rational(a, vals)?.div(eval_rational(b, vals)?)?,
        Expr::Floor(x) => {
            let r = eval_rational(x, vals)?;
            Rat::int(r.num.div_euclid(r.den))
        }
        Expr::Min(v) => {
            let mut it = v.iter().map(|x| eval_rational(x, vals));
            let mut acc = it.next().context("min() with no arguments")??;
            for x in it {
                let x = x?;
                if x.num * acc.den < acc.num * x.den {
                    acc = x;
                }
            }
            acc
        }
        Expr::Max(v) | Expr::Broadcast(v) => {
            let mut it = v.iter().map(|x| eval_rational(x, vals));
            let mut acc = it.next().context("max() with no arguments")??;
            for x in it {
                let x = x?;
                if x.num * acc.den > acc.num * x.den {
                    acc = x;
                }
            }
            acc
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn scope() -> SymbolScope {
        SymbolScope::default()
    }

    /// ⚠️ The scope MUST outlive the formatting. A `Symbol` holds a Weak
    /// reference to its scope, so a TDim formatted after its scope has dropped
    /// prints `<Sym0>` instead of the name — which looks like a translation bug
    /// and is not one.
    fn show(s: &str) -> String {
        let scope = scope();
        parse_onnx_dim(&scope, s).unwrap().to_string()
    }

    fn tdim(s: &str) -> TractResult<TDim> {
        parse_onnx_dim(&scope(), s)
    }

    /// The whole error chain, not just the outermost context. `to_string()` on
    /// an anyhow error shows only the top frame, which here is always
    /// "translating ONNX dim_param ..." and never the actual reason.
    fn why(s: &str) -> String {
        format!("{:?}", tdim(s).unwrap_err())
    }

    /// Evaluate the ONNX reading (exact rationals, real floor) and the
    /// translated TDim (tract's integer arithmetic) at the same point, and
    /// require they agree.
    fn agree_at(src: &str, sym: &str, values: &[i64]) {
        let scope = scope();
        let dim = parse_onnx_dim(&scope, src).unwrap();
        let mut p = Parser::new(src);
        let expr = flatten_nested_floor(p.expr().unwrap());
        for &v in values {
            let mut vals = HashMap::new();
            vals.insert(sym.to_string(), v);
            let want = eval_rational(&expr, &vals).unwrap();
            assert!(want.is_int(), "{src} at {sym}={v} is not integral: {want:?}");
            let got = dim.eval(&SymbolValues::default().with(&scope.sym(sym), v)).to_i64().unwrap();
            assert_eq!(
                got, want.num,
                "{src} at {sym}={v}: tdim {dim} gave {got}, ONNX says {}",
                want.num
            );
        }
    }

    // ---- the bug this module exists for -----------------------------------

    #[test]
    fn half_is_a_half_not_zero() {
        // The whole issue in one line: integer division made this `W/2 + 1`.
        agree_at("floor(W/2 - 1/2) + 1", "W", &[1, 2, 3, 4, 5, 6, 7, 8, 9, 318, 319, 320, 321]);
    }

    #[test]
    fn even_widths_were_the_ones_that_broke() {
        // At W=4 the old reading gave 3; the correct answer is 2.
        let scope = scope();
        let d = parse_onnx_dim(&scope, "floor(W/2 - 1/2) + 1").unwrap();
        let at =
            |v: i64| d.eval(&SymbolValues::default().with(&scope.sym("W"), v)).to_i64().unwrap();
        assert_eq!(at(4), 2);
        assert_eq!(at(6), 3);
        assert_eq!(at(320), 160);
        // Odd widths agreed even before the fix, which is why it hid so long.
        assert_eq!(at(5), 3);
        assert_eq!(at(321), 161);
    }

    #[test]
    fn nested_floor_from_two_stride_two_convolutions() {
        agree_at(
            "floor(floor(W/2 - 1/2)/2) + 1",
            "W",
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 318, 319, 320, 321],
        );
    }

    #[test]
    fn the_nested_case_collapses_to_one_division() {
        // Correctness is covered above; this pins the SHAPE of the result,
        // because a tower of divisions is what tract then fails to unify.
        assert_eq!(show("floor(floor(W/2 - 1/2)/2) + 1"), "(W+3)/4");
    }

    #[test]
    fn a_simple_fraction_keeps_its_denominator() {
        assert_eq!(show("floor(W/2 - 1/2) + 1"), "(W+1)/2");
    }

    // ---- the ordinary cases still work ------------------------------------

    #[test]
    fn plain_symbols_and_integers() {
        assert_eq!(show("N"), "N");
        assert_eq!(show("42"), "42");
        assert_eq!(show("DynamicDimension.1"), "DynamicDimension.1");
    }

    #[test]
    fn arithmetic_without_any_floor() {
        agree_at("2*W + 3", "W", &[0, 1, 7, 1000]);
        agree_at("(W - 1)*2", "W", &[1, 2, 50]);
        agree_at("W", "W", &[0, 5]);
    }

    #[test]
    fn integer_division_is_still_integer_division() {
        // No fractional literal anywhere, so this is what it always was.
        agree_at("W/2", "W", &[0, 2, 4, 100]);
    }

    #[test]
    fn min_and_max_survive() {
        assert!(tdim("min(N, 4)").is_ok());
        assert!(tdim("max(N, 4)").is_ok());
    }

    #[test]
    fn repeated_atoms_combine() {
        // Must become 2*floor(W/2), not two loose terms — and it must NOT
        // collapse to W, which is only equal for even values.
        assert_eq!(show("floor(W/2) + floor(W/2)"), "2*(W)/2");
    }

    // ---- failure is loud, not silent --------------------------------------

    #[test]
    fn dividing_by_a_symbol_is_refused() {
        let e = why("W/N");
        assert!(e.contains("divides by a symbol"), "{e}");
    }

    #[test]
    fn an_unknown_function_is_refused() {
        let e = why("ceil(W/2)");
        assert!(e.contains("unsupported function"), "{e}");
    }

    #[test]
    fn garbage_is_refused() {
        assert!(tdim("W +").is_err());
        assert!(tdim("(W").is_err());
        assert!(tdim("W)").is_err());
        assert!(tdim("").is_err());
        assert!(tdim("$").is_err());
    }

    #[test]
    fn division_by_zero_is_refused() {
        assert!(tdim("W/0").is_err());
    }

    // ---- the rational helper itself ---------------------------------------

    #[test]
    fn rationals_stay_in_lowest_terms() {
        assert_eq!(Rat::new(2, 4).unwrap(), Rat { num: 1, den: 2 });
        assert_eq!(Rat::new(-2, 4).unwrap(), Rat { num: -1, den: 2 });
        assert_eq!(Rat::new(2, -4).unwrap(), Rat { num: -1, den: 2 });
        assert_eq!(Rat::new(0, 5).unwrap(), Rat { num: 0, den: 1 });
    }

    #[test]
    fn overflow_is_reported_not_wrapped() {
        assert!(Rat::int(i64::MAX).mul(Rat::int(2)).is_err());
    }

    // ---- end to end, through the real importer ----------------------------

    /// Build the smallest possible ONNX model whose single input carries a
    /// paddle2onnx-style dimension, and load it through the real front end.
    ///
    /// The unit tests above prove the translation; this proves it is actually
    /// WIRED, which is a different claim. The bug shipped for years with a
    /// perfectly good `TDim` parser behind it, because the wrong parser was the
    /// one being called.
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
        // ⚠️ The graph needs a real node. With an empty one the declared input
        // fact never lands on a Source and every shape reads back as unknown,
        // which looks exactly like a broken parser and is not.
        let mut out_info = value_info.clone();
        out_info.name = "y".to_string();
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
    fn a_plain_symbol_survives_the_round_trip() {
        // The control for the two below: if this fails, the harness is wrong
        // rather than the translation.
        assert!(import_dim_param("W").contains('W'));
    }

    #[test]
    fn the_importer_uses_the_rational_parser() {
        // Before: "W/2+1", one too big for every even W. After: "(W+1)/2".
        let got = import_dim_param("floor(W/2 - 1/2) + 1");
        assert!(got.contains("(W+1)/2"), "imported shape was {got}");
        assert!(!got.contains("W/2+1"), "still the old integer reading: {got}");
    }

    #[test]
    fn the_importer_handles_the_nested_case() {
        let got = import_dim_param("floor(floor(W/2 - 1/2)/2) + 1");
        assert!(got.contains("(W+3)/4"), "imported shape was {got}");
    }

    #[test]
    fn an_untranslatable_dim_param_still_imports_as_unknown() {
        // torch dynamo emits sympy that this parser does not accept. Those
        // models loaded before and must keep loading: the dimension becomes
        // unknown and tract infers it from the graph.
        let got = import_dim_param("Mod(s0, 8)");
        assert!(!got.contains("Mod"), "an untranslatable dim leaked into the fact: {got}");
    }
}
