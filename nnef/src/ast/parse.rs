use nom_language::error::{convert_error, VerboseError};
use tract_core::internal::*;

use nom::branch::alt;
use nom::combinator::map;
use nom::{bytes::complete::*, character::complete::*, combinator::*, multi::*, sequence::*};
use nom::{Finish, IResult, Parser};

use crate::ast::*;

type R<'i, O> = IResult<&'i str, O, VerboseError<&'i str>>;

pub(super) fn translate_error(e: nom::Err<VerboseError<&str>>) -> TractError {
    format_err!("{}", e)
}

#[inline(never)]
pub fn unwrap_parse<'s, P, O>(input: &'s str, parser: P) -> TractResult<O>
where
    P: Parser<&'s str, Output = O, Error = VerboseError<&'s str>>,
{
    all_consuming(parser)
        .parse(input)
        .finish()
        .map(|(_, p)| p)
        .map_err(|e| anyhow!(convert_error(input, e)))
}

pub fn parse_document(doc: &str) -> TractResult<Document> {
    unwrap_parse(doc, document)
}

#[inline(never)]
pub fn parse_fragments(doc: &str) -> TractResult<Vec<FragmentDef>> {
    unwrap_parse(doc, fragments)
}

#[inline(never)]
pub fn parse_fragment_decl(doc: &str) -> TractResult<FragmentDecl> {
    unwrap_parse(doc, fragment_decl)
}

#[inline(never)]
pub fn parse_parameters(doc: &str) -> TractResult<Vec<Parameter>> {
    unwrap_parse(doc, parameter_list)
}

// <document> ::= <version> <extension>* <fragmentdefinition>* <graph-definition>
fn document(i: &str) -> R<'_, Document> {
    map(
        (version, many0(extension), fragments, graph_def),
        |(version, extension, fragments, graph_def)| Document {
            version,
            extension,
            fragments,
            graph_def,
        },
    )
    .parse(i)
}

fn fragments(i: &str) -> R<'_, Vec<FragmentDef>> {
    many0(fragment_def).parse(i)
}

// <version> ::= "version" <numeric-literal> ";"

fn version(i: &str) -> R<'_, NumericLiteral> {
    preceded(stag("version"), cut(terminated(numeric_literal, stag(";")))).parse(i)
}

// NNEF spec: <extension> ::= "extension" <identifier>+ ";"
// tract accepts: <extension> ::= "extension" <identifier> <anything-but-;>";"
fn extension(i: &str) -> R<'_, (Identifier, String)> {
    delimited(
        stag("extension"),
        pair(spaced(identifier), map(take_until(";"), |s: &str| s.to_string())),
        stag(";"),
    )
    .parse(i)
}

// FRAGMENT

// <fragment-definition> ::= <fragment-declaration> (<body> | ";")
fn fragment_def(i: &str) -> R<'_, FragmentDef> {
    spaced(map(
        pair(fragment_decl, alt((map(body, Some), map(stag(";"), |_| None)))),
        |(decl, body)| FragmentDef { decl, body },
    ))
    .parse(i)
}

// <fragment-declaration> ::= "fragment" <identifier> [<generic-declaration>] "(" <parameter-list> ")" "->" "(" <result-list> ")"
fn fragment_decl(i: &str) -> R<'_, FragmentDecl> {
    let (i, _) = stag("fragment").parse(i)?;
    let (i, id) = identifier(i)?;
    let (i, generic_decl) = opt(generic_decl).parse(i)?;
    let (i, _) = stag("(").parse(i)?;
    let (i, parameters) = parameter_list(i)?;
    let (i, _) = stag(")").parse(i)?;
    let (i, _) = stag("->").parse(i)?;
    let (i, _) = stag("(").parse(i)?;
    let (i, results) = result_list(i)?;
    let (i, _) = stag(")").parse(i)?;
    Ok((i, FragmentDecl { id, parameters, results, generic_decl }))
}

// <generic-declaration> ::= "<" "?" ["=" <type-name>] ">"
fn generic_decl(i: &str) -> R<'_, Option<TypeName>> {
    let (i, _) = stag("<").parse(i)?;
    let (i, _) = stag("?").parse(i)?;
    let (i, name) = opt(preceded(stag("="), type_name)).parse(i)?;
    let (i, _) = stag(">").parse(i)?;
    Ok((i, name))
}

// <parameter-list> ::= <parameter> ("," <parameter>)*
fn parameter_list(i: &str) -> R<'_, Vec<Parameter>> {
    separated_list0(stag(","), parameter).parse(i)
}

// <result-list> ::= <result> ("," <result>)*
fn result_list(i: &str) -> R<'_, Vec<Result_>> {
    separated_list0(stag(","), result).parse(i)
}

// <parameter> ::= <identifier> ":" <type-spec> ["=" <literal-expr>]
fn parameter(i: &str) -> R<'_, Parameter> {
    map(
        pair(
            separated_pair(identifier, stag(":"), type_spec),
            opt(preceded(stag("="), literal_expr)),
        ),
        |((id, spec), lit)| Parameter { id, spec, lit, doc: None },
    )
    .parse(i)
}

// <result> ::= <identifier> ":" <type-spec>
fn result(i: &str) -> R<'_, Result_> {
    map(separated_pair(identifier, stag(":"), type_spec), |(id, spec)| Result_ { id, spec })
        .parse(i)
}

fn literal_expr(i: &str) -> R<'_, Literal> {
    spaced(alt((
        literal,
        map(delimited(stag("["), separated_list0(stag(","), literal), stag("]")), Literal::Array),
        map(delimited(stag("("), separated_list0(stag(","), literal), stag(")")), Literal::Tuple),
    )))
    .parse(i)
}

// <type-spec> ::= <type-name> | <tensor-type-spec> | <array-type-spec> | <tuple-type-spec>
fn type_spec(i: &str) -> R<'_, TypeSpec> {
    fn non_array_type(i: &str) -> R<'_, TypeSpec> {
        alt((tuple_type_spec, map(type_name, TypeSpec::Single), tensor_type_spec)).parse(i)
    }
    alt((
        (map(terminated(non_array_type, pair(stag("["), stag("]"))), |t| {
            TypeSpec::Array(Box::new(t))
        })),
        non_array_type,
    ))
    .parse(i)
}

// <type-name> ::= "integer" | "scalar" | "logical" | "string" | "?"
fn type_name(i: &str) -> R<'_, TypeName> {
    spaced(alt((
        map(tag("integer"), |_| TypeName::Integer),
        map(tag("scalar"), |_| TypeName::Scalar),
        map(tag("logical"), |_| TypeName::Logical),
        map(tag("string"), |_| TypeName::String),
        #[cfg(feature = "complex")]
        map(tag("complex"), |_| TypeName::Complex),
        map(tag("?"), |_| TypeName::Any),
    )))
    .parse(i)
}

// <tensor-type-spec> ::= "tensor" "<" [<type-name>] ">"
fn tensor_type_spec(i: &str) -> R<'_, TypeSpec> {
    map(delimited(pair(stag("tensor"), stag("<")), type_name, stag(">")), TypeSpec::Tensor).parse(i)
}

// <tuple-type-spec> ::= "(" <type-spec> ("," <type-spec>)+ ")"
fn tuple_type_spec(i: &str) -> R<'_, TypeSpec> {
    map(delimited(stag("("), separated_list0(stag(","), type_spec), stag(")")), TypeSpec::Tuple)
        .parse(i)
}

// GRAPH

// <graph-definition> ::= <graph-declaration> <body>
// <graph-declaration> ::= "graph" <identifier> "(" <identifier-list> ")" "->" "(" <identifier-list> ")"
// <identifier-list> ::= <identifier> ("," <identifier>)*
fn graph_def(i: &str) -> R<'_, GraphDef> {
    let (i, _) = stag("graph").parse(i)?;
    let (i, id) = identifier(i)?;
    let (i, _) = stag("(").parse(i)?;
    let (i, parameters) = separated_list0(stag(","), identifier).parse(i)?;
    let (i, _) = stag(")").parse(i)?;
    let (i, _) = stag("->").parse(i)?;
    let (i, _) = stag("(").parse(i)?;
    let (i, results) = separated_list0(stag(","), identifier).parse(i)?;
    let (i, _) = stag(")").parse(i)?;
    let (i, body) = spaced(body).parse(i)?;
    Ok((i, GraphDef { id, parameters, results, body }))
}

// BODY

// <body> ::= "{" <assignment>+ "}"
fn body(i: &str) -> R<'_, Vec<Assignment>> {
    delimited(stag("{"), many0(assignment), stag("}")).parse(i)
}

// <assignment> ::= <lvalue-expr> "=" <rvalue-expr> ";"
fn assignment(i: &str) -> R<'_, Assignment> {
    spaced(terminated(
        map(separated_pair(lvalue, stag("="), rvalue), |(left, right)| Assignment { left, right }),
        stag(";"),
    ))
    .parse(i)
}

// <lvalue-expr> ::= <identifier> | <array-lvalue-expr> | <tuple-lvalue-expr>
// <array-lvalue-expr> ::= "[" [<lvalue-expr> ("," <lvalue-expr>)* ] "]"
// <tuple-lvalue-expr> ::= "(" <lvalue-expr> ("," <lvalue-expr>)+ ")" | <lvalue-expr> ("," <lvalue-expr>)+
fn lvalue(i: &str) -> R<'_, LValue> {
    fn inner_lvalue(i: &str) -> R<'_, LValue> {
        alt((
            map(
                delimited(stag("["), separated_list0(stag(","), inner_lvalue), stag("]")),
                LValue::Array,
            ),
            map(
                delimited(stag("("), separated_list0(stag(","), inner_lvalue), stag(")")),
                LValue::Tuple,
            ),
            map(spaced(identifier), LValue::Identifier),
        ))
        .parse(i)
    }

    map(separated_list0(stag(","), inner_lvalue), |mut iv| {
        if iv.len() == 1 {
            iv.remove(0)
        } else {
            LValue::Tuple(iv)
        }
    })
    .parse(i)
}

// <invocation> ::= <identifier> ["<" <type-name> ">"] "(" <argument-list> ")"
fn invocation(i: &str) -> R<'_, Invocation> {
    let (i, id) = spaced(identifier).parse(i)?;
    let (i, generic_type_name) = opt(delimited(stag("<"), type_name, stag(">"))).parse(i)?;
    let (i, _) = stag("(").parse(i)?;
    let (i, arguments) = argument_list.parse(i)?;
    let (i, _) = stag(")").parse(i)?;
    Ok((i, Invocation { id, generic_type_name, arguments }))
}

// <argument-list> ::= <argument> ("," <argument>)*
fn argument_list(i: &str) -> R<'_, Vec<Argument>> {
    separated_list0(stag(","), argument).parse(i)
}

// <argument> ::= <rvalue-expr> | <identifier> "=" <rvalue-expr>
fn argument(i: &str) -> R<'_, Argument> {
    spaced(map(pair(opt(terminated(identifier, stag("="))), rvalue), |(id, rvalue)| Argument {
        id,
        rvalue,
    }))
    .parse(i)
}

//<rvalue-expr> ::= <identifier> | <literal> | <binary-expr> | <unary-expr> | <paren-expr>
//                  | <array-rvalue-expr> | <tuple-rvalue-expr> | <subscript-expr> | <if-else-expr>
//                  | <comprehension-expr> | <builtin-expr> | <invocation>
fn rvalue(i: &str) -> R<'_, RValue> {
    fn atom(i: &str) -> R<'_, RValue> {
        spaced(alt((
            map(invocation, RValue::Invocation),
            map(literal, RValue::Literal),
            map(identifier, RValue::Identifier),
            map(pair(spaced(recognize(one_of("+-!"))), rvalue), |(op, rv)| {
                RValue::Unary(op.into(), Box::new(rv))
            }),
            map(delimited(tag("("), separated_list0(stag(","), rvalue), tag(")")), |mut rvs| {
                if rvs.len() == 1 {
                    rvs.remove(0)
                } else {
                    RValue::Tuple(rvs)
                }
            }),
            map(comprehension_expr, |c| RValue::Comprehension(Box::new(c))),
            map(delimited(tag("["), separated_list0(stag(","), rvalue), tag("]")), |rvs| {
                RValue::Array(rvs)
            }),
        )))
        .parse(i)
    }
    macro_rules! bin {
        ($name:ident, $operand: ident, $operator: expr) => {
            fn $name(i: &str) -> R<'_, RValue> {
                let (i, init) = $operand(i)?;
                fold_many0(
                    pair($operator, $operand),
                    move || init.clone(),
                    |left, (op, right)| {
                        RValue::Binary(Box::new(left), op.to_string(), Box::new(right))
                    },
                )
                .parse(i)
            }
        };
    }

    // <subscript-expr> ::= <rvalue-expr> "[" (<rvalue-expr> | [<rvalue-expr>] ":" [<rvalue-expr>]) "]"
    fn sub(i: &str) -> R<'_, RValue> {
        alt((
            map(
                pair(
                    atom,
                    delimited(
                        stag("["),
                        alt((
                            map(separated_pair(opt(rvalue), stag(":"), opt(rvalue)), |(a, b)| {
                                Subscript::Range(a, b)
                            }),
                            map(rvalue, Subscript::Single),
                        )),
                        stag("]"),
                    ),
                ),
                |(rv, range)| RValue::Subscript(Box::new(rv), Box::new(range)),
            ),
            atom,
        ))
        .parse(i)
    }

    bin!(exp, sub, tag("^"));
    bin!(mul, exp, one_of("*/"));
    bin!(add, mul, one_of("+-"));
    bin!(comp, add, alt((tag("=="), tag("!="), tag("<"), tag(">"), tag("<="), tag(">="))));
    bin!(boolean, comp, alt((tag("||"), tag("&&"))));
    bin!(in_for, boolean, tag("in"));

    // <if-else-expr> ::= <rvalue-expr> "if" <rvalue-expr> "else" <rvalue-expr>
    fn ite(i: &str) -> R<'_, RValue> {
        let (i, leftmost) = in_for(i)?;
        let (i, _) = space_and_comments(i)?;
        if i.starts_with("if") {
            let (i, _) = stag("if").parse(i)?;
            let (i, cond) = in_for(i)?;
            let (i, _) = stag("else").parse(i)?;
            let (i, otherwise) = in_for(i)?;
            Ok((i, RValue::IfThenElse(Box::new(IfThenElse { cond, then: leftmost, otherwise }))))
        } else {
            Ok((i, leftmost))
        }
    }

    ite(i)
}

// <comprehension-expr> ::= "[" "for" <loop-iter-list> ["if" <rvalue-expr>] "yield" <rvalue-expr> "]"
fn comprehension_expr(i: &str) -> R<'_, Comprehension> {
    delimited(
        pair(stag("["), stag("for")),
        map(separated_pair(loop_iters, stag("yield"), rvalue), |(loop_iters, yields)| {
            Comprehension { loop_iters, filter: None, yields }
        }),
        stag("]"),
    )
    .parse(i)
}

// <loop-iter> ::= <identifier> "in" <rvalue-expr>
// <loop-iter-list> ::= <loop-iter> ("," <loop-iter>)*
fn loop_iters(i: &str) -> R<'_, Vec<(Identifier, RValue)>> {
    separated_list0(stag(","), separated_pair(identifier, stag("in"), rvalue)).parse(i)
}

// TERMINALS

// identifier: identifiers must consist of the following ASCII characters: _, [a-z], [A-Z], [0-9].
// The identifier must not start with a digit.
pub(super) fn identifier(i: &str) -> R<'_, Identifier> {
    alt((escaped_identifier, direct_identifier)).parse(i)
}

pub(super) fn direct_identifier(i: &str) -> R<'_, Identifier> {
    map(
        recognize(pair(alt((alpha1, tag("_"))), many0(alt((alphanumeric1, tag("_")))))),
        Identifier::from,
    )
    .parse(i)
}

pub(super) fn escaped_identifier(i: &str) -> R<'_, Identifier> {
    map(preceded(tag("i"), string_literal), Identifier).parse(i)
}

// <literal> ::= <numeric-literal> | <string-literal> | <logical-literal>
fn literal(i: &str) -> R<'_, Literal> {
    spaced(alt((
        map(numeric_literal, Literal::Numeric),
        map(string_literal, Literal::String),
        map(logical_literal, Literal::Logical),
    )))
    .parse(i)
}

pub(super) fn numeric_literal(i: &str) -> R<'_, String> {
    fn exp_part(i: &str) -> R<'_, &str> {
        recognize((one_of("eE"), opt(tag("-")), digit1)).parse(i)
    }
    fn frac_part(i: &str) -> R<'_, &str> {
        recognize((tag("."), digit0)).parse(i)
    }
    spaced(map(
        recognize((opt(tag("-")), alt((digit1, tag("inf"))), opt(frac_part), opt(exp_part))),
        |s: &str| s.to_owned(),
    ))
    .parse(i)
}

fn string_literal(i: &str) -> R<'_, String> {
    fn inner(i: &str) -> R<'_, String> {
        map(
            many0(alt((
                preceded(tag("\\"), nom::character::complete::anychar),
                nom::character::complete::none_of("\\\"'"),
            ))),
            |v: Vec<char>| v.into_iter().collect(),
        )
        .parse(i)
    }
    map(alt((delimited(tag("'"), inner, tag("'")), delimited(tag("\""), inner, tag("\"")))), |s| s)
        .parse(i)
}

pub(super) fn logical_literal(i: &str) -> R<'_, bool> {
    spaced(alt((map(tag("true"), |_| true), map(tag("false"), |_| false)))).parse(i)
}

// SPACES

fn space_and_comments(i: &str) -> R<'_, ()> {
    map(
        many0(alt((recognize(one_of(" \t\n\r")), recognize((tag("#"), many0(none_of("\r\n"))))))),
        |_| (),
    )
    .parse(i)
}

fn spaced<'s, O, F>(it: F) -> impl Parser<&'s str, Output = O, Error = VerboseError<&'s str>>
where
    F: Parser<&'s str, Output = O, Error = VerboseError<&'s str>>,
{
    delimited(space_and_comments, it, space_and_comments)
}

pub(super) fn stag<'s>(
    t: &'static str,
) -> impl Parser<&'s str, Output = &'s str, Error = VerboseError<&'s str>> {
    spaced(tag(t))
}

#[cfg(test)]
mod test {
    use super::*;
    use TypeName::*;
    use TypeSpec::*;

    fn p<'s, P, O, E>(parser: P, i: &'s str) -> O
    where
        O: std::fmt::Debug,
        P: Fn(&'s str) -> IResult<&'s str, O, E>,
        E: nom::error::ParseError<&'s str> + std::fmt::Debug,
    {
        let res = all_consuming(parser).parse(i).unwrap();
        res.1
    }

    fn param(s: impl Into<std::string::String>, t: TypeSpec) -> Parameter {
        Parameter { id: Identifier(s.into()), spec: t, lit: None, doc: None }
    }

    fn result(s: impl Into<std::string::String>, t: TypeSpec) -> Result_ {
        Result_ { id: Identifier(s.into()), spec: t }
    }

    #[test]
    fn test_type_spec() {
        assert_eq!(p(type_spec, "scalar"), Single(Scalar));
        assert_eq!(p(type_spec, "scalar[]"), Array(Box::new(Single(Scalar))));
        assert_eq!(p(type_spec, "tensor<scalar>[]"), Array(Box::new(Tensor(TypeName::Scalar))));
        assert_eq!(
            p(type_spec, "(scalar,scalar[],tensor<scalar>)"),
            Tuple(vec!(Single(Scalar), Array(Box::new(Single(Scalar))), Tensor(Scalar)))
        );
        assert_eq!(p(type_spec, "tensor<?>[]"), Array(Box::new(Tensor(TypeName::Any))));
        assert_eq!(p(type_spec, "scalar[ ]"), Array(Box::new(Single(Scalar))));
        assert_eq!(
            p(type_spec, " ( scalar , scalar [ ] , tensor < scalar > ) "),
            Tuple(vec!(Single(Scalar), Array(Box::new(Single(Scalar))), Tensor(Scalar)))
        );
        #[cfg(feature = "complex")]
        assert_eq!(p(type_spec, "tensor<complex>[]"), Array(Box::new(Tensor(TypeName::Complex))));
    }

    #[test]
    fn test_fragment_decl_fizz() {
        let parsed = p(
            fragment_decl,
            "fragment fizz<? = scalar>( shape: integer[] ) -> ( output: tensor<?> )",
        );
        assert_eq!(
            parsed,
            FragmentDecl {
                id: "fizz".into(),
                generic_decl: Some(Some(Scalar)),
                parameters: vec!(param("shape", Array(Box::new(Single(Integer)))),),
                results: vec!(result("output", Tensor(Any))),
            }
        );
    }

    #[test]
    fn test_fragment_decl_logarithmic_quantize() {
        let parsed = p(fragment_decl,
                           "fragment logarithmic_quantize(x: tensor<scalar>, max: tensor<scalar>, bits: integer ) -> ( y: tensor<scalar> )"
                          );
        assert_eq!(
            parsed,
            FragmentDecl {
                id: "logarithmic_quantize".into(),
                generic_decl: None,
                parameters: vec!(
                    param("x", Tensor(Scalar)),
                    param("max", Tensor(Scalar)),
                    param("bits", Single(Integer))
                ),
                results: vec!(result("y", Tensor(Scalar))),
            }
        );
    }

    #[test]
    fn test_fragment_decl_external() {
        p(
            fragment_decl,
            "fragment external<? = scalar>( shape: integer[] ) -> ( output: tensor<?> )",
        );
    }

    #[test]
    fn test_fragment_reshape() {
        p(fragments, "fragment reshape<?>( input: tensor<?>, shape: integer[], axis_start: integer = 0, axis_count: integer = -1 ) -> ( output: tensor<?> );");
    }

    #[test]
    fn test_fragment_conv() {
        p(
            fragments,
            r#"
            fragment conv(
                input: tensor<scalar>,
                filter: tensor<scalar>,
                bias: tensor<scalar> = 0.0,
                border: string = 'constant',
                padding: (integer,integer)[] = [],
                stride: integer[] = [],
                dilation: integer[] = [],
                groups: integer = 1 )
            -> ( output: tensor<scalar> );
            "#,
        );
    }

    #[test]
    fn test_fragment_local_response_normalization() {
        p(
            fragments,
            r#"
            fragment local_response_normalization(
                input: tensor<scalar>,
                size: integer[],
                alpha: scalar = 1.0,
                beta: scalar = 0.5,
                bias: scalar = 1.0 )
            -> ( output: tensor<scalar> )
            {
                sigma = bias + alpha * box(sqr(input), size = size, normalize = true);
                output = input / (sigma ^ beta);
            }
            "#,
        );
    }

    #[test]
    fn test_batch_normalization() {
        p(
            fragments,
            r#"
            fragment batch_normalization( input: tensor<scalar>, mean: tensor<scalar>, variance: tensor<scalar>, offset: tensor<scalar>, scale: tensor<scalar>, epsilon: scalar )
            -> ( output: tensor<scalar> )
            {
                output = offset + scale * (input - mean) / sqrt(variance + epsilon);
            }
            "#,
        );
    }

    #[test]
    fn test_avg_roi_align() {
        p(
            fragments,
            r#"
                fragment avg_roi_align(
                    input: tensor<scalar>,
                    rois: tensor<scalar>,
                    batch_index: tensor<integer>,
                    output_size: integer[],
                    sampling_rate: integer[],
                    resize_method: string = 'symmetric' )
                -> ( output: tensor<scalar> )
                {
                    size = [for i in range_of(output_size) yield output_size[i] * sampling_rate[i]];
                    resized = roi_resample(input, rois, batch_index, output_size = size,
                                         method = resize_method);
                    output = avg_pool(resized, size = sampling_rate, stride = sampling_rate);
                }
            "#,
        );
    }

    #[test]
    fn test_min_max_linear_quantize() {
        p(
            fragments,
            r#"
                fragment min_max_linear_quantize(
                    x: tensor<scalar>,
                    min: tensor<scalar>,
                    max: tensor<scalar>,
                    bits: integer,
                    signed: logical,
                    symmetric: logical )
                -> ( y: tensor<scalar> )
                {
                    r = scalar(2 ^ bits - 1 - integer(signed && symmetric));
                    z = clamp(x, min, max);
                    p = scalar(2 ^ (bits - 1) - integer(symmetric) if signed else 0);
                    q = round((z - min) / (max - min) * r) - p;
                    y = (q + p) / r * (max - min) + min;
}
            "#,
        );
    }

    #[test]
    fn test_numeric() {
        p(numeric_literal, "12.0");
    }

    #[test]
    fn test_string() {
        assert_eq!(p(string_literal, r#""""#), "");
        assert_eq!(p(string_literal, r#""foo""#), "foo");
        assert_eq!(p(string_literal, r#"''"#), "");
        assert_eq!(p(string_literal, r#"'foo'"#), "foo");

        assert_eq!(p(string_literal, r"'f\oo'"), "foo");
        assert_eq!(p(string_literal, r"'f\'oo'"), "f'oo");
        assert_eq!(p(string_literal, r#"'f\"oo'"#), "f\"oo");
    }

    #[test]
    fn test_identifier() {
        p(identifier, "foo");
        assert!(identifier("1").is_err());
        assert!(identifier("1foo").is_err());
    }

    #[test]
    fn test_spacing() {
        p(space_and_comments, "");
        p(space_and_comments, "\n");
        p(space_and_comments, "#comment\n");
        p(space_and_comments, "#boum");
    }

    #[test]
    fn test_spaced() {
        assert!(spaced(identifier).parse("foo").is_ok());
        assert!(spaced(identifier).parse(" foo ").is_ok());
        assert!(many1(spaced(identifier)).parse(" foo bar ").is_ok());
        assert_eq!(
            many1(spaced(identifier)).parse(" foo bar\n").unwrap().1,
            &[Identifier("foo".to_string()), Identifier("bar".to_string())]
        );
        assert_eq!(
            many1(spaced(identifier)).parse(" foo # bar\n").unwrap().1,
            &[Identifier("foo".to_string())]
        );
        assert_eq!(
            many1(spaced(identifier)).parse(" foo # bar\nbaz").unwrap().1,
            &[Identifier("foo".to_string()), Identifier("baz".to_string())]
        );
    }

    #[test]
    fn test_document() {
        assert!(document("version 1.0; graph foo() -> () {}").is_ok());
    }

    #[test]
    fn test_version() {
        p(version, "version 1.0;");
    }

    #[test]
    fn test_body() {
        p(body, "{}");
        p(body, "{foo=bar;}");
    }

    #[test]
    fn test_lvalue() {
        p(lvalue, "foo");
        p(lvalue, "foo,bar");
        p(lvalue, "foo , bar");
        p(lvalue, "(foo,bar)");
    }

    #[test]
    fn test_graph_def() {
        p(graph_def, "graph foo() -> () {}");
    }

    #[test]
    fn test_assignment() {
        p(assignment, "input = external(12);");
        p(assignment, "input = external(shape = [1, 3, 224, 224]);");
        p(assignment, "sigma = bias + alpha * box(sqr(input), size = size, normalize = true);");
        p(assignment, "output = offset + scale * (input - mean) / sqrt(variance + epsilon);");
        p(
            assignment,
            "size = [for i in range_of(output_size) yield output_size[i] * sampling_rate[i]];",
        );
        p(assignment, "r = scalar(2 ^ bits - 1 - integer(signed && symmetric));");
        p(assignment, "output, index = max_pool_with_index(input, size = size, border = border, padding = padding, stride = stride, dilation = dilation);");
    }

    #[test]
    fn test_invocation() {
        p(invocation, "external(12)");
        p(invocation, "sqrt(var + eps)");
    }

    #[test]
    fn test_arguments() {
        p(argument, "2");
        p(argument, "12");
        p(argument, "shape = [1, 3, 224, 224]");
    }

    #[test]
    fn test_rvalue() {
        p(rvalue, "12");
        p(rvalue, "(0, 0)");
        p(rvalue, "x ^ 2.0");
        p(rvalue, "1+2");
        p(rvalue, "1+sqrt(var)");
        p(rvalue, "1+sqrt(var+eps)");
        p(rvalue, "1 + sqrt(var + eps)");
        p(rvalue, "[for i in range_of(output_size) yield output_size[i] * sampling_rate[i]]");
        p(rvalue, "scalar(2 ^ (bits - 1) - integer(symmetric) if signed else 0)");
    }

    #[test]
    fn test_comprehenion() {
        p(comprehension_expr, "[for i in range_of(output_size) yield output_size * sampling_rate]");
    }

    #[test]
    fn test_freeze() {
        p(
            document,
            r#"
version 1.0;

graph y( x, s, bias ) -> ( y ) {
  x = external<scalar>(shape = [1, 2, 1, 3]);
  s = external<scalar>(shape = [2]);
  bias = external<scalar>(shape = [2]);
  y = add(
        mul(
            mul(
                sub(
                    x,
                    mul(
                        0.33333334,
                        sum_reduce(
                            x,
                            axes = [0, 2, 3]
                        )
                    )
                ),
                rsqrt(
                    add(
                        0.00001,
                        mul(
                            0.33333334,
                            sum_reduce(
                                square(
                                    sub(
                                        x,
                                        mul(
                                            0.33333334,
                                            sum_reduce(
                                                x,
                                                axes = [0, 2, 3]
                                            )
                                        )
                                    )
                                ),
                                axes = [0, 2, 3]
                            )
                        )
                    )
                )
            ),
            unsqueeze(
                unsqueeze(
                    unsqueeze(
                        s,
                        axes = [0]
                    ),
                axes = [2]
                ),
            axes = [2]
            )
        ),
        unsqueeze(
            unsqueeze(
                unsqueeze(
                    bias,
                    axes = [0]
                ),
                axes = [2]
            ),
            axes = [2]
        )
    );
}

"#,
        );
    }

    #[test]
    fn test_fragments() {
        p(
            fragments,
            r#"
            fragment add( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<scalar> );
            fragment sub( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<scalar> );
            "#,
        );
    }
}
