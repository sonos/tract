use nom::branch::alt;
use nom::combinator::map;
use nom::IResult;
use nom::{bytes::complete::*, character::complete::*, combinator::*, multi::*, sequence::*};

use crate::ast::*;

// <document> ::= <version> <extension>* <graph-definition>
pub fn document(i: &str) -> IResult<&str, Document> {
    map(tuple((version, many0(extension), graph_def)), |(version, extension, graph_def)| Document {
        version,
        extension,
        graph_def,
    })(i)
}

// <version> ::= "version" <numeric-literal> ";"
pub fn version(i: &str) -> IResult<&str, NumericLiteral> {
    delimited(spaced(tag("version")), numeric_literal, spaced(tag(";")))(i)
}

// <extension> ::= "extension" <identifier>+ ";"
pub fn extension(i: &str) -> IResult<&str, Vec<String>> {
    delimited(spaced(tag("extension")), many1(spaced(identifier)), spaced(tag(";")))(i)
}

// FRAGMENT DECLARATION

// <fragment-declaration> ::= "fragment" <identifier> [<generic-declaration>] "(" <parameter-list> ")" "->" "(" <result-list> ")"
pub fn fragment_decl(i: &str) -> IResult<&str, FragmentDecl> {
    let (i, _) = spaced(tag("fragment"))(i)?;
    let (i, id) = identifier(i)?;
    let (i, generic_decl) = opt(generic_decl)(i)?;
    let (i, _) = spaced(tag("("))(i)?;
    let (i, parameters) = parameter_list(i)?;
    let (i, _) = spaced(tag(")"))(i)?;
    let (i, _) = spaced(tag("->"))(i)?;
    let (i, _) = spaced(tag("("))(i)?;
    let (i, results) = result_list(i)?;
    let (i, _) = spaced(tag(")"))(i)?;
    Ok((i, FragmentDecl { id, parameters, results, generic_decl }))
}

// <generic-declaration> ::= "<" "?" ["=" <type-name>] ">"
fn generic_decl(i: &str) -> IResult<&str, Option<TypeName>> {
    let (i, _) = spaced(tag("<"))(i)?;
    let (i, _) = spaced(tag("?"))(i)?;
    let (i, name) = opt(preceded(spaced(tag("=")), type_name))(i)?;
    let (i, _) = spaced(tag(">"))(i)?;
    Ok((i, name))
}

// <parameter-list> ::= <parameter> ("," <parameter>)*
pub fn parameter_list(i: &str) -> IResult<&str, Vec<Parameter>> {
    separated_list(spaced(tag(",")), parameter)(i)
}

// <result-list> ::= <result> ("," <result>)*
pub fn result_list(i: &str) -> IResult<&str, Vec<Result_>> {
    separated_list(spaced(tag(",")), result)(i)
}

// <parameter> ::= <identifier> ":" <type-spec> ["=" <literal-expr>]
pub fn parameter(i: &str) -> IResult<&str, Parameter> {
    map(separated_pair(identifier, spaced(tag(":")), type_spec), |(id, spec)| Parameter {
        id,
        spec,
    })(i)
}

// <result> ::= <identifier> ":" <type-spec>
pub fn result(i: &str) -> IResult<&str, Result_> {
    map(separated_pair(identifier, spaced(tag(":")), type_spec), |(id, spec)| Result_ { id, spec })(
        i,
    )
}

// <type-spec> ::= <type-name> | <tensor-type-spec> | <array-type-spec> | <tuple-type-spec>
pub fn type_spec(i: &str) -> IResult<&str, TypeSpec> {
    pub fn non_array_type(i: &str) -> IResult<&str, TypeSpec> {
        alt((tuple_type_spec, map(type_name, TypeSpec::Single), tensor_type_spec))(i)
    }
    alt((
        (map(terminated(non_array_type, pair(spaced(tag("[")), spaced(tag("]")))), |t| {
            TypeSpec::Array(Box::new(t))
        })),
        non_array_type,
    ))(i)
}

// <type-name> ::= "integer" | "scalar" | "logical" | "string" | "?"
pub fn type_name(i: &str) -> IResult<&str, TypeName> {
    spaced(alt((
        map(tag("integer"), |_| TypeName::Integer),
        map(tag("scalar"), |_| TypeName::Scalar),
        map(tag("logical"), |_| TypeName::Logical),
        map(tag("string"), |_| TypeName::String),
        map(tag("?"), |_| TypeName::Any),
    )))(i)
}

// <tensor-type-spec> ::= "tensor" "<" [<type-name>] ">"
pub fn tensor_type_spec(i: &str) -> IResult<&str, TypeSpec> {
    map(
        delimited(pair(spaced(tag("tensor")), spaced(tag("<"))), type_name, spaced(tag(">"))),
        TypeSpec::Tensor,
    )(i)
}

// <tuple-type-spec> ::= "(" <type-spec> ("," <type-spec>)+ ")"
pub fn tuple_type_spec(i: &str) -> IResult<&str, TypeSpec> {
    map(
        delimited(spaced(tag("(")), separated_list(spaced(tag(",")), type_spec), spaced(tag(")"))),
        TypeSpec::Tuple,
    )(i)
}

// GRAPH

// <graph-definition> ::= <graph-declaration> <body>
// <graph-declaration> ::= "graph" <identifier> "(" <identifier-list> ")" "->" "(" <identifier-list> ")"
// <identifier-list> ::= <identifier> ("," <identifier>)*
pub fn graph_def(i: &str) -> IResult<&str, GraphDef> {
    let (i, _) = spaced(tag("graph"))(i)?;
    let (i, id) = identifier(i)?;
    let (i, _) = spaced(tag("("))(i)?;
    let (i, parameters) = separated_list(spaced(tag(",")), identifier)(i)?;
    let (i, _) = spaced(tag(")"))(i)?;
    let (i, _) = spaced(tag("->"))(i)?;
    let (i, _) = spaced(tag("("))(i)?;
    let (i, results) = separated_list(spaced(tag(",")), identifier)(i)?;
    let (i, _) = spaced(tag(")"))(i)?;
    let (i, body) = spaced(body)(i)?;
    Ok((i, GraphDef { id, parameters, results, body }))
}

// BODY

// <body> ::= "{" <assignment>+ "}"
pub fn body(i: &str) -> IResult<&str, Vec<Assignment>> {
    delimited(spaced(tag("{")), many0(assignment), spaced(tag("}")))(i)
}

// <assignment> ::= <lvalue-expr> "=" <rvalue-expr> ";"
pub fn assignment(i: &str) -> IResult<&str, Assignment> {
    spaced(terminated(
        map(separated_pair(lvalue, spaced(tag("=")), rvalue), |(left, right)| Assignment {
            left,
            right,
        }),
        spaced(tag(";")),
    ))(i)
}

// <lvalue-expr> ::= <identifier> | <array-lvalue-expr> | <tuple-lvalue-expr>
// <array-lvalue-expr> ::= "[" [<lvalue-expr> ("," <lvalue-expr>)* ] "]"
// <tuple-lvalue-expr> ::= "(" <lvalue-expr> ("," <lvalue-expr>)+ ")" | <lvalue-expr> ("," <lvalue-expr>)+
pub fn lvalue(i: &str) -> IResult<&str, LValue> {
    pub fn inner_lvalue(i: &str) -> IResult<&str, LValue> {
        alt((
            map(
                delimited(
                    spaced(tag("[")),
                    separated_list(spaced(tag(",")), inner_lvalue),
                    spaced(tag("]")),
                ),
                LValue::Array,
            ),
            map(
                delimited(
                    spaced(tag("(")),
                    separated_list(spaced(tag(",")), inner_lvalue),
                    spaced(tag(")")),
                ),
                LValue::Tuple,
            ),
            map(spaced(identifier), LValue::Identifier),
        ))(i)
    }

    map(separated_list(spaced(tag(",")), inner_lvalue), LValue::Tuple)(i)
}

// <invocation> ::= <identifier> ["<" <type-name> ">"] "(" <argument-list> ")"
pub fn invocation(i: &str) -> IResult<&str, Invocation> {
    let (i, id) = spaced(identifier)(i)?;
    let (i, generic_type_name) = opt(delimited(spaced(tag("<")), type_name, spaced(tag(">"))))(i)?;
    let (i, _) = spaced(tag("("))(i)?;
    let (i, arguments) = argument_list(i)?;
    let (i, _) = spaced(tag(")"))(i)?;
    Ok((i, Invocation { id, generic_type_name, arguments }))
}

// <argument-list> ::= <argument> ("," <argument>)*
pub fn argument_list(i: &str) -> IResult<&str, Vec<Argument>> {
    separated_list(spaced(tag(",")), argument)(i)
}

// <argument> ::= <rvalue-expr> | <identifier> "=" <rvalue-expr>
pub fn argument(i: &str) -> IResult<&str, Argument> {
    spaced(map(pair(opt(terminated(identifier, spaced(tag("=")))), rvalue), |(id, rvalue)| {
        Argument { id, rvalue }
    }))(i)
}

//<rvalue-expr> ::= <identifier> | <literal> | <binary-expr> | <unary-expr> | <paren-expr>
//                  | <array-rvalue-expr> | <tuple-rvalue-expr> | <subscript-expr> | <if-else-expr>
//                  | <comprehension-expr> | <builtin-expr> | <invocation>
pub fn rvalue(i: &str) -> IResult<&str, RValue> {
    spaced(alt((
        map(invocation, RValue::Invocation),
        map(identifier, RValue::Identifier),
        map(literal, RValue::Literal),
        map(delimited(tag("("), separated_list(spaced(tag(",")), rvalue), tag(")")), |mut rvs| {
            if rvs.len() == 1 {
                rvs.remove(0)
            } else {
                RValue::Tuple(rvs)
            }
        }),
        map(delimited(tag("["), separated_list(spaced(tag(",")), rvalue), tag("]")), |rvs| {
            RValue::Array(rvs)
        }),
    )))(i)
}

// TERMINALS

// identifier: identifiers must consist of the following ASCII characters: _, [a-z], [A-Z], [0-9].
// The identifier must not start with a digit.
pub fn identifier(i: &str) -> IResult<&str, String> {
    map(
        recognize(pair(alpha1, nom::multi::many0(nom::branch::alt((alphanumeric1, tag("_")))))),
        String::from,
    )(i)
}

// <literal> ::= <numeric-literal> | <string-literal> | <logical-literal>
pub fn literal(i: &str) -> IResult<&str, Literal> {
    spaced(alt((
        map(numeric_literal, Literal::Numeric),
        map(string_literal, Literal::String),
        map(logical_literal, Literal::Logical),
    )))(i)
}

pub fn numeric_literal(i: &str) -> IResult<&str, NumericLiteral> {
    fn exp_part(i: &str) -> IResult<&str, &str> {
        recognize(tuple((one_of("eE"), opt(tag("-")), digit1)))(i)
    }
    fn frac_part(i: &str) -> IResult<&str, &str> {
        recognize(tuple((tag("."), digit0)))(i)
    }
    spaced(map(recognize(tuple((digit1, opt(frac_part), opt(exp_part)))), |s: &str| {
        NumericLiteral(s.to_owned())
    }))(i)
}

pub fn string_literal(i: &str) -> IResult<&str, StringLiteral> {
    pub fn inner(i: &str) -> IResult<&str, String> {
        map(
            many0(alt((
                preceded(tag("\\"), nom::character::complete::anychar),
                nom::character::complete::none_of("\\\"'"),
            ))),
            |v: Vec<char>| v.into_iter().collect(),
        )(i)
    }
    map(alt((delimited(tag("'"), inner, tag("'")), delimited(tag("\""), inner, tag("\"")))), |s| {
        StringLiteral(s.into())
    })(i)
}

pub fn logical_literal(i: &str) -> IResult<&str, LogicalLiteral> {
    spaced(alt((
        map(tag("true"), |_| LogicalLiteral(true)),
        map(tag("false"), |_| LogicalLiteral(false)),
    )))(i)
}

pub fn space_and_comments(i: &str) -> IResult<&str, ()> {
    map(
        many0(alt((
            recognize(one_of(" \t\n\r")),
            recognize(tuple((tag("#"), many0(none_of("\r\n"))))),
        ))),
        |_| (),
    )(i)
}

pub fn spaced<'s, O, F>(it: F) -> impl Fn(&'s str) -> IResult<&'s str, O>
where
    F: Fn(&'s str) -> IResult<&'s str, O>,
{
    delimited(space_and_comments, it, space_and_comments)
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
        let res = parser(i).unwrap();
        if res.0.len() != 0 {
            panic!("Did not consumed all input: {:?}", res)
        }
        res.1
    }

    fn param(s: impl Into<std::string::String>, t: TypeSpec) -> Parameter {
        Parameter { id: s.into(), spec: t }
    }

    fn result(s: impl Into<std::string::String>, t: TypeSpec) -> Result_ {
        Result_ { id: s.into(), spec: t }
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
        assert_eq!(p(type_spec, "scalar[ ]"), Array(Box::new(Single(Scalar))));
        assert_eq!(
            p(type_spec, " ( scalar , scalar [ ] , tensor < scalar > ) "),
            Tuple(vec!(Single(Scalar), Array(Box::new(Single(Scalar))), Tensor(Scalar)))
        );
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
    fn test_numeric() {
        p(numeric_literal, "12.0");
    }

    #[test]
    fn test_string() {
        fn s(s: &str) -> StringLiteral {
            StringLiteral(s.into())
        }
        assert_eq!(p(string_literal, r#""""#), s(""));
        assert_eq!(p(string_literal, r#""foo""#), s("foo"));
        assert_eq!(p(string_literal, r#"''"#), s(""));
        assert_eq!(p(string_literal, r#"'foo'"#), s("foo"));

        assert_eq!(p(string_literal, r#"'f\oo'"#), s("foo"));
        assert_eq!(p(string_literal, r#"'f\'oo'"#), s("f'oo"));
        assert_eq!(p(string_literal, r#"'f\"oo'"#), s("f\"oo"));
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
        assert!(spaced(identifier)("foo").is_ok());
        assert!(spaced(identifier)(" foo ").is_ok());
        assert!(many1(spaced(identifier))(" foo bar ").is_ok());
        assert_eq!(many1(spaced(identifier))(" foo bar\n").unwrap().1, &["foo", "bar"]);
        assert_eq!(many1(spaced(identifier))(" foo # bar\n").unwrap().1, &["foo"]);
        assert_eq!(many1(spaced(identifier))(" foo # bar\nbaz").unwrap().1, &["foo", "baz"]);
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
    }

    #[test]
    fn test_graph_def() {
        p(graph_def, "graph foo() -> () {}");
    }

    #[test]
    fn test_assignment() {
        p(assignment, "input = external(12);");
        p(assignment, "input = external(shape = [1, 3, 224, 224]);");
    }

    #[test]
    fn test_invocation() {
        p(invocation, "external(12)");
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
    }
}
