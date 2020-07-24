use nom::branch::alt;
use nom::combinator::map;
use nom::IResult;
use nom::{bytes::complete::*, character::complete::*, combinator::*, multi::*, sequence::*};

use crate::ast::*;

/*
<fragment-definition> ::= <fragment-declaration> (<body> | ";")
*/

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

// BODY

// <lvalue-expr> ::= <identifier> | <array-lvalue-expr> | <tuple-lvalue-expr>
// <array-lvalue-expr> ::= "[" [<lvalue-expr> ("," <lvalue-expr>)* ] "]"
// <tuple-lvalue-expr> ::= "(" <lvalue-expr> ("," <lvalue-expr>)+ ")" | <lvalue-expr> ("," <lvalue-expr>)+
pub fn lvalue_expr(i: &str) -> IResult<&str, LValue> {
    alt((
        map(
            delimited(
                spaced(tag("[")),
                separated_list(spaced(tag(",")), lvalue_expr),
                spaced(tag("]")),
            ),
            LValue::Array,
        ),
        map(
            delimited(
                spaced(tag("(")),
                separated_list(spaced(tag(",")), lvalue_expr),
                spaced(tag(")")),
            ),
            LValue::Tuple,
        ),
        map(separated_list(spaced(tag(",")), lvalue_expr), LValue::Tuple),
        map(spaced(identifier), LValue::Identifier),
    ))(i)
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
        map(identifier, RValue::Identifier),
        map(delimited(tag("("), rvalue, tag(")")), |rv| RValue::RValue(Box::new(rv))),
        map(invocation, RValue::Invocation),
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
pub fn string_literal(i: &str) -> IResult<&str, String> {
    pub fn inner(i: &str) -> IResult<&str, String> {
        map(
            many0(alt((
                preceded(tag("\\"), nom::character::complete::anychar),
                nom::character::complete::none_of("\"'")
            ))),
            |v: Vec<char>| v.into_iter().collect(),
        )(i)
    }
    spaced(alt((
        delimited(tag("'"), inner, tag("'")),
        delimited(tag("\""), inner, tag("\"")),
    )))(i)
}

pub fn spaced<I, O, E: nom::error::ParseError<I>, F>(it: F) -> impl Fn(I) -> nom::IResult<I, O, E>
where
    I: nom::InputTakeAtPosition,
    <I as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
    F: Fn(I) -> nom::IResult<I, O, E>,
{
    delimited(space0, it, space0)
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
    fn test_string() {
        assert_eq!(p(string_literal, r#""""#), "");
        assert_eq!(p(string_literal, r#""foo""#), "foo");
        assert_eq!(p(string_literal, r#"''"#), "");
        assert_eq!(p(string_literal, r#"'foo'"#), "foo");

        assert_eq!(p(string_literal, r#"'f\oo'"#), "foo");
        assert_eq!(p(string_literal, r#"'f\'oo'"#), "f'oo");
        assert_eq!(p(string_literal, r#"'f\"oo'"#), "f\"oo");
    }
}
