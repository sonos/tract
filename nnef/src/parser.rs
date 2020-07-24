use nom::branch::alt;
use nom::combinator::map;
use nom::IResult;
use nom::{bytes::complete::*, character::complete::*, combinator::*, multi::*, sequence::*};

use crate::ast::*;

/*
<fragment-definition> ::= <fragment-declaration> (<body> | ";")
*/

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

// identifier: identifiers must consist of the following ASCII characters: _, [a-z], [A-Z], [0-9]. The identifier must not start with a digit.
pub fn identifier(i: &str) -> IResult<&str, String> {
    map(
        recognize(pair(alpha1, nom::multi::many0(nom::branch::alt((alphanumeric1, tag("_")))))),
        String::from,
    )(i)
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
}
