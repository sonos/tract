use nom::branch::alt;
use nom::combinator::map;
use nom::lib::std::ops::{AddAssign, Shl, Shr};
use nom::IResult;
use nom::*;
use nom::{bytes::complete::*, character::complete::*, combinator::*, multi::*, sequence::*};

/*
<fragment-definition> ::= <fragment-declaration> (<body> | ";")
<fragment-declaration> ::= "fragment" <identifier> [<generic-declaration>]
"(" <parameter-list> ")" "->" "(" <result-list> ")"
<generic-declaration> ::= "<" "?" ["=" <type-name>] ">"
<parameter-list> ::= <parameter> ("," <parameter>)*
<parameter> ::= <identifier> ":" <type-spec> ["=" <literal-expr>]
<result-list> ::= <result> ("," <result>)*
<result> ::= <identifier> ":" <type-spec>
*/

#[derive(Debug, PartialEq)]
pub enum TypeSpec {
    Single(TypeName),
    Tensor(TypeName),
    Array(Box<TypeSpec>),
    Tuple(Vec<TypeSpec>),
}

#[derive(Debug, PartialEq)]
pub enum TypeName {
    Integer,
    Scalar,
    Logical,
    String,
    Any,
}

/*
struct FragmentDef {
decl: FragmentDecl,
body: Body,
}
*/

#[derive(Debug, PartialEq)]
pub struct FragmentDecl {
    id: String,
    parameters: Vec<Parameter>,
    results: Vec<Result_>,
}

#[derive(Debug, PartialEq)]
pub struct Parameter {
    id: String,
    spec: TypeSpec,
}

#[derive(Debug, PartialEq)]
pub struct Result_ {
    id: String,
    spec: TypeSpec,
}

// identifier: identifiers must consist of the following ASCII characters: _, [a-z], [A-Z], [0-9]. The identifier must not start with a digit.
pub fn identifier(i: &str) -> IResult<&str, &str> {
    recognize(pair(alpha1, nom::multi::many0(nom::branch::alt((alphanumeric1, tag("_"))))))(i)
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

    #[test]
    fn test_type_spec() {
        use TypeName::*;
        use TypeSpec::*;
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
}
