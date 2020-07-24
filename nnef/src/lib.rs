use nom::branch::alt;
use nom::combinator::map;
use nom::IResult;
use nom::character::complete::tag;

/*
<fragment-definition> ::= <fragment-declaration> (<body> | ";")
<fragment-declaration> ::= "fragment" <identifier> [<generic-declaration>]
"(" <parameter-list> ")" "->" "(" <result-list> ")"
<generic-declaration> ::= "<" "?" ["=" <type-name>] ">"
<parameter-list> ::= <parameter> ("," <parameter>)*
<parameter> ::= <identifier> ":" <type-spec> ["=" <literal-expr>]
<result-list> ::= <result> ("," <result>)*
<result> ::= <identifier> ":" <type-spec>

<type-name> ::= "integer" | "scalar" | "logical" | "string" | "?"
*/

enum TypeName {
    Integer,
    Scalar,
    Logical,
    String,
    Any,
}

pub fn type_name(i: &str) -> IResult<&str, TypeName> {
    alt((map(tag("integer"), |_| TypeName::Integer), map(tag("scalar"), |_| TypeName::Scalar)))()
}
