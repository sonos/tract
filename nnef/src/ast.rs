pub mod dump;
pub mod parse;

use std::sync::Arc;

pub fn stdlib() -> Vec<Arc<FragmentDef>> {
    parse::parse_fragments(include_str!("../stdlib.nnef"))
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    pub version: NumericLiteral,
    pub extension: Vec<Vec<String>>,
    pub fragments: Vec<FragmentDef>,
    pub graph_def: GraphDef,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeSpec {
    Single(TypeName),
    Tensor(TypeName),
    Array(Box<TypeSpec>),
    Tuple(Vec<TypeSpec>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TypeName {
    Integer,
    Scalar,
    Logical,
    String,
    Any,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphDef {
    pub id: String,
    pub parameters: Vec<String>,
    pub results: Vec<String>,
    pub body: Vec<Assignment>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FragmentDef {
    pub decl: FragmentDecl,
    pub body: Option<Vec<Assignment>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FragmentDecl {
    pub id: String,
    pub generic_decl: Option<Option<TypeName>>,
    pub parameters: Vec<Parameter>,
    pub results: Vec<Result_>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub id: String,
    pub spec: TypeSpec,
    pub lit: Option<Literal>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Result_ {
    pub id: String,
    pub spec: TypeSpec,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Assignment {
    pub left: LValue,
    pub right: RValue,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LValue {
    Identifier(String),
    Array(Vec<LValue>),
    Tuple(Vec<LValue>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Invocation {
    pub id: String,
    pub generic_type_name: Option<TypeName>,
    pub arguments: Vec<Argument>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Argument {
    pub id: Option<String>,
    pub rvalue: RValue,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RValue {
    Identifier(String),
    Literal(Literal),
    Binary(Box<RValue>, String, Box<RValue>),
    Unary(String, Box<RValue>),
    Tuple(Vec<RValue>),
    Array(Vec<RValue>),
    Subscript(Box<RValue>, Box<Subscript>),
    Comprehension(Box<Comprehension>),
    IfThenElse(Box<IfThenElse>),
    Invocation(Invocation),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Comprehension {
    pub loop_iters: Vec<(String, RValue)>,
    pub filter: Option<RValue>,
    pub yields: RValue,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Subscript {
    Single(RValue),
    Range(Option<RValue>, Option<RValue>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct IfThenElse {
    pub cond: RValue,
    pub then: RValue,
    pub otherwise: RValue,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    Numeric(NumericLiteral),
    String(StringLiteral),
    Logical(LogicalLiteral),
    Array(Vec<Literal>),
    Tuple(Vec<Literal>),
}

pub type NumericLiteral = String;
pub type StringLiteral = String;
pub type LogicalLiteral = bool;
