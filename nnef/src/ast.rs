use crate::internal::*;

pub mod dump;
pub mod parse;

#[derive(Clone, Debug)]
pub struct ProtoModel {
    pub doc: Document,
    pub tensors: HashMap<String, Arc<Tensor>>,
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

impl TypeSpec {
    pub fn array(self) -> TypeSpec {
        TypeSpec::Array(Box::new(self))
    }
    pub fn named(self, s: impl Into<String>) -> Parameter {
        Parameter { id: s.into(), spec: self, lit: None }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TypeName {
    Integer,
    Scalar,
    Logical,
    String,
    Any,
}

impl TypeName {
    pub fn tensor(self) -> TypeSpec {
        TypeSpec::Tensor(self)
    }
    pub fn spec(self) -> TypeSpec {
        TypeSpec::Single(self)
    }
    pub fn array(self) -> TypeSpec {
        self.spec().array()
    }
    pub fn named(self, s: impl Into<String>) -> Parameter {
        self.spec().named(s)
    }
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

impl Parameter {
    pub fn default(self, lit: impl Into<Literal>) -> Parameter {
        Parameter { lit: Some(lit.into()), ..self }
    }
}

pub fn param(s: impl Into<String>, spec: TypeSpec) -> Parameter {
    Parameter { id: s.into(), spec, lit: None }
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

impl From<bool> for Literal {
    fn from(b: bool) -> Literal {
        Literal::Logical(b)
    }
}

impl From<i64> for Literal {
    fn from(i: i64) -> Literal {
        Literal::Numeric(i.to_string())
    }
}

impl From<f32> for Literal {
    fn from(f: f32) -> Literal {
        Literal::Numeric(f.to_string())
    }
}

impl<'a> From<&'a str> for Literal {
    fn from(s: &'a str) -> Literal {
        Literal::String(s.to_string())
    }
}

pub type NumericLiteral = String;
pub type StringLiteral = String;
pub type LogicalLiteral = bool;
