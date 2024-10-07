use crate::internal::*;
use tract_itertools::Itertools;

pub mod dump;
pub mod dump_doc;
pub mod parse;
pub mod quant;

#[derive(Clone, Debug)]
pub struct ProtoModel {
    pub doc: Document,
    pub tensors: HashMap<Identifier, Arc<Tensor>>,
    pub quantization: Option<HashMap<Identifier, QuantFormat>>,
    pub resources: HashMap<String, Arc<dyn Resource>>,
}

impl ProtoModel {
    pub fn validate(&self) -> TractResult<()> {
        self.doc.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuantFormat {
    Linear { params: QParams, bits: i8, signed: bool },
}

impl QuantFormat {
    pub fn from_dt(datum_type: DatumType) -> Option<QuantFormat> {
        if let Some(params) = datum_type.qparams() {
            let quant_format = QuantFormat::Linear {
                params,
                bits: 8 * datum_type.size_of() as i8,
                signed: datum_type.is_signed(),
            };
            Some(quant_format)
        } else {
            None
        }
    }

    pub fn datum_type(&self) -> DatumType {
        match self {
            QuantFormat::Linear { params, bits, signed } => match (bits, signed) {
                (8, true) => DatumType::QI8(*params),
                (8, false) => DatumType::QU8(*params),
                (32, true) => DatumType::QI32(*params),
                (32, false) => DatumType::U32,
                _ => todo!(),
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Document {
    pub version: NumericLiteral,
    pub extension: Vec<(Identifier, String)>,
    pub fragments: Vec<FragmentDef>,
    pub graph_def: GraphDef,
}

impl Document {
    pub fn validate(&self) -> TractResult<()> {
        for frag in &self.fragments {
            frag.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
    pub fn named(self, s: impl AsRef<str>) -> Parameter {
        Parameter { id: s.as_ref().into(), spec: self, lit: None, doc: None }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeName {
    Integer,
    Scalar,
    #[cfg(feature = "complex")]
    Complex,
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
    pub fn named(self, s: impl AsRef<str>) -> Parameter {
        self.spec().named(s)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphDef {
    pub id: Identifier,
    pub parameters: Vec<Identifier>,
    pub results: Vec<Identifier>,
    pub body: Vec<Assignment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentDef {
    pub decl: FragmentDecl,
    pub body: Option<Vec<Assignment>>,
}

impl FragmentDef {
    pub fn validate(&self) -> TractResult<()> {
        self.decl.validate().with_context(|| format!("Invalid fragment {:?}", self.decl.id))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentDecl {
    pub id: Identifier,
    pub generic_decl: Option<Option<TypeName>>,
    pub parameters: Vec<Parameter>,
    pub results: Vec<Result_>,
}

impl FragmentDecl {
    pub fn validate(&self) -> TractResult<()> {
        if let Some(dup) = self
            .parameters
            .iter()
            .map(|p| &p.id)
            .sorted()
            .group_by(|x| x.to_owned())
            .into_iter()
            .find_map(|(key, values)| if values.count() > 1 { Some(key) } else { None })
        {
            bail!("Duplicate parameter name found {:?}", dup);
        }
        if let Some(dup) = self
            .results
            .iter()
            .map(|p| &p.id)
            .sorted()
            .group_by(|x| x.to_owned())
            .into_iter()
            .find_map(|(key, values)| if values.count() > 1 { Some(key) } else { None })
        {
            bail!("Duplicate result name found {:?}", dup);
        }
        if let Some(dup) = self
            .parameters
            .iter()
            .map(|p| &p.id)
            .chain(self.results.iter().map(|p| &p.id))
            .sorted()
            .group_by(|x| x.to_owned())
            .into_iter()
            .find_map(|(key, values)| if values.count() > 1 { Some(key) } else { None })
        {
            bail!("Same name used as parameter and result {:?}", dup);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parameter {
    pub id: Identifier,
    pub spec: TypeSpec,
    pub lit: Option<Literal>,
    pub doc: Option<String>,
}

impl Parameter {
    pub fn default(self, lit: impl Into<Literal>) -> Parameter {
        Parameter { lit: Some(lit.into()), ..self }
    }

    pub fn doc(mut self, s: impl Into<String>) -> Parameter {
        self.doc = Some(s.into());
        self
    }
}

pub fn param(s: impl AsRef<str>, spec: TypeSpec) -> Parameter {
    Parameter { id: s.as_ref().into(), spec, lit: None, doc: None }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Result_ {
    pub id: Identifier,
    pub spec: TypeSpec,
}

impl<S: Into<String>> From<(S, TypeSpec)> for Result_ {
    fn from(v: (S, TypeSpec)) -> Result_ {
        Result_ { id: Identifier(v.0.into()), spec: v.1 }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default, Ord, PartialOrd, Hash)]
pub struct Identifier(pub String);

impl From<&str> for Identifier {
    fn from(value: &str) -> Self {
        Identifier(value.to_string())
    }
}

impl AsRef<str> for Identifier {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assignment {
    pub left: LValue,
    pub right: RValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LValue {
    Identifier(Identifier),
    Array(Vec<LValue>),
    Tuple(Vec<LValue>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Invocation {
    pub id: Identifier,
    pub generic_type_name: Option<TypeName>,
    pub arguments: Vec<Argument>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Argument {
    pub id: Option<Identifier>,
    pub rvalue: RValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RValue {
    Identifier(Identifier),
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

impl RValue {
    pub fn boxed(self) -> Box<RValue> {
        Box::new(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Comprehension {
    pub loop_iters: Vec<(Identifier, RValue)>,
    pub filter: Option<RValue>,
    pub yields: RValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Subscript {
    Single(RValue),
    Range(Option<RValue>, Option<RValue>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IfThenElse {
    pub cond: RValue,
    pub then: RValue,
    pub otherwise: RValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
        Literal::Numeric(format!("{f:?}"))
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
