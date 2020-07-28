use crate::ast::*;
use std::collections::HashMap;
use tract_core::internal::*;

use crate::primitives::Primitives;

#[derive(Clone, Debug)]
pub enum Value {
    Wire(OutletId),
    Tensor(Arc<Tensor>),
}

impl Value {
    pub fn to_outlet(&self, builder: &mut ModelBuilder) -> TractResult<OutletId> {
        match self {
            Value::Wire(w) => Ok(*w),
            Value::Tensor(t) => builder.model.add_const("", t.clone()),
        }
    }

    pub fn to_tensor(&self, builder: &mut ModelBuilder) -> TractResult<Arc<Tensor>> {
        match self {
            Value::Wire(w) => {
                builder.model.outlet_fact(*w)?.konst.clone().ok_or("Not a constant".into())
            }
            Value::Tensor(t) => Ok(t.clone()),
        }
    }
}

pub struct ModelBuilder {
    pub framework: Framework,
    pub model: TypedModel,
    pub assigned: HashMap<String, Value>,
}

impl ModelBuilder {
    pub fn wire(&mut self, proto: &ProtoModel) -> TractResult<()> {
        for assignment in &proto.doc.graph_def.body {
            let value = self.wire_rvalue(&assignment.right)?;
            let identifier = assignment.left.to_identifier()?;
            if let Value::Wire(outlet) = value {
                self.model.node_mut(outlet.node).name = identifier.to_string();
            }
            self.assigned.insert(identifier.to_string(), value);
        }
        Ok(())
    }

    pub fn wire_rvalue(&mut self, rv: &RValue) -> TractResult<Value> {
        match rv {
            RValue::Identifier(id) => self
                .assigned
                .get(id)
                .cloned()
                .ok_or_else(|| format!("No value for name {}", id).into()),
            RValue::Invocation(inv) => self.wire_invocation(inv),
            RValue::Literal(Literal::Numeric(f)) => {
                Ok(Value::Tensor(if f.0.contains(".") || f.0.contains("e") {
                    f.0.parse::<f64>()
                        .map(rctensor0)
                        .map_err(|_| format!("Can not parse {} as f64", f.0))?
                } else {
                    f.0.parse::<i64>()
                        .map(rctensor0)
                        .map_err(|_| format!("Can not parse {} as i64", f.0))?
                }))
            }
            _ => panic!(),
        }
    }

    pub fn wire_invocation(&mut self, invocation: &Invocation) -> TractResult<Value> {
        let prim = self
            .framework
            .primitives
            .get(&invocation.id)
            .ok_or_else(|| format!("No definition for {:?}", invocation.id))?
            .clone();
        (prim)(self, invocation)
    }
}

#[derive(Clone, Debug)]
pub struct ProtoModel {
    pub doc: Document,
}

impl ProtoModel {
    pub fn into_typed_model(&self) -> TractResult<TypedModel> {
        let framework = Framework::new();
        let mut builder =
            ModelBuilder { framework, model: TypedModel::default(), assigned: HashMap::new() };
        builder.wire(self)?;
        Ok(builder.model)
    }
}

pub struct Framework {
    primitives: Primitives,
}

impl Framework {
    fn new() -> Framework {
        Framework { primitives: crate::primitives::primitives() }
    }
}

impl LValue {
    fn to_identifier(&self) -> TractResult<&str> {
        match self {
            LValue::Identifier(id) => Ok(&**id),
            _ => bail!("Expected an identifier, found a tuple: {:?}", self),
        }
    }
    fn to_identifiers(&self) -> TractResult<TVec<&str>> {
        match self {
            LValue::Identifier(_) => Ok(tvec!(self.to_identifier()?)),
            LValue::Tuple(ids) => ids.iter().map(|id| id.to_identifier()).collect(),
            LValue::Array(ids) => ids.iter().map(|id| id.to_identifier()).collect(),
        }
    }
}

impl Invocation {
    /*
    fn named_arg_lit(&self, name: &str) -> TractResult<&Literal> {
    let rv = self.named_arg(name)?;
    rv.as_literal().ok_or_else(|| {
    format!("Expected argument `{}' to be a literal, got {:?} instead", name, rv).into()
    })
    }
    */
    pub fn named_arg(&self, name: &str) -> TractResult<&RValue> {
        self.get_named_arg(name).ok_or_else(|| format!("expected argument {}", name).into())
    }

    pub fn get_named_arg(&self, name: &str) -> Option<&RValue> {
        self.arguments.iter().find(|arg| arg.id.as_deref() == Some(name)).map(|arg| &arg.rvalue)
    }
}

impl RValue {
    pub fn eval(&self, builder: &mut ModelBuilder) -> TractResult<Value> {
        builder.wire_rvalue(self)
    }

    pub fn to_shape_fact(&self) -> TractResult<ShapeFact> {
        let dims = self.as_array().ok_or_else(|| format!("Expected {:?} to be a shape", self))?;
        let dims = dims.iter().map(|d| d.to_dim()).collect::<TractResult<TVec<_>>>()?;
        ShapeFact::from_dims(dims)
    }

    pub fn to_dim(&self) -> TractResult<TDim> {
        self.as_literal()
            .map(|l| l.to_dim())
            .transpose()?
            .ok_or_else(|| format!("Expected {:?} to be a dim", self).into())
    }

    pub fn as_array(&self) -> Option<&[RValue]> {
        match self {
            RValue::Array(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_literal(&self) -> Option<&Literal> {
        match self {
            RValue::Literal(lit) => Some(lit),
            _ => None,
        }
    }
}

impl Literal {
    /*
    fn as_shape(&self) -> TractResult<ShapeFact> {
    match self {
    Literal::Array(dims) => ShapeFact::from_dims(
    &dims.iter().map(|d| d.as_dim()).collect::<TractResult<TVec<TDim>>>()?,
    ),
    _ => bail!("not a shape"),
    }
    }
    */

    fn to_dim(&self) -> TractResult<TDim> {
        match self {
            Literal::Numeric(d) => Ok(d.0.parse::<i64>()?.to_dim()),
            _ => bail!("not a dimension"),
        }
    }
}
