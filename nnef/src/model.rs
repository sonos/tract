use crate::ast::*;
use std::collections::HashMap;
use tract_core::internal::*;

#[derive(Clone, Debug)]
pub struct ProtoModel {
    pub doc: Document,
}

impl ProtoModel {
    pub fn into_typed_model(&self) -> TractResult<TypedModel> {
        println!("parameters: {:?}", self.doc.graph_def.parameters);
        println!("results: {:?}", self.doc.graph_def.results);
        let fw = Framework::new();
        let mut model = TypedModel::default();
        for assignment in &self.doc.graph_def.body {
            println!("{:?}", assignment.left);
            println!("{:?}", assignment.right);
            fw.wire_rvalue(&mut model, &assignment.right)?;
            panic!();
        }
        todo!()
    }
}

type Primitives = HashMap<
    String,
    &'static dyn Fn(&Framework, &mut TypedModel, &Invocation) -> TractResult<TVec<OutletId>>,
>;

pub struct Framework {
    primitives: Primitives,
}

impl Framework {
    fn new() -> Framework {
        let mut primitives: Primitives = Default::default();
        primitives.insert("external".to_string(), &external);
        Framework { primitives }
    }

    fn wire_rvalue(&self, model: &mut TypedModel, rv: &RValue) -> TractResult<TVec<OutletId>> {
        match rv {
            RValue::Invocation(inv) => self.wire_invocation(model, inv),
            _ => panic!(),
        }
    }

    fn wire_invocation(
        &self,
        model: &mut TypedModel,
        invocation: &Invocation,
    ) -> TractResult<TVec<OutletId>> {
        let prim = self
            .primitives
            .get(&invocation.id)
            .ok_or_else(|| format!("No definition for {:?}", invocation.id))?;
        (prim)(self, model, invocation)
    }
}

fn external(
    _fw: &Framework,
    model: &mut TypedModel,
    invocation: &Invocation,
) -> TractResult<TVec<OutletId>> {
    let type_name = invocation.generic_type_name.unwrap_or(TypeName::Scalar); // TODO pull default from stdlib (?)
    let dt = if type_name == TypeName::Scalar { f32::datum_type() } else { todo!() };
    let shape = invocation.named_arg("shape")?.to_shape_fact()?;
    Ok(tvec!(model.add_source("", TypedFact::dt_shape(dt, shape)?)?))
}

impl LValue {
    fn to_identifier(&self) -> TractResult<&str> {
        match self {
            LValue::Identifier(id) => Ok(&**id),
            _ => bail!("Not an identifier")
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
    fn named_arg(&self, name: &str) -> TractResult<&RValue> {
        self.get_named_arg(name).ok_or_else(|| format!("expected argument {}", name).into())
    }

    fn get_named_arg(&self, name: &str) -> Option<&RValue> {
        self.arguments.iter().find(|arg| arg.id.as_deref() == Some(name)).map(|arg| &arg.rvalue)
    }
}

impl RValue {
    fn to_shape_fact(&self) -> TractResult<ShapeFact> {
        let dims = self.as_array().ok_or_else(|| format!("Expected {:?} to be a shape", self))?;
        let dims = dims.iter().map(|d| d.to_dim()).collect::<TractResult<TVec<_>>>()?;
        ShapeFact::from_dims(dims)
    }

    fn to_dim(&self) -> TractResult<TDim> {
        self.as_literal()
            .map(|l| l.to_dim())
            .transpose()?
            .ok_or_else(|| format!("Expected {:?} to be a dim", self).into())
    }

    fn as_array(&self) -> Option<&[RValue]> {
        match self {
            RValue::Array(values) => Some(values),
            _ => None,
        }
    }

    fn as_literal(&self) -> Option<&Literal> {
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
