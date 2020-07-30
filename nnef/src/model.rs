use crate::ast::*;
use std::collections::HashMap;
use tract_core::internal::*;

use crate::primitives::Primitives;

pub struct Framework {
    stdlib: Vec<Arc<FragmentDef>>,
    primitives: Primitives,
}

impl Framework {
    fn new() -> Framework {
        Framework {
            stdlib: crate::parser::parse_fragments(include_str!("../stdlib.nnef"))
                .unwrap()
                .into_iter()
                .map(Arc::new)
                .collect(),
            primitives: crate::primitives::primitives(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProtoModel {
    pub doc: Document,
}

impl ProtoModel {
    pub fn into_typed_model(&self) -> TractResult<TypedModel> {
        let framework = Framework::new();
        let mut builder = ModelBuilder {
            framework,
            fragments: self.doc.fragments.iter().map(|f| Arc::new(f.clone())).collect(),
            model: TypedModel::default(),
            naming_scopes: vec!(),
            scopes: vec![],
        };
        builder.scopes.push(HashMap::new());
        builder.naming_scopes.push(self.doc.graph_def.id.to_string());
        builder.wire_body(&self.doc.graph_def.body)
            .chain_err(|| format!("Mapping graph `{}' to tract", self.doc.graph_def.id))?;
        let vars = builder.scopes.pop().unwrap();
        let outputs = self.doc.graph_def.results.iter().map(|s| vars[s]).collect::<TVec<OutletId>>();
        builder.model.set_output_outlets(&outputs)?;
        Ok(builder.model)
    }
}

pub struct ModelBuilder {
    pub framework: Framework,
    pub fragments: Vec<Arc<FragmentDef>>,
    pub model: TypedModel,
    pub naming_scopes: Vec<String>,
    pub scopes: Vec<HashMap<String, OutletId>>,
}

impl ModelBuilder {
    pub fn wire(&mut self, op: impl Into<Box<dyn TypedOp>>, inputs: &[OutletId]) -> TractResult<TVec<OutletId>> {
        let op = op.into();
        self.model.wire_node(format!("{}.{}", self.naming_scopes.join("."), op.as_op().name()), op, inputs)
    }
}

#[derive(Clone, Debug)]
pub struct AugmentedInvocation<'a> {
    pub invocation: &'a Invocation,
    pub fragment: Arc<FragmentDef>,
}

impl<'a> AugmentedInvocation<'a> {
    pub fn pos_arg(&self, pos: usize) -> TractResult<Cow<RValue>> {
        self.get_pos_arg(pos).ok_or_else(|| format!("expected argument {}", pos).into())
    }

    pub fn named_arg(&self, name: &str) -> TractResult<Cow<RValue>> {
        self.get_named_arg(name).ok_or_else(|| format!("expected argument {}", name).into())
    }

    pub fn get_named_arg(&self, name: &str) -> Option<Cow<RValue>> {
        // first look explicit name in invocation arguments
        if let Some(arg) =
            self.invocation.arguments.iter().find(|arg| arg.id.as_deref() == Some(name))
        {
            return Some(Cow::Borrowed(&arg.rvalue));
        }
        // then use fragment prototype:
        if let Some((ix, param)) =
            self.fragment.decl.parameters.iter().enumerate().find(|(_ix, param)| param.id == name)
        {
            // check that all previous (and our) arguments are positional (todo:
            // valid args when building augmented_invocation)
            if self.invocation.arguments.iter().take(ix + 1).all(|arg| arg.id.is_none()) {
                return Some(Cow::Borrowed(&self.invocation.arguments[ix].rvalue));
            }
            if let Some(rv) = &param.lit {
                return Some(Cow::Owned(RValue::Literal(rv.clone())));
            }
        }
        None
    }

    pub fn get_pos_arg(&self, pos: usize) -> Option<Cow<RValue>> {
        self.invocation.arguments.get(pos).map(|arg| Cow::Borrowed(&arg.rvalue))
    }
}

impl ModelBuilder {
    fn augmented_invocation<'a>(
        &self,
        invocation: &'a Invocation,
    ) -> TractResult<AugmentedInvocation<'a>> {
        let fragment = self
            .framework
            .stdlib
            .iter()
            .find(|f| f.decl.id == invocation.id)
            .cloned()
            .or_else(|| self.fragments.iter().find(|f| f.decl.id == invocation.id).cloned())
            .ok_or_else(|| format!("No fragment definition found for `{}'", invocation.id))?;
        Ok(AugmentedInvocation { invocation, fragment })
    }


    pub fn wire_body(&mut self, body: &[Assignment]) -> TractResult<()> {
        for assignment in body {
            let identifiers = assignment.left.to_identifiers()?;
            self.naming_scopes.push(identifiers[0].to_string());
            let outlets = assignment
                .right
                .to_wires(self)
                .chain_err(|| format!("Plugging in assignement for {:?}", identifiers.join(", ")))?;
            if outlets.len() != identifiers.len() {
                bail!("Assignement for {} received {} value(s).", identifiers.join(","), outlets.len())
            }
            self.model.node_mut(outlets[0].node).name = format!("{}", self.naming_scopes.join("."));
            for (id, outlet) in identifiers.iter().zip(outlets.iter()) {
                self.scopes.last_mut().unwrap().insert(id.to_string(), *outlet);
            }
            self.naming_scopes.pop();
        }
        Ok(())
    }

    pub fn wire_invocation(&mut self, invocation: &Invocation) -> TractResult<TVec<OutletId>> {
        let augmented_invocation = self.augmented_invocation(invocation)?;
        if let Some(prim) = self.framework.primitives.get(&invocation.id).cloned() {
            (prim)(self, &augmented_invocation)
                .chain_err(|| format!("Plugging primitive `{}'", invocation.id))
        } else if augmented_invocation.fragment.body.is_some() {
            self.wire_fragment_invocation(&augmented_invocation)
                .chain_err(|| format!("Expanding fragment `{}'", invocation.id))
        } else {
            bail!(
                "fragment for `{}' is declarative, not defining. Maybe a missing primitive ?",
                invocation.id
            )
        }
    }

    pub fn wire_fragment_invocation(
        &mut self,
        invocation: &AugmentedInvocation,
    ) -> TractResult<TVec<OutletId>> {
        let mut inner_scope = HashMap::new();
        for (ix, par) in invocation.fragment.decl.parameters.iter().enumerate() {
            if let Some(arg) = invocation.get_named_arg(&par.id) {
                inner_scope.insert(par.id.to_string(), arg.to_wire(self)?);
            } else if let Some(arg) = invocation.get_pos_arg(ix) {
                inner_scope.insert(par.id.to_string(), arg.to_wire(self)?);
            } else if let Some(lit) = &par.lit {
                inner_scope.insert(par.id.to_string(), RValue::Literal(lit.clone()).to_wire(self)?);
            }
        }
        self.scopes.push(inner_scope);
        self.naming_scopes.push(invocation.invocation.id.to_string());
        self.wire_body(&invocation.fragment.body.as_ref().unwrap())?;
        self.naming_scopes.pop();
        let inner_scope = self.scopes.pop().unwrap();
        Ok(invocation
            .fragment
            .decl
            .results
            .iter()
            .map(|res| inner_scope.get(&res.id).unwrap())
            .cloned()
            .collect())
    }
}

impl LValue {
    fn to_identifier(&self) -> TractResult<&str> {
        match self {
            LValue::Identifier(id) => Ok(&**id),
            _ => bail!("Expected an identifier, found a tuple: {:?}", self),
        }
    }

    #[allow(dead_code)]
    fn to_identifiers(&self) -> TractResult<TVec<&str>> {
        match self {
            LValue::Identifier(_) => Ok(tvec!(self.to_identifier()?)),
            LValue::Tuple(ids) => ids.iter().map(|id| id.to_identifier()).collect(),
            LValue::Array(ids) => ids.iter().map(|id| id.to_identifier()).collect(),
        }
    }
}

impl Invocation {}

impl RValue {
    pub fn to_wire(&self, builder: &mut ModelBuilder) -> TractResult<OutletId> {
        let wires = self.to_wires(builder)?;
        if wires.len() != 1 {
            bail!("Expected 1 wire, got {:?}", wires.len());
        }
        Ok(wires[0])
    }

    pub fn to_wires(&self, builder: &mut ModelBuilder) -> TractResult<TVec<OutletId>> {
        self.to_wires_rec(builder, true)
    }

    fn to_wires_rec(
        &self,
        builder: &mut ModelBuilder,
        can_try_const: bool,
    ) -> TractResult<TVec<OutletId>> {
        match self {
            RValue::Identifier(id) => {
                let outlet = builder
                    .scopes
                    .last()
                    .unwrap()
                    .get(id)
                    .cloned()
                    .ok_or_else(|| format!("No value for name {}", id))?;
                Ok(tvec!(outlet))
            }
            RValue::Invocation(inv) => builder.wire_invocation(inv),
            RValue::Binary(left, op, right) => {
                let op = match &**op {
                    "+" => "add",
                    "-" => "sub",
                    "*" => "mul",
                    "/" => "div",
                    "^" => "pow",
                    ">" => "gt",
                    "<" => "lt",
                    "==" => "eq",
                    "!=" => "ne",
                    ">=" => "ge",
                    "<=" => "le",
                    op => bail!("Unknown binary operator: {}", op),
                };
                let inv = Invocation {
                    id: op.to_string(),
                    generic_type_name: None,
                    arguments: vec![
                        Argument { id: None, rvalue: left.as_ref().clone() },
                        Argument { id: None, rvalue: right.as_ref().clone() },
                    ],
                };
                builder.wire_invocation(&inv)
            }
            _ if can_try_const => {
                let tensor = self.to_tensor(builder)?;
                Ok(tvec!(builder.model.add_const("", tensor)?))
            }
            _ => bail!("failed to wire {:?}", self),
        }
    }

    pub fn to_tensor(&self, builder: &mut ModelBuilder) -> TractResult<Arc<Tensor>> {
        match self {
            RValue::Literal(Literal::Array(array)) => {
                if array.len() == 0 {
                    return Ok(rctensor1::<i64>(&[]));
                }
                todo!()
            }
            RValue::Literal(Literal::Logical(LogicalLiteral(b))) => Ok(rctensor0(*b)),
            RValue::Literal(Literal::Numeric(f)) => {
                if f.0.contains(".") || f.0.contains("e") {
                    f.0.parse::<f32>()
                        .map(rctensor0)
                        .map_err(|_| format!("Can not parse {} as f32", f.0).into())
                } else {
                    f.0.parse::<i64>()
                        .map(rctensor0)
                        .map_err(|_| format!("Can not parse {} as i64", f.0).into())
                }
            }
            RValue::Literal(Literal::String(StringLiteral(s))) => Ok(rctensor0(s.to_owned())),
            RValue::Array(array) | RValue::Tuple(array) => {
                if array.len() == 0 {
                    return Ok(rctensor1::<i64>(&[]));
                }
                let values: Vec<Arc<Tensor>> = array
                    .iter()
                    .map(|item| item.to_tensor(builder))
                    .collect::<TractResult<Vec<Arc<Tensor>>>>()?;
                let values: Vec<Tensor> = values
                    .into_iter()
                    .map(|t| {
                        let mut t = t.into_tensor();
                        t.insert_axis(0)?;
                        Ok(t)
                    })
                    .collect::<TractResult<Vec<_>>>()?;
                Tensor::stack_tensors(0, &values).map(|t| t.into_arc_tensor())
            }
            _ => {
                let wire = self
                    .to_wires_rec(builder, false)
                    .chain_err(|| "Failed to get a tensor, trying an wire instead.")?[0];
                builder.model.outlet_fact(wire)?.konst.clone().ok_or("Not a constant".into())
            }
        }
    }

    pub fn to_scalar<D: Datum>(&self, builder: &mut ModelBuilder) -> TractResult<D> {
        let d = self.to_tensor(builder)?;
        Ok(d.to_scalar::<D>()?.clone())
    }

    pub fn to_shape_fact(&self, builder: &mut ModelBuilder) -> TractResult<ShapeFact> {
        let shape = self.to_tensor(builder)?;
        let shape = shape.cast_to::<TDim>()?;
        if shape.rank() != 1 {
            bail!("Shape are expected to be vectors (1D tensor) found: {:?}")
        }
        ShapeFact::from_dims(shape.as_slice::<TDim>()?)
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
    fn to_dim(&self) -> TractResult<TDim> {
        match self {
            Literal::Numeric(d) => Ok(d.0.parse::<i64>()?.to_dim()),
            _ => bail!("not a dimension"),
        }
    }
}
