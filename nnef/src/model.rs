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
        let mut builder = ModelBuilder { framework, model: TypedModel::default(), scopes: vec![] };
        builder.wire(self)?;
        Ok(builder.model)
    }
}

pub struct ModelBuilder {
    pub framework: Framework,
    pub model: TypedModel,
    pub scopes: Vec<HashMap<String, OutletId>>,
}

impl ModelBuilder {
    pub fn wire(&mut self, proto: &ProtoModel) -> TractResult<()> {
        self.scopes.push(HashMap::new());
        self.wire_body(&proto.doc.graph_def.body)
            .chain_err(|| format!("Mapping graph `{}' to tract", proto.doc.graph_def.id))?;
        self.scopes.pop();
        Ok(())
    }

    pub fn wire_body(&mut self, body: &[Assignment]) -> TractResult<()> {
        for assignment in body {
            let outlets = assignment.right.to_wires(self)?;
            assert!(outlets.len() == 1);
            let outlet = outlets[0];
            let identifier = assignment.left.to_identifier()?;
            self.model.node_mut(outlet.node).name = identifier.to_string();
            self.scopes.last_mut().unwrap().insert(identifier.to_string(), outlet);
        }
        Ok(())
    }

    pub fn wire_invocation(&mut self, invocation: &Invocation) -> TractResult<TVec<OutletId>> {
        if let Some(prim) = self.framework.primitives.get(&invocation.id).cloned() {
            (prim)(self, invocation)
                .chain_err(|| format!("Plugging primitive `{}'", invocation.id))
        } else if let Some(fragment) =
            self.framework.stdlib.iter().find(|f| f.decl.id == invocation.id).cloned()
        {
            if fragment.body.is_none() {
                bail!("fragment for `{}' is declarative, not defining. Maybe a missing primitive ?", invocation.id)
            }
            self.wire_fragment_invocation(fragment, invocation)
                .chain_err(|| format!("Expanding fragment `{}'", invocation.id))
        } else {
            bail!("No fragment for {:?}", invocation.id)
        }
    }

    pub fn wire_fragment_invocation(
        &mut self,
        fragment: Arc<FragmentDef>,
        invocation: &Invocation,
    ) -> TractResult<TVec<OutletId>> {
        let mut inner_scope = HashMap::new();
        for (ix, arg) in invocation.arguments.iter().enumerate() {
            let value = arg.rvalue.to_wire(self)?;
            let id = arg.id.as_deref().unwrap_or(&fragment.decl.parameters[ix].id).to_string();
            inner_scope.insert(id, value);
        }
        self.scopes.push(inner_scope);
        self.wire_body(&fragment.body.as_ref().unwrap())?;
        let inner_scope = self.scopes.pop().unwrap();
        Ok(fragment
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

impl Invocation {
    pub fn pos_arg(&self, pos: usize) -> TractResult<&RValue> {
        self.get_pos_arg(pos).ok_or_else(|| format!("expected argument {}", pos).into())
    }

    pub fn named_arg(&self, name: &str) -> TractResult<&RValue> {
        self.get_named_arg(name).ok_or_else(|| format!("expected argument {}", name).into())
    }

    pub fn get_named_arg(&self, name: &str) -> Option<&RValue> {
        self.arguments.iter().find(|arg| arg.id.as_deref() == Some(name)).map(|arg| &arg.rvalue)
    }

    pub fn get_pos_arg(&self, pos: usize) -> Option<&RValue> {
        self.arguments.get(pos).map(|arg| &arg.rvalue)
    }
}

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

    fn multicast(builder: &mut ModelBuilder, inputs: &[OutletId]) -> TractResult<TVec<OutletId>> {
        let ranks = inputs
            .iter()
            .map(|&i| Ok(builder.model.outlet_fact(i)?.rank()))
            .collect::<TractResult<Vec<usize>>>()?;
        let max_rank = ranks.iter().copied().max().unwrap();
        (inputs.iter())
            .zip(ranks.iter())
            .map(|(&i, &r)| {
                (r..max_rank)
                    .try_fold(i, |w, n| Ok(builder.model.wire_node("", AxisOp::Add(n), &[w])?[0]))
            })
            .collect()
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
                let left = left.to_wire(builder)?;
                let right = right.to_wire(builder)?;
                let inputs = Self::multicast(builder, &[left, right])?;
                let op = match &**op {
                    "+" => tract_core::ops::math::add::bin_typed(),
                    "-" => tract_core::ops::math::sub::bin_typed(),
                    "*" => tract_core::ops::math::mul::bin_typed(),
                    "/" => tract_core::ops::math::div::bin_typed(),
                    "^" => tract_core::ops::math::pow::bin_typed(),
                    "==" => tract_core::ops::logic::equals::bin_typed(),
                    "!=" => tract_core::ops::logic::not_equals::bin_typed(),
                    ">" => tract_core::ops::logic::greater::bin_typed(),
                    ">=" => tract_core::ops::logic::greater_equal::bin_typed(),
                    "<" => tract_core::ops::logic::lesser::bin_typed(),
                    "<=" => tract_core::ops::logic::lesser_equal::bin_typed(),
                    op => bail!("Unknown binary operator: {}", op),
                };
                builder.model.wire_node("", op, &inputs)
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
                let wire = self.to_wires_rec(builder, false)?[0];
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
