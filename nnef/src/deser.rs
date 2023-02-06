use std::ops::ControlFlow;

use tract_core::tract_data::itertools::Itertools;

use crate::ast::*;
use crate::internal::*;

pub struct ModelBuilder<'a> {
    pub framework: &'a Nnef,
    pub registries: Vec<Identifier>,
    pub model: TypedModel,
    pub naming_scopes: Vec<Identifier>,
    pub scopes: Vec<HashMap<Identifier, Value>>,
    pub proto_model: &'a ProtoModel,
    pub symbols: Vec<Symbol>,
    pub allow_new_symbol: bool,
}

impl<'mb> ModelBuilder<'mb> {
    pub fn new(
        framework: &'mb Nnef,
        proto_model: &'mb ProtoModel,
        symbols: &SymbolTable,
    ) -> ModelBuilder<'mb> {
        let model = TypedModel { symbol_table: symbols.to_owned(), ..TypedModel::default() };
        ModelBuilder {
            registries: vec!["tract_nnef".into()],
            framework,
            model,
            naming_scopes: vec![],
            scopes: vec![],
            proto_model,
            symbols: vec![],
            allow_new_symbol: false,
        }
    }

    fn translate(&mut self) -> TractResult<()> {
        'ext: for ext in &self.proto_model.doc.extension {
            match &*ext[0].0 {
                "tract_registry" => {
                    if self.framework.registries.iter().any(|reg| reg.id == ext[1]) {
                        self.registries.push(ext[1].clone())
                    } else if let Some(reg) =
                        self.framework.registries.iter().find(|reg| reg.aliases.contains(&ext[1]))
                    {
                        self.registries.push(reg.id.clone())
                    } else {
                        bail!("Registry not found {:?}", &ext[1])
                    }
                }
                "tract_symbol" => {
                    if ext.len() != 2 {
                        bail!("tract_symbol expects symbol: example: \"extension tract_symbol S;\"")
                    }
                    let symbol = self.model.symbol_table.new_with_prefix(&ext[1].0);
                    self.symbols.push(symbol);
                }
                _ => {
                    for reg in &self.framework.registries {
                        for reg_ext in &reg.extensions {
                            match reg_ext(self, ext)? {
                                ControlFlow::Continue(_) => (),
                                ControlFlow::Break(_) => continue 'ext,
                            }
                        }
                    }
                    warn!("Ignore unknown extension {}", ext.iter().map(|i| &i.0).join(" "));
                }
            };
        }
        self.scopes.push(HashMap::new());
        self.wire_body(&self.proto_model.doc.graph_def.body).context("Wiring root graph body")?;
        let vars = self.scopes.pop().unwrap();

        let outputs = self
            .proto_model
            .doc
            .graph_def
            .results
            .iter()
            .map(|s| {
                vars.get(s)
                    .with_context(|| format!("Could not find variable for output named {s:?}"))
            })
            .collect::<TractResult<TVec<&Value>>>()?;

        let outputs = outputs
            .into_iter()
            .map(|s| s.to::<OutletId>(self))
            .collect::<TractResult<TVec<OutletId>>>()?;
        self.model.set_output_outlets(&outputs)?;

        self.parse_properties().context("Parsing properties")?;

        for (ix, name) in self.proto_model.doc.graph_def.results.iter().enumerate() {
            self.model.set_outlet_label(outputs[ix], name.0.to_string())?;
        }

        Ok(())
    }

    pub fn into_typed_model(mut self) -> Result<TypedModel, (TypedModel, TractError)> {
        match self.translate().context("In ModelBuilder::translate") {
            Ok(()) => Ok(self.model),
            Err(e) => Err((self.model, e)),
        }
    }

    fn parse_properties(&mut self) -> TractResult<()> {
        if let Some(properties) = self
            .proto_model
            .doc
            .fragments
            .iter()
            .find(|f| &f.decl.id.0 == "tract_core_properties")
            .and_then(|f| f.body.as_ref())
            .and_then(|body| body.get(0))
        {
            let properties: TVec<(String, Arc<Tensor>)> =
                properties.right.resolve(self, &[])?.to(self)?;
            self.model.properties = properties.into_iter().collect();
        }
        Ok(())
    }

    pub fn wire_body(&mut self, body: &[Assignment]) -> TractResult<()> {
        // todo: can i relax the outlet id constraint ?
        for assignment in body {
            let identifiers = assignment.left.to_identifiers()?;
            let datum_types = identifiers
                .iter()
                .map(|s| {
                    self.proto_model
                        .quantization
                        .as_ref()
                        .and_then(|qm| qm.get(*s).map(|q| q.datum_type()))
                })
                .collect::<Vec<_>>();
            self.naming_scopes.push(identifiers[0].clone());
            let mut values = if identifiers.len() == 1 {
                let value: OutletId = assignment
                    .right
                    .resolve(self, &datum_types)
                    .and_then(|v| v.to(self))
                    .with_context(|| {
                        format!(
                            "Plugging in assignement for {:?}",
                            identifiers.iter().map(|i| &i.0).join(", ")
                        )
                    })?;
                tvec!(value)
            } else {
                let values: TVec<OutletId> = assignment
                    .right
                    .resolve(self, &datum_types)
                    .and_then(|v| v.to(self))
                    .with_context(|| {
                        format!(
                            "Plugging in assignement for {:?}",
                            identifiers.iter().map(|i| &i.0).join(", ")
                        )
                    })?;
                if values.len() != identifiers.len() {
                    bail!(
                        "Assignement for {} received {} value(s).",
                        identifiers.iter().map(|i| &i.0).join(","),
                        values.len()
                    )
                }
                values
            };
            for (qparam, value) in datum_types.into_iter().zip(values.iter_mut()) {
                if let Some(qparam) = qparam {
                    if qparam != self.model.outlet_fact(*value)?.datum_type {
                        self.model.node_mut(value.node).name =
                            format!("{}_raw", self.naming_scopes.iter().map(|i| &i.0).join("_"));
                        *value = self.model.wire_node(
                            "foo",
                            tract_core::ops::cast::cast(qparam),
                            &[*value],
                        )?[0];
                    }
                }
            }
            for (id, outlet) in identifiers.iter().zip(values.iter()) {
                self.scopes.last_mut().unwrap().insert((*id).clone(), Value::Wire(*outlet));
            }
            self.naming_scopes.pop();
            for (value, identifier) in values.iter().zip(identifiers.into_iter()) {
                if self.model.node_mut(value.node).name.is_empty() {
                    self.naming_scopes.push(identifier.clone());
                    self.model.node_mut(value.node).name = self.generate_node_name();
                    self.naming_scopes.pop();
                }
            }
        }
        Ok(())
    }

    pub fn wire_invocation(
        &mut self,
        invocation: &Invocation,
        dt: &[Option<DatumType>],
    ) -> TractResult<Value> {
        for frag in &self.proto_model.doc.fragments {
            if frag.decl.id == invocation.id && frag.body.is_some() {
                let resolved = ResolvedInvocation {
                    invocation,
                    dt_from_quant_file: dt,
                    default_params: &frag.decl.parameters,
                };
                return self.wire_fragment_invocation(
                    &resolved,
                    &frag.decl,
                    frag.body.as_deref().unwrap(),
                );
            }
        }
        for registry in &self.framework.registries {
            if self.registries.contains(&registry.id) {
                if let Some(outputs) = registry
                    .deserialize(self, invocation, dt)
                    .with_context(|| format!("Interrogating registry {:?}", registry.id))?
                {
                    return Ok(outputs);
                }
            }
        }
        bail!("No definition for operator {:?}", invocation.id);
    }

    pub fn wire_fragment_invocation(
        &mut self,
        invocation: &ResolvedInvocation,
        decl: &FragmentDecl,
        body: &[Assignment],
    ) -> TractResult<Value> {
        let mut inner_scope = HashMap::new();
        for par in invocation.default_params.iter() {
            inner_scope.insert(par.id.clone(), invocation.named_arg_as::<Value>(self, &par.id.0)?);
        }
        self.scopes.push(inner_scope);
        self.naming_scopes.push(invocation.invocation.id.clone());
        self.wire_body(body)?;
        self.naming_scopes.pop();
        let inner_scope = self.scopes.pop().unwrap();
        Ok(Value::Tuple(
            decl.results.iter().map(|res| inner_scope.get(&res.id).unwrap()).cloned().collect(),
        ))
    }

    fn generate_node_name(&self) -> String {
        let name = self.naming_scopes.iter().map(|n| &n.0).join("_");
        if self.model.nodes().iter().any(|n| n.name.starts_with(&name)) {
            for i in 0.. {
                let candidate = format!("{name}_{i}");
                if !self.model.nodes().iter().any(|n| n.name.starts_with(&candidate)) {
                    return candidate;
                }
            }
        }
        name
    }

    pub fn wire_as_outlets(
        &mut self,
        op: impl Into<Box<dyn TypedOp>>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let op = op.into();
        let name = self.generate_node_name();
        self.model.wire_node(name, op, inputs).with_context(|| format!("inputs are {inputs:?}"))
    }

    pub fn wire(
        &mut self,
        op: impl Into<Box<dyn TypedOp>>,
        inputs: &[OutletId],
    ) -> TractResult<Value> {
        self.wire_as_outlets(op, inputs).map(Value::from)
    }
}

#[derive(Clone)]
pub struct ResolvedInvocation<'a> {
    pub invocation: &'a Invocation,
    pub dt_from_quant_file: &'a [Option<DatumType>],
    pub default_params: &'a [Parameter],
}

impl<'a> ResolvedInvocation<'a> {
    pub fn named_arg_as<T>(&self, builder: &mut ModelBuilder, name: &str) -> TractResult<T>
    where
        T: CoerceFrom<Value>,
    {
        let rv = self.named_arg(name)?;
        let v = rv
            .resolve(builder, &[])
            .with_context(|| format!("Resolving argument `{name}' ({rv:?})"))?;
        v.to::<T>(builder).with_context(|| format!("Converting argument `{name}' from {v:?}"))
    }

    pub fn named_arg(&self, name: &str) -> TractResult<Cow<RValue>> {
        self.get_named_arg(name).ok_or_else(|| format_err!("expected argument {}", name))
    }

    pub fn get_named_arg(&self, name: &str) -> Option<Cow<RValue>> {
        // first look explicit name in invocation arguments
        if let Some(arg) = self
            .invocation
            .arguments
            .iter()
            .find(|arg| arg.id.as_ref().map(|i| &*i.0) == Some(name))
        {
            return Some(Cow::Borrowed(&arg.rvalue));
        }
        // then use fragment prototype:
        if let Some((ix, param)) =
            self.default_params.iter().enumerate().find(|(_ix, param)| &*param.id.0 == name)
        {
            // check that all previous (and our) arguments are positional (todo:
            // valid args when building augmented_invocation)
            if self.invocation.arguments.len() > ix
                && self.invocation.arguments.iter().take(ix + 1).all(|arg| arg.id.is_none())
            {
                return Some(Cow::Borrowed(&self.invocation.arguments[ix].rvalue));
            }
            if let Some(rv) = &param.lit {
                return Some(Cow::Owned(RValue::Literal(rv.clone())));
            }
        }
        None
    }

    pub fn get_named_arg_as<T>(
        &self,
        builder: &mut ModelBuilder,
        name: &str,
    ) -> TractResult<Option<T>>
    where
        T: CoerceFrom<Value>,
    {
        let Some(rv) = self.get_named_arg(name) else { return Ok(None) };
        let v = rv
            .resolve(builder, &[])
            .with_context(|| format!("Resolving argument `{name}' ({rv:?})"))?;
        v.to::<T>(builder)
            .with_context(|| format!("Converting argument `{name}' from {v:?}"))
            .map(Some)
    }
}

impl<'mb> ModelBuilder<'mb> {}

impl LValue {
    fn to_identifier(&self) -> TractResult<&Identifier> {
        match self {
            LValue::Identifier(id) => Ok(id),
            _ => bail!("Expected an identifier, found a tuple: {:?}", self),
        }
    }

    #[allow(dead_code)]
    fn to_identifiers(&self) -> TractResult<TVec<&Identifier>> {
        match self {
            LValue::Identifier(_) => Ok(tvec!(self.to_identifier()?)),
            LValue::Tuple(ids) => ids.iter().map(|id| id.to_identifier()).collect(),
            LValue::Array(ids) => ids.iter().map(|id| id.to_identifier()).collect(),
        }
    }
}

impl Invocation {}

impl RValue {
    pub fn resolve(
        &self,
        builder: &mut ModelBuilder,
        dt: &[Option<DatumType>],
    ) -> TractResult<Value> {
        match self {
            RValue::Identifier(id) => {
                if let Some(mut outlet) = builder.scopes.last().unwrap().get(id).cloned() {
                    if let Value::Wire(outlet_id) = outlet {
                        let out_dt = builder.model.node(outlet_id.node).outputs[outlet_id.slot]
                            .fact
                            .datum_type;
                        if let Some(Some(dt)) = dt.get(0) {
                            if out_dt.unquantized() != dt.unquantized() {
                                return Err(format_err!(
                                    "Mismatched types expected {:?}, got {:?}",
                                    dt,
                                    out_dt
                                ));
                            }
                            if out_dt != *dt {
                                outlet =
                                    builder.wire(tract_core::ops::cast::cast(*dt), &[outlet_id])?;
                            }
                        }
                    }
                    Ok(outlet)
                } else if builder.allow_new_symbol {
                    let sym = builder.model.symbol_table.sym(&id.0);
                    Ok(Value::Dim(sym.into()))
                } else {
                    bail!("Can not resolve {:?}. Not a known identifier, and symbol introduction is forbiddent out of \"external\" shape field", id);
                }
            }
            RValue::Invocation(inv) => builder
                .wire_invocation(inv, dt)
                .with_context(|| format!("Resolving invocation {:?}", inv.id)),
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
                    id: op.into(),
                    generic_type_name: None,
                    arguments: vec![
                        Argument { id: None, rvalue: left.as_ref().clone() },
                        Argument { id: None, rvalue: right.as_ref().clone() },
                    ],
                };
                builder
                    .wire_invocation(&inv, dt)
                    .with_context(|| format!("Resolving invocation {:?}", &inv.id))
            }
            RValue::Array(array) => Ok(Value::Array(
                array
                    .iter()
                    .zip(std::iter::repeat(&dt.get(0).copied().flatten()))
                    .map(|(i, dt)| i.resolve(builder, &[*dt]))
                    .collect::<TractResult<_>>()?,
            )),
            RValue::Tuple(array) => {
                let dt_iter: Box<dyn Iterator<Item = &Option<DatumType>>> =
                    if dt.len() == 0 || dt.len() == 1 && dt[0].is_none() {
                        Box::new(std::iter::repeat(&None))
                    } else if dt.len() == array.len() {
                        Box::new(dt.iter())
                    } else {
                        bail!("Wrong number of types for a tuple, got {:?} for {:?}", dt, array)
                    };
                Ok(Value::Tuple(
                    array
                        .iter()
                        .zip(dt_iter)
                        .map(|(i, dt)| {
                            if dt.is_none() {
                                i.resolve(builder, &[])
                            } else {
                                i.resolve(builder, &[*dt])
                            }
                        })
                        .collect::<TractResult<_>>()?,
                ))
            }
            RValue::Literal(Literal::Numeric(f)) => {
                if f.contains('.') || f.contains('e') {
                    f.parse::<f32>()
                        .map(Value::Scalar)
                        .with_context(|| format!("Can not parse {f} as f32"))
                } else if let Ok(i) = f.parse::<i64>() {
                    Ok(Value::Dim(i.into()))
                } else if let Some(s) = builder.model.symbol_table.get(f) {
                    Ok(Value::Dim(s.into()))
                } else {
                    bail!("Can not parse {}", f)
                }
            }
            RValue::Literal(Literal::String(s)) => Ok(Value::String(s.clone())),
            RValue::Literal(Literal::Logical(s)) => Ok(Value::Bool(*s)),
            RValue::Literal(Literal::Array(array)) => Ok(Value::Array(
                array
                    .iter()
                    .zip(std::iter::repeat(&dt.get(0).copied().flatten()))
                    .map(|(i, dt)| RValue::Literal(i.clone()).resolve(builder, &[*dt]))
                    .collect::<TractResult<_>>()?,
            )),
            _ => panic!("{self:?}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Tensor(Arc<Tensor>),
    Wire(OutletId),
    Array(Vec<Value>),
    Tuple(Vec<Value>),
    String(String),
    Bool(bool),
    Scalar(f32),
    Dim(TDim),
}

impl Value {
    pub fn to<T>(&self, builder: &mut ModelBuilder) -> TractResult<T>
    where
        T: CoerceFrom<Value>,
    {
        T::coerce(builder, self)
    }
}

impl From<TVec<OutletId>> for Value {
    fn from(outled_ids: TVec<OutletId>) -> Self {
        Self::Tuple(outled_ids.into_iter().map(Self::Wire).collect())
    }
}

pub trait CoerceFrom<F> {
    fn coerce(builder: &mut ModelBuilder, from: &F) -> TractResult<Self>
    where
        Self: Sized;
}

impl CoerceFrom<Value> for Value {
    fn coerce(_builder: &mut ModelBuilder, from: &Value) -> TractResult<Self> {
        Ok(from.clone())
    }
}

impl CoerceFrom<Value> for Arc<Tensor> {
    fn coerce(builder: &mut ModelBuilder, from: &Value) -> TractResult<Self> {
        match from {
            Value::Dim(t) => Ok(rctensor0(t.to_i32()?)),
            Value::Tensor(t) => Ok(t.clone()),
            Value::Tuple(t) if t.len() == 1 => t[0].to(builder),
            Value::Scalar(f) => Ok(rctensor0(*f)),
            Value::String(f) => Ok(rctensor0(f.clone())),
            Value::Wire(o) => builder
                .model
                .outlet_fact(*o)?
                .konst
                .clone()
                .ok_or_else(|| format_err!("Not a const")),
            Value::Array(items) => {
                let mut tensors = vec![];
                for item in items {
                    let tensor = Arc::<Tensor>::coerce(builder, item)?;
                    let mut tensor = tensor.into_tensor();
                    tensor.insert_axis(0)?;
                    tensors.push(tensor);
                }
                let tensor = Tensor::stack_tensors(0, &tensors)?;
                Ok(tensor.into_arc_tensor())
            }
            _ => bail!("Can not build a tensor from {:?}", from),
        }
    }
}

impl CoerceFrom<Value> for (Arc<Tensor>, DatumType) {
    fn coerce(builder: &mut ModelBuilder, from: &Value) -> TractResult<Self> {
        match from {
            Value::Tensor(t) => Ok((t.clone(), t.datum_type())),
            Value::Scalar(f) => Ok((rctensor0(*f), DatumType::F32)),
            Value::String(f) => Ok((rctensor0(f.clone()), DatumType::String)),
            Value::Wire(o) => {
                let outlet_fact = builder.model.outlet_fact(*o)?;
                Ok((
                    outlet_fact.konst.clone().ok_or_else(|| format_err!("Not a const"))?,
                    outlet_fact.datum_type,
                ))
            }
            _ => bail!("Can not build a tensor from {:?}", from),
        }
    }
}

impl CoerceFrom<Value> for OutletId {
    fn coerce(builder: &mut ModelBuilder, from: &Value) -> TractResult<Self> {
        match from {
            Value::Tensor(t) => {
                Ok(builder.wire_as_outlets(tract_core::ops::konst::Const::new(t.clone()), &[])?[0])
            }
            Value::Scalar(f) => {
                Ok(builder
                    .wire_as_outlets(tract_core::ops::konst::Const::new(rctensor0(*f)), &[])?[0])
            }
            Value::Dim(i) => Ok(builder
                .wire_as_outlets(tract_core::ops::konst::Const::new(rctensor0(i.clone())), &[])?
                [0]),
            Value::Wire(outlet) => Ok(*outlet),
            Value::Tuple(tuple) if tuple.len() == 1 => OutletId::coerce(builder, &tuple[0]),
            Value::Array(_) => {
                let tensor = Arc::<Tensor>::coerce(builder, from)?;
                Ok(builder.wire_as_outlets(tract_core::ops::konst::Const::new(tensor), &[])?[0])
            }
            _ => bail!("Can not build an outletid from {:?}", from),
        }
    }
}

impl CoerceFrom<Value> for u64 {
    fn coerce(builder: &mut ModelBuilder, from: &Value) -> TractResult<Self> {
        match from {
            Value::Dim(d) => Ok(d.to_i64()? as u64),
            Value::Tensor(t) => Ok(t.cast_to_scalar::<u64>()?),
            Value::Wire(_) => Ok(from.to::<Arc<Tensor>>(builder)?.cast_to_scalar::<u64>()?),
            _ => bail!("Can not build a u64 from {:?}", from),
        }
    }
}

impl CoerceFrom<Value> for i64 {
    fn coerce(builder: &mut ModelBuilder, from: &Value) -> TractResult<Self> {
        match from {
            Value::Dim(d) => d.to_i64(),
            Value::Tensor(t) => Ok(*t.to_scalar::<i64>()?),
            Value::Wire(_) => Ok(from.to::<Arc<Tensor>>(builder)?.cast_to_scalar::<i64>()?),
            _ => bail!("Can not build a i64 from {:?}", from),
        }
    }
}

impl CoerceFrom<Value> for TDim {
    fn coerce(builder: &mut ModelBuilder, from: &Value) -> TractResult<Self> {
        match from {
            Value::Dim(d) => Ok(d.clone()),
            Value::Tensor(t) => Ok(t.to_scalar::<TDim>()?.clone()),
            Value::Wire(_) => {
                Ok(from.to::<Arc<Tensor>>(builder)?.cast_to::<TDim>()?.to_scalar::<TDim>()?.clone())
            }
            _ => bail!("Can not build a TDim from {:?}", from),
        }
    }
}

impl CoerceFrom<Value> for String {
    fn coerce(_builder: &mut ModelBuilder, from: &Value) -> TractResult<Self> {
        match from {
            Value::String(s) => Ok(s.to_string()),
            Value::Tensor(t) => Ok(t.to_scalar::<String>()?.clone()),
            _ => bail!("Can not build a String from {:?}", from),
        }
    }
}

impl CoerceFrom<Value> for bool {
    fn coerce(_builder: &mut ModelBuilder, from: &Value) -> TractResult<Self> {
        if let Value::Bool(b) = from {
            Ok(*b)
        } else {
            bail!("Can not build a boolean from {:?}", from)
        }
    }
}

impl CoerceFrom<Value> for usize {
    fn coerce(builder: &mut ModelBuilder, from: &Value) -> TractResult<Self> {
        Ok(i64::coerce(builder, from)? as usize)
    }
}

impl CoerceFrom<Value> for isize {
    fn coerce(builder: &mut ModelBuilder, from: &Value) -> TractResult<Self> {
        Ok(i64::coerce(builder, from)? as isize)
    }
}

impl CoerceFrom<Value> for f32 {
    fn coerce(builder: &mut ModelBuilder, from: &Value) -> TractResult<Self> {
        match from {
            Value::Scalar(f) => Ok(*f),
            Value::Tensor(t) => Ok(*t.to_scalar::<f32>()?),
            Value::Wire(_) => {
                Ok(*from.to::<Arc<Tensor>>(builder)?.cast_to::<f32>()?.to_scalar::<f32>()?)
            }
            _ => bail!("Can not build a f32 from {:?}", from),
        }
    }
}

impl<D: CoerceFrom<Value>> CoerceFrom<Value> for TVec<D> {
    fn coerce(builder: &mut ModelBuilder, from: &Value) -> TractResult<Self> {
        match from {
            Value::Array(vec) => vec.iter().map(|item| D::coerce(builder, item)).collect(),
            Value::Tuple(vec) => vec.iter().map(|item| D::coerce(builder, item)).collect(),
            any => Ok(tvec!(D::coerce(builder, any)?)),
        }
    }
}

macro_rules! tuple {
    ($($d: ident),*) => {
        impl<$($d),*> CoerceFrom<Value> for ($($d),*)
            where
                $($d: CoerceFrom<Value>),*
                {
                    fn coerce(builder: &mut ModelBuilder, from: &Value) -> TractResult<Self> {
                        match from {
                            Value::Tuple(vec) => {
                                let mut vec = vec.iter();
                                Ok((
                                        $($d::coerce(builder, vec.next().context("Too small a tuple")?)?),*
                                   ))
                            }
                            _ => bail!("Can not build a tuple from {:?}", from),
                        }
                    }
                }
    }
}

tuple!(D1, D2);
tuple!(D1, D2, D3);
tuple!(D1, D2, D3, D4);
