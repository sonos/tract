use crate::ast::*;
use crate::internal::*;
use tract_itertools::Itertools;

pub fn to_proto_model(framework: &Nnef, model: &TypedModel) -> TractResult<ProtoModel> {
    let mut into_ast = IntoAst::new(framework, model);
    into_ast.translate()?;
    into_ast.into_proto_model()
}

pub fn to_fragment_def(
    parent: &IntoAst,
    model: &TypedModel,
) -> TractResult<(FragmentDef, Vec<RequiredTensorParameter>)> {
    let mut into_ast = IntoAst::new(parent.framework, model);
    into_ast.parent = Some(parent);
    into_ast.translate()?;
    into_ast.into_fragment()
}

pub struct IntoAst<'a> {
    pub framework: &'a Nnef,
    pub parent: Option<&'a IntoAst<'a>>,
    pub registries: Vec<String>,
    pub prefix: Option<String>,
    pub model: &'a TypedModel,
    pub parameters: Vec<String>,
    pub results: Vec<String>,
    pub mapping: HashMap<OutletId, Arc<RValue>>,
    pub tensors: HashMap<String, Arc<Tensor>>,
    pub fragments: HashMap<String, FragmentDef>,
    pub body: Vec<Assignment>,
}

pub struct RequiredTensorParameter {
    pub parameter_id: String,
    pub label: String,
    pub value: Arc<Tensor>,
}

impl<'a> IntoAst<'a> {
    pub fn new(framework: &'a Nnef, model: &'a TypedModel) -> IntoAst<'a> {
        let prefix = Self::extract_prefix(model);
        IntoAst {
            framework,
            registries: vec![],
            prefix,
            model,
            parameters: vec![],
            results: vec![],
            mapping: Default::default(),
            tensors: Default::default(),
            fragments: Default::default(),
            body: vec![],
            parent: None,
        }
    }

    fn extract_prefix(model: &TypedModel) -> Option<String> {
        let names = model
            .nodes()
            .iter()
            .filter(|n| !model.input_outlets().unwrap().contains(&n.id.into()))
            .map(|n| &n.name)
            .collect::<Vec<_>>();
        if names.len() > 2 {
            Some(names[1..].iter().fold(names[0].to_string(), |prefix, name| {
                (prefix.chars())
                    .zip(name.chars())
                    .take_while(|(a, b)| a == b)
                    .map(|(a, _)| a)
                    .collect()
            }))
            .filter(|p| p.len() > 0)
        } else {
            None
        }
    }

    fn translate(&mut self) -> TractResult<()> {
        for input in self.model.input_outlets()? {
            let left = self.scoped_id(&self.model.node(input.node).name);
            self.parameters.push(left.clone());
            self.node(self.model.node(input.node))?;
            self.mapping.insert(*input, RValue::Identifier(left).into());
        }
        for node in self.model.eval_order()? {
            if self.model.input_outlets()?.iter().any(|io| io.node == node) {
                continue;
            }
            self.node(self.model.node(node))?;
        }
        let outlets: Vec<OutletId> = self.model.output_outlets()?.to_vec();
        for (ix, o) in outlets.into_iter().enumerate() {
            let rv = self.force_assign(format!("output_{}", ix), &self.mapping[&o].clone());
            if let RValue::Identifier(name) = rv.as_ref() {
                self.results.push(name.clone());
            } else {
                unreachable!()
            };
        }
        Ok(())
    }

    pub fn into_fragment(self) -> TractResult<(FragmentDef, Vec<RequiredTensorParameter>)> {
        let mut tensor_params = vec![];
        for (name, t) in &self.tensors {
            tensor_params.push(RequiredTensorParameter {
                parameter_id: self.scoped_id(name),
                label: name.clone(),
                value: t.clone(),
            })
        }
        let IntoAst { prefix, body, mut parameters, results, .. } = self;
        parameters.extend(tensor_params.iter().map(|rtp| rtp.parameter_id.clone()).sorted());
        let mut id = prefix
            .map(|p| p.trim_end_matches(&['-', '/', '.'][..]).replace(&['-', '/', '.'][..], "_"))
            .unwrap_or("network".into());
        if id.len() > 0 && char::is_digit(id.chars().next().unwrap(), 10) {
            id = "_".to_string() + &id;
        }
        let body = body
            .into_iter()
            .filter(|assign| match &assign.left {
                LValue::Identifier(id) => !parameters.contains(&id),
                _ => true,
            })
            .collect();
        Ok((
            FragmentDef {
                decl: FragmentDecl {
                    id,
                    generic_decl: None,
                    parameters: parameters
                        .into_iter()
                        .map(|s| TypeName::Scalar.tensor().named(s))
                        .collect(),
                    results: results
                        .into_iter()
                        .map(|s| Result_ { id: s, spec: TypeName::Scalar.tensor() })
                        .collect(),
                },
                body: Some(body),
            },
            tensor_params,
        ))
    }

    pub fn into_proto_model(self) -> TractResult<ProtoModel> {
        let IntoAst { prefix, fragments, body, tensors, parameters, results, .. } = self;
        let mut id = prefix
            .map(|p| p.trim_end_matches(&['-', '/', '.'][..]).replace(&['-', '/', '.'][..], "_"))
            .unwrap_or("network".into());
        if id.len() > 0 && char::is_digit(id.chars().next().unwrap(), 10) {
            id = "_".to_string() + &id;
        }
        let mut extension = vec![];
        for reg in self.registries {
            if reg != "tract_nnef" {
                extension.push(vec!["tract_registry".to_string(), reg]);
            }
        }
        let doc = Document {
            version: "1.0".into(),
            extension,
            fragments: fragments.into_iter().map(|(_, v)| v).collect(),
            graph_def: GraphDef { id, parameters, results, body },
        };
        Ok(ProtoModel { doc, tensors })
    }

    fn node(&mut self, node: &TypedNode) -> TractResult<TVec<Arc<RValue>>> {
        for reg in &self.framework.registries {
            if let Some(outputs) = reg.serialize(self, node)? {
                if !self.registries.contains(&reg.id) {
                    self.registries.push(reg.id.clone())
                }
                let scoped = self.scoped_id(&node.name);
                let names: Vec<String> = (0..node.outputs.len())
                    .map(|ix| if ix > 0 { format!("{}_{}", scoped, ix) } else { scoped.clone() })
                    .collect();
                let lvalue = if node.outputs.len() > 1 {
                    LValue::Tuple(names.iter().map(|n| LValue::Identifier(n.clone())).collect())
                } else {
                    LValue::Identifier(names[0].clone())
                };
                self.body.push(Assignment { left: lvalue, right: outputs.as_ref().clone() });
                let mut outputs = tvec!();
                for (ix, o) in names.into_iter().enumerate() {
                    let rv = Arc::new(ident(o));
                    self.mapping.insert((node.id, ix).into(), rv.clone());
                    outputs.push(rv);
                }
                return Ok(outputs);
            }
        }
        bail!("No serializer found for node {}", node);
    }

    pub fn scoped_id(&self, name: impl Into<String>) -> String {
        let mut name = name.into();
        if let Some(p) = &self.prefix {
            if name.starts_with(p) && &*name != p {
                name = name.chars().skip(p.len()).collect()
            }
        }
        Self::sanitize(name)
    }

    pub fn sanitize(name: impl Into<String>) -> String {
        let mut name = name.into();
        if name.len() > 0 && char::is_digit(name.chars().next().unwrap(), 10) {
            name = "_".to_string() + &name;
        }
        name.replace("/", "_").replace(".", "_").replace("-", "_").into()
    }

    pub fn force_assign(&mut self, name: impl Into<String>, exp: &Arc<RValue>) -> Arc<RValue> {
        if let RValue::Identifier(_) = exp.as_ref() {
            exp.clone()
        } else {
            let name = self.scoped_id(name);
            self.assignment(name.clone(), exp.clone());
            ident(name).into()
        }
    }

    pub fn konst(&mut self, name: impl Into<String>, tensor: &Arc<Tensor>) -> Arc<RValue> {
        self.do_konst(name, tensor, false)
    }

    pub fn konst_variable(&mut self, name: impl Into<String>, tensor: &Arc<Tensor>) -> Arc<RValue> {
        self.do_konst(name, tensor, true)
    }

    fn do_konst(
        &mut self,
        name: impl Into<String>,
        tensor: &Arc<Tensor>,
        force_variable: bool,
    ) -> Arc<RValue> {
        if !force_variable && tensor.is_uniform().unwrap() {
            numeric(tensor.cast_to_scalar::<f32>().unwrap()).into()
        } else {
            let name = name.into();
            self.tensors.insert(name.clone(), tensor.clone());
            let id = self.scoped_id(&name);
            self.assignment(
                &id,
                RValue::Invocation(Invocation {
                    id: "variable".to_string(),
                    generic_type_name: Some(TypeName::Scalar),
                    arguments: vec![
                        named_arg("label", string(&name)),
                        named_arg("shape", ints(tensor.shape())),
                    ],
                })
                .into(),
            );
            ident(id).into()
        }
    }

    fn assignment(&mut self, name: impl Into<String>, right: Arc<RValue>) {
        self.body.push(assignment(name, right))
    }
}

pub fn assignment(name: impl Into<String>, right: Arc<RValue>) -> Assignment {
    Assignment { left: LValue::Identifier(name.into()), right: right.as_ref().to_owned() }
}

pub fn ints(shape: &[usize]) -> RValue {
    RValue::Array(shape.iter().map(|s| RValue::Literal(Literal::Numeric(s.to_string()))).collect())
}

pub fn string(s: impl Into<String>) -> RValue {
    RValue::Literal(Literal::String(s.into()))
}

pub fn logical(b: bool) -> RValue {
    RValue::Literal(Literal::Logical(b))
}

pub fn lident(s: impl Into<String>) -> LValue {
    LValue::Identifier(s.into())
}

pub fn ident(s: impl Into<String>) -> RValue {
    RValue::Identifier(s.into())
}

pub fn array(items: impl AsRef<[RValue]>) -> RValue {
    RValue::Array(items.as_ref().iter().cloned().collect())
}

pub fn tuple_2(a: RValue, b: RValue) -> RValue {
    RValue::Tuple(vec![a, b])
}

pub fn tuple_3(a: RValue, b: RValue, c: RValue) -> RValue {
    RValue::Tuple(vec![a, b, c])
}

pub fn tuple_4(a: RValue, b: RValue, c: RValue, d: RValue) -> RValue {
    RValue::Tuple(vec![a, b, c, d])
}

pub fn numeric<D: std::fmt::Debug>(num: D) -> RValue {
    RValue::Literal(Literal::Numeric(format!("{:?}", num))).into()
}

pub fn named_arg(id: &str, rv: RValue) -> Argument {
    Argument { id: Some(id.into()), rvalue: rv }
}

pub fn invocation(id: &str, positional: &[Arc<RValue>], named: &[(&str, RValue)]) -> Arc<RValue> {
    let arguments = positional
        .iter()
        .map(|rv| Argument { id: None, rvalue: rv.as_ref().clone() })
        .chain(named.iter().map(|(n, v)| named_arg(n, v.clone())))
        .collect();
    RValue::Invocation(Invocation { id: id.to_owned(), generic_type_name: None, arguments }).into()
}
