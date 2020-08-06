use tract_core::internal::*;

use crate::ast::*;
use crate::model::ProtoModel;

use std::any::TypeId;

pub fn to_proto_model(model: &TypedModel) -> TractResult<ProtoModel> {
    let names = model
        .nodes()
        .iter()
        .filter(|n| !model.input_outlets().unwrap().contains(&n.id.into()))
        .map(|n| &n.name)
        .collect::<Vec<_>>();
    let prefix:String = names[1..].iter().fold(names[0].to_string(), |prefix, name| {
        (prefix.chars()).zip(name.chars()).take_while(|(a, b)| a == b).map(|(a, _)| a).collect()
    });
    let mut into_ast = IntoAst::new(model, prefix.to_string());
    for input in model.input_outlets()? {
        let left = into_ast.scoped_id(&model.node(input.node).name);
        into_ast.parameters.push(left.clone());
        let input_shape = model.outlet_fact(*input)?.shape.as_finite().ok_or("No dim (yet)")?;
        let right = RValue::Invocation(Invocation {
            id: "external".into(),
            generic_type_name: Some(TypeName::Scalar),
            arguments: vec![Argument { id: Some("shape".into()), rvalue: shape(input_shape) }],
        });
        into_ast.assignment(left.clone(), right.into());
        into_ast.mapping.insert(*input, RValue::Identifier(left).into());
    }
    for node in model.eval_order()? {
        if model.input_outlets()?.iter().any(|io| io.node == node) {
            continue;
        }
        into_ast.node(model.node(node))?;
    }
    for o in model.output_outlets()? {
        let name = into_ast.scoped_id(&model.node(o.node).name);
        into_ast.assignment(&name, into_ast.mapping[&o].clone());
        into_ast.results.push(name);
    }
    into_ast.into_proto_model()
}

pub struct IntoAst<'a> {
    pub prefix: String,
    pub model: &'a TypedModel,
    pub parameters: Vec<String>,
    pub results: Vec<String>,
    pub mapping: HashMap<OutletId, Arc<RValue>>,
    pub tensors: HashMap<String, Arc<Tensor>>,
    pub fragments: HashMap<String, FragmentDef>,
    pub body: Vec<Assignment>,
    pub registry: HashMap<TypeId, crate::ops::ser::OpDumper>,
}

impl<'a> IntoAst<'a> {
    fn new(model: &'a TypedModel, prefix: String) -> IntoAst {
        IntoAst {
            prefix,
            model,
            parameters: vec![],
            results: vec![],
            mapping: Default::default(),
            tensors: Default::default(),
            fragments: Default::default(),
            body: vec![],
            registry: crate::ops::ser::registry(),
        }
    }

    fn into_proto_model(self) -> TractResult<ProtoModel> {
        let IntoAst { prefix, fragments, body, tensors, parameters, results, .. } = self;
        let id = prefix.trim_end_matches(&['-', '/', '.'][..]).replace(&['-', '/', '.'][..], "_");
        let doc = Document {
            version: "1.0".into(),
            extension: vec![],
            fragments: fragments.into_iter().map(|(_,v)| v).collect(),
            graph_def: GraphDef { id, parameters, results, body },
        };
        Ok(ProtoModel { doc, tensors })
    }

    fn node(&mut self, node: &TypedNode) -> TractResult<Arc<RValue>> {
        let dumper = self
            .registry
            .get(&node.op().type_id())
            .ok_or_else(|| format!("No serializer registered for {:?}", node.op()))?;
        let outputs = dumper(self, node)?;
        self.mapping.insert(node.id.into(), outputs.clone());
        Ok(outputs)
    }

    pub fn scoped_id(&self, name: impl Into<String>) -> String {
        let mut name = name.into();
        if name.starts_with(&self.prefix) {
            name = name.chars().skip(self.prefix.len()).collect()
        }
        name.replace("/", "_").replace(".", "_").replace("-", "_").into()
    }

    pub fn force_assign(&mut self, name: impl Into<String>, exp: &Arc<RValue>) -> Arc<RValue> {
        if let RValue::Identifier(_) = exp.as_ref() {
            exp.clone()
        } else {
            let name = self.scoped_id(name);
            self.assignment(name.clone(), exp.clone());
            RValue::Identifier(name).into()
        }
    }

    pub fn konst(&mut self, name: impl Into<String>, tensor: &Arc<Tensor>) -> Arc<RValue> {
        if tensor.is_uniform().unwrap() {
            RValue::Literal(Literal::Numeric(tensor.cast_to_scalar::<f32>().unwrap().to_string()))
                .into()
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
                        named_arg("shape", shape(tensor.shape())),
                    ],
                })
                .into(),
            );
            RValue::Identifier(id).into()
        }
    }

    fn assignment(&mut self, name: impl Into<String>, right: Arc<RValue>) {
        self.body.push(Assignment {
            left: LValue::Identifier(name.into()),
            right: right.as_ref().to_owned(),
        });
    }
}

pub fn shape(shape: &[usize]) -> RValue {
    RValue::Array(shape.iter().map(|s| RValue::Literal(Literal::Numeric(s.to_string()))).collect())
}

pub fn string(s: impl Into<String>) -> RValue {
    RValue::Literal(Literal::String(s.into()))
}

pub fn lident(s: impl Into<String>) -> LValue {
    LValue::Identifier(s.into())
}

pub fn ident(s: impl Into<String>) -> RValue {
    RValue::Identifier(s.into())
}

pub fn numeric<D: std::fmt::Display>(num: D) -> RValue {
    RValue::Literal(Literal::Numeric(num.to_string())).into()
}

pub fn named_arg(id: &str, rv: RValue) -> Argument {
    Argument { id: Some(id.into()), rvalue: rv }
}

pub fn invoke(id: &str, positional: &[Arc<RValue>], named: &[Argument]) -> Arc<RValue> {
    let arguments = positional
        .iter()
        .map(|rv| Argument { id: None, rvalue: rv.as_ref().clone() })
        .chain(named.iter().cloned())
        .collect();
    RValue::Invocation(Invocation { id: id.to_owned(), generic_type_name: None, arguments }).into()
}
