use tract_core::internal::*;
use tract_core::ops;

use crate::ast::*;
use crate::model::ProtoModel;

use std::any::TypeId;

pub fn to_proto_model(model: &TypedModel) -> TractResult<ProtoModel> {
    let mut into_ast = IntoAst::new(model);
    for input in model.input_outlets()? {
        let left = model.node(input.node).name.to_owned();
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
        let name = model.node(o.node).name.to_owned();
        into_ast.assignment(&name, into_ast.mapping[&o].clone());
        into_ast.results.push(name);
    }
    into_ast.into_proto_model()
}

pub struct IntoAst<'a> {
    pub model: &'a TypedModel,
    pub parameters: Vec<String>,
    pub results: Vec<String>,
    pub mapping: HashMap<OutletId, Arc<RValue>>,
    pub tensors: HashMap<String, Arc<Tensor>>,
    pub body: Vec<Assignment>,
    pub registry: HashMap<TypeId, crate::ops::ser::OpDumper>,
}

impl<'a> IntoAst<'a> {
    fn new(model: &'a TypedModel) -> IntoAst {
        IntoAst {
            model,
            parameters: vec![],
            results: vec![],
            mapping: Default::default(),
            tensors: Default::default(),
            body: vec![],
            registry: crate::ops::ser::registry(),
        }
    }

    fn into_proto_model(self) -> TractResult<ProtoModel> {
        let IntoAst { body, tensors, parameters, results, .. } = self;
        let doc = Document {
            version: "1.0".into(),
            extension: vec![],
            fragments: vec![],
            graph_def: GraphDef { id: "network".into(), parameters, results, body },
        };
        Ok(ProtoModel { doc, tensors })
    }

    fn node(&mut self, node: &TypedNode) -> TractResult<Arc<RValue>> {
        let dumper = self
            .registry
            .get(&node.op().type_id())
            .ok_or_else(|| format!("No serializer registered for {:?}", node.op()))?;
        let outputs = dumper(self, node)?;
        /*
        let outputs = if let Some(op) = node.op().downcast_ref::<ops::cnn::conv::ConvUnary>() {
        conv(self, &node, op)?
        } else if let Some(op) = node.op().downcast_ref::<ops::matmul::MatMulUnary>() {
        self.matmul(&node, op)?
        } else if let Some(op) = node.op().downcast_ref::<ops::binary::UnaryOp>() {
        let a = self.konst(format!("{}.a", node.name), &op.a);
        let b = self.mapping[&node.inputs[0]].clone();
        if let Some(_) = op.mini_op.downcast_ref::<ops::math::Add>() {
        invoke("add", &[a, b], &[])
        } else if let Some(_) = op.mini_op.downcast_ref::<ops::math::Max>() {
        invoke("max", &[a, b], &[])
        } else {
        panic!()
        }
        } else {
        panic!("{:?}", node);
        };
        dbg!(&outputs);
        */
        self.mapping.insert(node.id.into(), outputs.clone());
        Ok(outputs)
    }

    pub fn force_assign(&mut self, name: impl Into<String>, exp: &Arc<RValue>) -> Arc<RValue> {
        if let RValue::Identifier(_) = exp.as_ref() {
            exp.clone()
        } else {
            let name = name.into();
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
            self.assignment(
                &name,
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
            RValue::Identifier(name).into()
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
