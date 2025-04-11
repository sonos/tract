use crate::ast::*;
use crate::internal::*;
use tract_core::ndarray::ArrayViewD;
use tract_core::ndarray::Axis;
use tract_itertools::Itertools;
use tract_linalg::block_quant::BlockQuantValue;

pub fn rewrite_model(model: &mut TypedModel) -> TractResult<()> {
    model.prop_consts()?;
    tract_core::ops::einsum::prefix_matmul::rewrite_einsum_to_prefix_matmul(model)?;
    Rewriter::default()
        .with_rule_for(
            "rewrite_block_quant_const_to_scalar",
            crate::ops::nnef::ser::rewrite_block_quant_const_to_scalar,
        )
        .with_rule_for(
            "rewrite_matmul_to_same_rank",
            crate::ops::nnef::ser::rewrite_matmul_to_same_rank,
        )
        .with_rule_for("rewrite_conv_with_n_axis", tract_core::ops::cnn::rewrite_conv_with_n_axis)
        .with_rule_for(
            "rewrite_deconv_with_n_axis",
            tract_core::ops::cnn::rewrite_deconv_with_n_axis,
        )
        .with_rule_for(
            "rewrite_kernel_conv_in_oihw",
            crate::ops::nnef::ser::rewrite_kernel_conv_in_oihw,
        )
        .with_rule_for(
            "rewrite_kernel_deconv_in_oihw",
            crate::ops::nnef::ser::rewrite_kernel_deconv_in_oihw,
        )
        .with_rule_for(
            "rewrite_consistent_quantized_conv",
            crate::ops::nnef::ser::rewrite_consistent_quantized_conv,
        )
        .with_rule_for("expand_mean_of_square", tract_core::ops::nn::expand_mean_of_squares)
        .rewrite(&(), model)
}

pub fn to_proto_model(framework: &Nnef, model: &TypedModel) -> TractResult<ProtoModel> {
    let mut fixed_model = model.clone();
    rewrite_model(&mut fixed_model)?;
    let mut into_ast = IntoAst::new(framework, &fixed_model);
    into_ast.translate().context("Translating model to AST")?;
    into_ast.into_proto_model().context("Translating AST to proto model")
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
    pub registries: Vec<Identifier>,
    pub model: &'a TypedModel,
    pub parameters: Vec<Identifier>,
    pub results: Vec<Identifier>,
    pub mapping: HashMap<OutletId, Arc<RValue>>,
    pub tensors: HashMap<Identifier, Arc<Tensor>>,
    pub quantization: HashMap<Identifier, QuantFormat>,
    pub resources: HashMap<String, Arc<dyn Resource>>,
    pub fragments: HashMap<Identifier, FragmentDef>,
    pub body: Vec<Assignment>,
}

pub struct RequiredTensorParameter {
    pub parameter_id: Identifier,
    pub label: Identifier,
    pub value: Arc<Tensor>,
}

impl<'a> IntoAst<'a> {
    pub fn new(framework: &'a Nnef, model: &'a TypedModel) -> IntoAst<'a> {
        IntoAst {
            framework,
            registries: Default::default(),
            model,
            parameters: Default::default(),
            results: Default::default(),
            mapping: Default::default(),
            tensors: Default::default(),
            quantization: Default::default(),
            resources: Default::default(),
            fragments: Default::default(),
            body: Default::default(),
            parent: None,
        }
    }

    fn ensure_registry(&mut self, id: &Identifier) -> TractResult<()> {
        if !self.framework.registries.iter().any(|r| &r.id == id) {
            bail!("Registry {} required, consider allowing it on the NNEF framework.", id.0);
        }
        if !self.registries.iter().any(|r| r == id) {
            self.registries.push(id.clone());
        }
        Ok(())
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
            self.node(self.model.node(node))
                .with_context(|| format!("translating node {}", self.model.node(node)))?;
        }
        let outlets: Vec<OutletId> = self.model.output_outlets()?.to_vec();
        for (ix, o) in outlets.into_iter().enumerate() {
            let rv = if let Some(label) = self.model.outlet_label(o) {
                self.force_variable_and_name(label, &self.mapping[&o].clone())
            } else {
                self.force_variable(format!("output_{ix}"), &self.mapping[&o].clone())
            };
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
        let IntoAst { body, mut parameters, results, .. } = self;
        parameters.extend(tensor_params.iter().map(|rtp| rtp.parameter_id.clone()).sorted());
        let body = body
            .into_iter()
            .filter(|assign| match &assign.left {
                LValue::Identifier(id) => !parameters.contains(id),
                _ => true,
            })
            .collect();
        Ok((
            FragmentDef {
                decl: FragmentDecl {
                    id: Identifier("network".into()),
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

    pub fn into_proto_model(mut self) -> TractResult<ProtoModel> {
        let mut properties = self
            .model
            .properties
            .iter()
            .sorted_by_key(|(k, _v)| k.to_owned())
            .map(|(k, v)| Ok(tuple_2(string(k), self.konst(k, v)?.as_ref().clone())))
            .collect::<TractResult<Vec<_>>>()?;
        let version = env!("CARGO_PKG_VERSION");
        properties.push(tuple_2(
            string("tract_nnef_ser_version"),
            self.konst("tract_nnef_ser_version", &rctensor0(version.to_string()))?.as_ref().clone(),
        ));
        properties.push(tuple_2(
            string("tract_nnef_format_version"),
            self.konst("tract_nnef_format_version", &rctensor0("beta1".to_string()))?
                .as_ref()
                .clone(),
        ));
        let properties: Assignment = assignment("properties", Arc::new(array(properties)));
        let IntoAst { mut fragments, body, tensors, parameters, results, .. } = self;
        let mut extension = vec![];
        self.registries.sort();
        for reg in self.registries {
            if reg.0 != "tract_nnef" {
                extension.push(("tract_registry".into(), reg.0));
            }
        }
        for sym in self.model.symbols.all_symbols() {
            extension.push(("tract_symbol".into(), sym.to_string()));
        }
        let locked = self.model.symbols.0.lock();
        for assert in locked.borrow().all_assertions() {
            extension.push(("tract_assert".into(), assert.to_string()));
        }
        for scenario in locked.borrow().scenarios() {
            for assert in locked.borrow().scenario(scenario) {
                extension.push(("tract_assert".into(), format!("{scenario}: {assert}")));
            }
        }
        let properties = FragmentDef {
            decl: FragmentDecl {
                id: Identifier("tract_core_properties".to_string()),
                generic_decl: None,
                parameters: vec![],
                results: vec![Result_ {
                    id: Identifier("properties".to_string()),
                    spec: TypeSpec::Tuple(vec![TypeName::String.spec(), TypeName::Scalar.tensor()])
                        .array(),
                }],
            },
            body: Some(vec![properties]),
        };
        fragments.insert(properties.decl.id.clone(), properties);
        let doc = Document {
            version: "1.0".into(),
            extension,
            fragments: fragments.into_values().collect(),
            graph_def: GraphDef { id: Identifier("network".into()), parameters, results, body },
        };
        let quantization = if self.quantization.len() > 0 { Some(self.quantization) } else { None };
        Ok(ProtoModel { doc, tensors, quantization, resources: self.resources })
    }

    fn node(&mut self, node: &TypedNode) -> TractResult<TVec<Arc<RValue>>> {
        let mut required_registries = Vec::new();
        for reg in &self.framework.registries {
            if let Some(outputs) = reg.serialize(self, node).context("Serializing op")? {
                if self.ensure_registry(&reg.id).is_err() {
                    required_registries.push(&reg.id);
                    continue;
                };
                let scoped = self.scoped_id(&node.name);
                let names: Vec<_> = (0..node.outputs.len())
                    .map(|ix| {
                        if ix > 0 {
                            Identifier(format!("{}_{}", scoped.0, ix))
                        } else {
                            scoped.clone()
                        }
                    })
                    .collect();
                if node.outputs.len() > 1 {
                    self.body.push(Assignment {
                        left: LValue::Tuple(
                            names.iter().map(|n| LValue::Identifier(n.clone())).collect(),
                        ),
                        right: outputs.as_ref().clone(),
                    });
                } else {
                    self.assignment(names[0].clone(), outputs);
                };

                for (outlet, name) in node.outputs.iter().zip(names.iter()) {
                    if let Some(qf) = QuantFormat::from_dt(outlet.fact.datum_type) {
                        self.quantization.insert(name.clone(), qf);
                    }
                }

                let mut outputs = tvec!();
                for (ix, o) in names.into_iter().enumerate() {
                    let rv = Arc::new(ident(o));
                    self.mapping.insert((node.id, ix).into(), rv.clone());
                    outputs.push(rv);
                }

                return Ok(outputs);
            }
        }
        if required_registries.is_empty() {
            bail!("No serializer found for node {}", node);
        } else if required_registries.len() == 1 {
            bail!(
                "Registry {} required, consider allowing it on the NNEF framework.",
                required_registries[0].0
            );
        } else {
            bail!("One of the following registries is required: {:?}, consider allowing one on the NNEF framework.", required_registries);
        }
    }

    pub fn scoped_id(&self, name: impl AsRef<str>) -> Identifier {
        let name = name.as_ref().to_string();
        Identifier(name)
    }

    pub fn force_variable(&mut self, name: impl AsRef<str>, exp: &Arc<RValue>) -> Arc<RValue> {
        if let RValue::Identifier(_) = exp.as_ref() {
            exp.clone()
        } else {
            let name = self.scoped_id(name);
            self.assignment(name.clone(), exp.clone());
            ident(name).into()
        }
    }

    pub fn force_variable_and_name(
        &mut self,
        name: impl Into<String>,
        exp: &Arc<RValue>,
    ) -> Arc<RValue> {
        let name = name.into();
        if let RValue::Identifier(id) = exp.as_ref() {
            if name == id.0 {
                return exp.clone();
            }
        }
        let name = self.scoped_id(name);
        self.assignment(name.clone(), exp.clone());
        ident(name).into()
    }

    pub fn konst(
        &mut self,
        name: impl AsRef<str>,
        tensor: &Arc<Tensor>,
    ) -> TractResult<Arc<RValue>> {
        self.do_konst(name, tensor, false)
    }

    pub fn konst_variable(
        &mut self,
        name: impl AsRef<str>,
        tensor: &Arc<Tensor>,
    ) -> TractResult<Arc<RValue>> {
        self.do_konst(name, tensor, true)
    }

    fn dump_rec_tensor<T: Datum>(
        t: &ArrayViewD<T>,
        el: impl for<'t> Fn(&'t T) -> RValue + Copy,
    ) -> RValue {
        if t.ndim() == 0 {
            el(&t.as_slice().unwrap()[0])
        } else {
            let values: TVec<RValue> = (0..t.shape()[0])
                .map(|i| Self::dump_rec_tensor(&t.index_axis(Axis(0), i), el))
                .collect();
            array(values)
        }
    }

    fn do_konst(
        &mut self,
        name: impl AsRef<str>,
        tensor: &Arc<Tensor>,
        force_variable: bool,
    ) -> TractResult<Arc<RValue>> {
        let mut name: Identifier = name.as_ref().into();
        let have_tract_core = self.ensure_registry(&"tract_core".into()).is_ok();
        if tensor.datum_type() == TDim::datum_type() {
            return Ok(Self::dump_rec_tensor(&tensor.to_array_view::<TDim>()?, tdim).into());
        }
        if !force_variable && tensor.len() <= 8 {
            if tensor.datum_type() == String::datum_type() {
                return Ok(Self::dump_rec_tensor(&tensor.to_array_view::<String>()?, |f| {
                    string(f)
                })
                .into());
            } else if tensor.datum_type() == DatumType::F32 {
                return Ok(
                    Self::dump_rec_tensor(&tensor.to_array_view::<f32>()?, |f| numeric(f)).into()
                );
            } else if have_tract_core && tensor.datum_type() == DatumType::F16 {
                let array =
                    Self::dump_rec_tensor(&tensor.to_array_view::<f16>()?, |f| numeric(f)).into();
                return Ok(invocation("tract_core_cast", &[array], &[("to", string("f16"))]));
            } else if have_tract_core && tensor.datum_type().is_integer() {
                if let Ok(value) = tensor.cast_to::<i64>() {
                    let value =
                        Self::dump_rec_tensor(&value.to_array_view::<i64>().unwrap(), |i| {
                            numeric(i)
                        });
                    let to = string(format!("{:?}", tensor.datum_type()).to_lowercase());
                    return Ok(invocation("tract_core_cast", &[value.into()], &[("to", to)]));
                }
            };
        }

        if self.tensors.contains_key(&name) {
            name = (0..)
                .map(|it| Identifier::from(&*format!("{}_{}", name.0, it)))
                .find(|it| !self.tensors.contains_key(it))
                .unwrap();
        }

        self.tensors.insert(name.clone(), tensor.clone());
        let id = self.scoped_id(&name);
        let shape = if tensor.datum_type().is_opaque() {
            if let Some(bqv) = tensor.to_scalar::<Opaque>()?.downcast_ref::<BlockQuantValue>() {
                bqv.fact.shape()
            } else {
                bail!("Unexpected opaque tensor in serialization {tensor:?}");
            }
        } else {
            tensor.shape()
        };
        self.assignment(
            id.clone(),
            RValue::Invocation(Invocation {
                id: "variable".into(),
                generic_type_name: Some(TypeName::Scalar),
                arguments: vec![
                    named_arg("label", string(name.0)),
                    named_arg("shape", ints(shape)),
                ],
            })
            .into(),
        );
        if let Some(qp) = QuantFormat::from_dt(tensor.datum_type()) {
            self.quantization.insert(id.clone(), qp);
        }
        Ok(ident(id).into())
    }

    fn assignment(&mut self, name: impl AsRef<str>, right: Arc<RValue>) {
        let name = name.as_ref();
        if *right == ident(name) {
            return;
        }
        self.body.push(assignment(name, right))
    }
}

pub fn assignment(name: impl AsRef<str>, right: Arc<RValue>) -> Assignment {
    Assignment { left: LValue::Identifier(name.as_ref().into()), right: right.as_ref().to_owned() }
}

pub fn ints(shape: &[usize]) -> RValue {
    RValue::Array(shape.iter().map(|s| RValue::Literal(Literal::Numeric(s.to_string()))).collect())
}

pub fn tdims(shape: &[TDim]) -> RValue {
    RValue::Array(shape.iter().map(tdim).collect())
}

pub fn tdim(dim: &TDim) -> RValue {
    match dim {
        TDim::Val(x) => numeric(x),
        TDim::Sym(s) => ident(s.to_string()),
        TDim::Add(terms) => terms
            .iter()
            .map(tdim)
            .reduce(|x, y| RValue::Binary(x.boxed(), "+".to_string(), y.boxed()))
            .unwrap(),
        TDim::Mul(terms) => terms
            .iter()
            .map(tdim)
            .reduce(|x, y| RValue::Binary(x.boxed(), "*".to_string(), y.boxed()))
            .unwrap(),
        TDim::MulInt(x, y) => RValue::Binary(numeric(x).boxed(), "*".to_string(), tdim(y).boxed()),
        TDim::Div(x, y) => RValue::Binary(tdim(x).boxed(), "/".to_string(), numeric(y).boxed()),
        TDim::Broadcast(_) => todo!(),
        TDim::Min(_) | TDim::Max(_) => todo!(),
    }
}

pub fn string(s: impl AsRef<str>) -> RValue {
    RValue::Literal(Literal::String(s.as_ref().into()))
}

pub fn datum_type(dt: DatumType) -> RValue {
    string(format!("{:?}", dt.unquantized()).to_lowercase())
}

pub fn logical(b: bool) -> RValue {
    RValue::Literal(Literal::Logical(b))
}

pub fn lident(s: impl AsRef<str>) -> LValue {
    LValue::Identifier(s.as_ref().into())
}

pub fn ident(s: impl AsRef<str>) -> RValue {
    RValue::Identifier(s.as_ref().into())
}

pub fn array(items: impl AsRef<[RValue]>) -> RValue {
    RValue::Array(items.as_ref().to_vec())
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
    RValue::Literal(Literal::Numeric(format!("{num:?}")))
}

pub fn named_arg(id: &str, rv: RValue) -> Argument {
    Argument { id: Some(id.into()), rvalue: rv }
}

pub fn invocation(
    id: impl AsRef<str>,
    positional: &[Arc<RValue>],
    named: &[(&str, RValue)],
) -> Arc<RValue> {
    let arguments = positional
        .iter()
        .map(|rv| Argument { id: None, rvalue: rv.as_ref().clone() })
        .chain(named.iter().map(|(n, v)| named_arg(n, v.clone())))
        .collect();
    RValue::Invocation(Invocation { id: id.as_ref().into(), generic_type_name: None, arguments })
        .into()
}
