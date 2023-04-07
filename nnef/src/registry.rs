use std::ops::ControlFlow;

use crate::ast::Identifier;
use crate::internal::*;

use crate::ast;
use crate::deser::Value;
use crate::ser::ints;

use tract_core::dyn_clone::clone_box;
use tract_core::ops::binary::*;

pub type ToTract = fn(&mut ModelBuilder, &ResolvedInvocation) -> TractResult<Value>;
pub type FromTract = fn(&mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>>;
pub type BinOp = (Identifier, Box<dyn BinMiniOp>);
pub type Extension = Box<
    dyn Fn(&mut crate::deser::ModelBuilder, &[Identifier]) -> TractResult<ControlFlow<(), ()>>
        + Send
        + Sync,
>;

#[derive(Clone)]
pub struct PrimitiveDecl {
    pub decl: FragmentDecl,
    pub docstrings: Option<Vec<String>>,
    pub to_tract: ToTract,
}

impl PrimitiveDecl {
    pub fn validate(&self) -> TractResult<()> {
        self.decl.validate().with_context(|| format!("Invalid primitive `{}'", self.decl.id.0))
    }

    pub fn with_doc(&mut self, docstring: impl Into<String>) -> &mut Self {
        self.docstrings.get_or_insert_with(Vec::new).push(docstring.into());
        self
    }
}

pub struct Registry {
    pub id: Identifier,
    pub docstrings: Option<Vec<String>>,
    pub aliases: Vec<Identifier>,
    pub fragments: HashMap<Identifier, FragmentDef>,
    pub primitives: HashMap<Identifier, PrimitiveDecl>,
    pub unit_element_wise_ops: Vec<(Identifier, Box<dyn ElementWiseMiniOp>)>,
    pub element_wise_ops: Vec<(Identifier, TypeId, FromTract, Vec<ast::Parameter>, ToTract)>,
    pub binary_ops: Vec<BinOp>,
    pub from_tract: HashMap<TypeId, FromTract>,
    pub extensions: Vec<Extension>,
}

impl Registry {
    pub fn new(id: impl AsRef<str>) -> Registry {
        Registry {
            id: id.as_ref().into(),
            docstrings: None,
            aliases: Default::default(),
            primitives: Default::default(),
            fragments: Default::default(),
            from_tract: Default::default(),
            unit_element_wise_ops: Default::default(),
            element_wise_ops: Default::default(),
            binary_ops: Default::default(),
            extensions: Default::default(),
        }
    }

    pub fn with_doc(mut self, docstring: impl Into<String>) -> Registry {
        self.docstrings.get_or_insert_with(Vec::new).push(docstring.into());
        self
    }

    pub fn register_dumper(&mut self, id: TypeId, func: FromTract) {
        self.from_tract.insert(id, func);
    }

    pub fn register_primitive(
        &mut self,
        id: impl AsRef<str>,
        params: &[ast::Parameter],
        results: &[impl Into<ast::Result_> + Clone],
        func: ToTract,
    ) -> &mut PrimitiveDecl {
        let id: Identifier = id.as_ref().into();
        let decl = FragmentDecl {
            id: id.clone(),
            generic_decl: None,
            parameters: params.to_vec(),
            results: results.iter().cloned().map(|it| it.into()).collect(),
        };
        self.primitives
            .insert(id.clone(), PrimitiveDecl { decl, docstrings: None, to_tract: func });
        self.primitives.get_mut(&id).expect("Unexpected empty entry in primitives hashmap")
    }

    pub fn register_fragment(&mut self, def: FragmentDef) {
        self.fragments.insert(def.decl.id.clone(), def);
    }

    pub fn register_unit_element_wise(&mut self, id: impl AsRef<str>, ew: &dyn ElementWiseMiniOp) {
        assert!(std::mem::size_of_val(ew) == 0);
        self.unit_element_wise_ops.push((id.as_ref().into(), clone_box(ew)));
    }

    pub fn register_element_wise(
        &mut self,
        id: impl AsRef<str>,
        type_id: TypeId,
        dumper: FromTract,
        parameters: Vec<ast::Parameter>,
        loader: ToTract,
    ) {
        self.element_wise_ops.push((id.as_ref().into(), type_id, dumper, parameters, loader));
    }

    pub fn register_binary(&mut self, id: impl AsRef<str>, op: &dyn BinMiniOp) {
        self.binary_ops.push((id.as_ref().into(), clone_box(op)));
    }

    pub fn serialize(
        &self,
        ast: &mut IntoAst,
        node: &TypedNode,
    ) -> TractResult<Option<Arc<RValue>>> {
        use tract_core::ops;
        if node.op_is::<ops::identity::Identity>() {
            return Ok(Some(ast.mapping[&node.inputs[0]].clone()));
        } else if let Some(op) = node.op().downcast_ref::<ops::element_wise::ElementWiseOp>() {
            if std::mem::size_of_val(op.0.as_ref()) == 0 {
                if let Some(op) = self
                    .unit_element_wise_ops
                    .iter()
                    .find(|ew| ew.1.as_ref().type_id() == op.0.type_id())
                {
                    let a = ast.mapping[&node.inputs[0]].clone();
                    return Ok(Some(invocation(&op.0, &[a], &[])));
                }
            } else if let Some(op) = self.element_wise_ops.iter().find(|ew| ew.1 == op.0.type_id())
            {
                if let Some(result) = (op.2)(ast, node)? {
                    return Ok(Some(result));
                }
            }
        } else if let Some(bin_op) = node.op().downcast_ref::<ops::binary::TypedBinOp>() {
            if let Some(mini_op_name) = self
                .binary_ops
                .iter()
                .find(|(_name, op)| op.as_ref().type_id() == bin_op.op.type_id())
                .map(|(name, _op)| name)
            {
                let mut a = ast.mapping[&node.inputs[0]].clone();
                let a_rank = ast.model.outlet_fact(node.inputs[0])?.rank();
                a = ser_axes_mapping(a_rank, bin_op.axes.axis_ops_to_canonical(InOut::In(0))?, a)?;
                let mut b = ast.mapping[&node.inputs[1]].clone();
                let b_rank = ast.model.outlet_fact(node.inputs[1])?.rank();
                b = ser_axes_mapping(b_rank, bin_op.axes.axis_ops_to_canonical(InOut::In(1))?, b)?;
                let mut c = invocation(mini_op_name, &[a, b], &[]);
                let c_axes_map = bin_op
                    .axes
                    .axis_ops_to_canonical(InOut::Out(0))?
                    .into_iter()
                    .map(|op| op.recip())
                    .rev()
                    .collect();
                c = ser_axes_mapping(bin_op.axes.iter_all_axes().count(), c_axes_map, c)?;
                return Ok(Some(c));
            }
        } else if let Some(op) = self.from_tract.get(&node.op().type_id()) {
            if let Some(result) = op(ast, node)? {
                return Ok(Some(result));
            }
        }
        Ok(None)
    }

    pub fn deserialize(
        &self,
        builder: &mut ModelBuilder,
        invocation: &ast::Invocation,
        dt: &[Option<DatumType>],
    ) -> TractResult<Option<Value>> {
        if let Some(op) = self.primitives.get(&invocation.id) {
            let resolved = ResolvedInvocation {
                invocation,
                default_params: &op.decl.parameters,
                dt_from_quant_file: dt,
            };
            let out_value = (op.to_tract)(builder, &resolved)
                .with_context(|| format!("Deserializing op `{}'", invocation.id.0))?;
            return Ok(Some(out_value));
        }
        if let Some(ew) = self.unit_element_wise_ops.iter().find(|ew| ew.0 == invocation.id) {
            let input =
                invocation.arguments[0].rvalue.resolve(builder, &[])?.to::<OutletId>(builder)?;
            let outlet = builder.wire_as_outlets(
                tract_core::ops::element_wise::ElementWiseOp(ew.1.clone()),
                &[input],
            )?;
            if let Some(Some(assumed_out_dt)) = dt.get(0) {
                let out_dt = builder.model.outlet_fact(outlet[0])?.datum_type;
                if out_dt != *assumed_out_dt {
                    return Ok(Some(
                        builder.wire(tract_core::ops::cast::cast(*assumed_out_dt), &outlet)?,
                    ));
                }
            }
            return Ok(Some(Value::Wire(outlet[0])));
        }
        if let Some(ew) = self.element_wise_ops.iter().find(|ew| ew.0 == invocation.id) {
            let resolved =
                ResolvedInvocation { invocation, default_params: &ew.3, dt_from_quant_file: dt };
            return Ok(Some(
                (ew.4)(builder, &resolved)
                    .with_context(|| format!("Deserializing op `{}'", invocation.id.0))?,
            ));
        }
        if let Some(bin) = self.binary_ops.iter().find(|bin| bin.0 == invocation.id) {
            let mut a =
                invocation.arguments[0].rvalue.resolve(builder, &[])?.to::<OutletId>(builder)?;
            let mut b =
                invocation.arguments[1].rvalue.resolve(builder, &[])?.to::<OutletId>(builder)?;
            let a_dt = builder.model.outlet_fact(a)?.datum_type;
            let b_dt = builder.model.outlet_fact(b)?.datum_type;

            // mitigation of nnef "scalar" type mismatch with tract-core more
            // strict types
            if a_dt != b_dt {
                if builder.model.node(a.node).op_is::<tract_core::ops::konst::Const>() {
                    a = builder.wire_as_outlets(tract_core::ops::cast::cast(b_dt), &[a])?[0];
                } else {
                    b = builder.wire_as_outlets(tract_core::ops::cast::cast(a_dt), &[b])?[0];
                };
            }
            let inputs = multicast(builder, &[a, b])?;
            let mut wire =
                wire_bin(builder.generate_node_name(), &mut builder.model, bin.1.clone(), &inputs)?
                    [0];
            if let Some(Some(out_dt)) = dt.get(0) {
                if out_dt != &a_dt {
                    wire =
                        builder.wire_as_outlets(tract_core::ops::cast::cast(*out_dt), &[wire])?[0];
                }
            }
            return Ok(Some(Value::Wire(wire)));
        }
        if let Some(frag) = self.fragments.get(&invocation.id) {
            let resolved = ResolvedInvocation {
                invocation,
                default_params: &frag.decl.parameters,
                dt_from_quant_file: dt,
            };
            return Ok(Some(builder.wire_fragment_invocation(
                &resolved,
                &frag.decl,
                frag.body.as_deref().unwrap(),
            )?));
        }
        Ok(None)
    }
}

pub fn multicast(builder: &mut ModelBuilder, inputs: &[OutletId]) -> TractResult<TVec<OutletId>> {
    let ranks = inputs
        .iter()
        .map(|&i| Ok(builder.model.outlet_fact(i)?.rank()))
        .collect::<TractResult<Vec<usize>>>()?;
    let max_rank = ranks.iter().copied().max().unwrap();
    (inputs.iter())
        .zip(ranks.iter())
        .map(|(&i, &r)| {
            (r..max_rank).try_fold(i, |w, n| Ok(builder.wire_as_outlets(AxisOp::Add(n), &[w])?[0]))
        })
        .collect()
}

fn ser_axes_mapping(
    mut rank: usize,
    ops: Vec<AxisOp>,
    mut input: Arc<RValue>,
) -> TractResult<Arc<RValue>> {
    for op in ops {
        input = match op {
            AxisOp::Rm(axis) => {
                rank -= 1;
                invocation("squeeze", &[input], &[("axes", ints(&[axis]))])
            }
            AxisOp::Add(axis) => {
                rank += 1;
                invocation("unsqueeze", &[input], &[("axes", ints(&[axis]))])
            }
            AxisOp::Move(from, to) => {
                let mut perm: TVec<usize> = (0..rank).collect();
                if from < to {
                    perm[from..(to + 1)].rotate_left(1);
                } else {
                    perm[to..(from + 1)].rotate_right(1);
                }
                invocation("transpose", &[input], &[("axes", ints(&perm))])
            }
            _ => unreachable!("AxesMapping traduction do not use reshaep"),
        }
    }
    Ok(input)
}
