use std::ops::ControlFlow;

use crate::internal::*;

use crate::ast;
use crate::deser::Value;

use tract_core::dyn_clone::clone_box;
use tract_core::ops::binary::*;

pub type ToTract = fn(&mut ModelBuilder, &ResolvedInvocation) -> TractResult<TVec<OutletId>>;
pub type FromTract = fn(&mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>>;

pub struct Registry {
    pub id: String,
    pub aliases: Vec<String>,
    pub fragments: HashMap<String, FragmentDef>,
    pub primitives: HashMap<String, (Vec<ast::Parameter>, ToTract)>,
    pub unit_element_wise_ops: Vec<(String, Box<dyn ElementWiseMiniOp>)>,
    pub element_wise_ops: Vec<(String, TypeId, FromTract, Vec<ast::Parameter>, ToTract)>,
    pub binary_ops: Vec<(String, Box<dyn BinMiniOp>, Option<Box<dyn BinMiniOp>>)>,
    pub from_tract: HashMap<TypeId, FromTract>,
    pub extensions: Vec<
        Box<
            dyn Fn(&mut crate::deser::ModelBuilder, &[String]) -> TractResult<ControlFlow<(), ()>>
                + Send
                + Sync,
        >,
    >,
}

impl Registry {
    pub fn new(id: impl Into<String>) -> Registry {
        Registry {
            id: id.into(),
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

    pub fn register_dumper(&mut self, id: TypeId, func: FromTract) {
        self.from_tract.insert(id, func);
    }

    pub fn register_primitive(&mut self, id: &str, decl: &[ast::Parameter], func: ToTract) {
        self.primitives.insert(id.to_string(), (decl.to_vec(), func));
    }

    pub fn register_fragment(&mut self, def: FragmentDef) {
        self.fragments.insert(def.decl.id.to_string(), def);
    }

    pub fn register_unit_element_wise(
        &mut self,
        id: impl Into<String>,
        ew: &dyn ElementWiseMiniOp,
    ) {
        assert!(std::mem::size_of_val(ew) == 0);
        self.unit_element_wise_ops.push((id.into(), clone_box(ew)));
    }

    pub fn register_element_wise(
        &mut self,
        id: impl Into<String>,
        type_id: TypeId,
        dumper: FromTract,
        parameters: Vec<ast::Parameter>,
        loader: ToTract,
    ) {
        self.element_wise_ops.push((id.into(), type_id, dumper, parameters, loader));
    }

    pub fn register_binary(&mut self, id: impl Into<String>, op: &dyn BinMiniOp) {
        self.binary_ops.push((id.into(), clone_box(op), None));
    }

    pub fn register_binary_with_flipped(
        &mut self,
        id: impl Into<String>,
        op: &dyn BinMiniOp,
        flipped: &dyn BinMiniOp,
    ) {
        self.binary_ops.push((id.into(), clone_box(op), Some(clone_box(flipped))));
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
                    return Ok(Some(invocation(&*op.0, &[a], &[])));
                }
            } else {
                if let Some(op) = self.element_wise_ops.iter().find(|ew| ew.1 == op.0.type_id()) {
                    if let Some(result) = (op.2)(ast, node)? {
                        return Ok(Some(result));
                    }
                }
            }
        } else if let Some(op) = node.op().downcast_ref::<ops::binary::TypedBinOp>() {
            if let Some(op) =
                self.binary_ops.iter().find(|ew| ew.1.as_ref().type_id() == op.0.type_id())
            {
                let a = ast.mapping[&node.inputs[0]].clone();
                let b = ast.mapping[&node.inputs[1]].clone();
                return Ok(Some(invocation(&*op.0, &[a, b], &[])));
            } else if let Some(op) = self
                .binary_ops
                .iter()
                .find(|ew| ew.2.as_ref().map(|op| op.type_id()) == Some(op.0.type_id()))
            {
                let a = ast.mapping[&node.inputs[0]].clone();
                let b = ast.mapping[&node.inputs[1]].clone();
                return Ok(Some(invocation(&*op.0, &[b, a], &[])));
            }
        } else if let Some(unary) = node.op().downcast_ref::<ops::binary::UnaryOp>() {
            if let Some(o) =
                self.binary_ops.iter().find(|bo| bo.1.as_ref().type_id() == unary.mini_op.type_id())
            {
                let a = ast.konst(format!("{}-a", node.name), &unary.a)?;
                let b = ast.mapping[&node.inputs[0]].clone();
                return Ok(Some(invocation(&*o.0, &[a, b], &[])));
            } else if let Some(o) = self
                .binary_ops
                .iter()
                .find(|bo| bo.2.as_ref().map(|op| op.type_id()) == Some(unary.mini_op.type_id()))
            {
                let a = ast.konst(format!("{}-a", node.name), &unary.a)?;
                let b = ast.mapping[&node.inputs[0]].clone();
                return Ok(Some(invocation(&*o.0, &[b, a], &[])));
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
            let resolved =
                ResolvedInvocation { invocation, default_params: &*op.0, dt_from_quant_file: dt };
            let outlets = (op.1)(builder, &resolved)
                .with_context(|| format!("Deserializing op `{}'", invocation.id))?;
            return Ok(Some(Value::Tuple(outlets.into_iter().map(Value::Wire).collect())));
        }
        if let Some(ew) = self.unit_element_wise_ops.iter().find(|ew| ew.0 == invocation.id) {
            let input =
                invocation.arguments[0].rvalue.resolve(builder, &[])?.to::<OutletId>(builder)?;
            let outlet = builder
                .wire(tract_core::ops::element_wise::ElementWiseOp(ew.1.clone()), &[input])?;
            if let Some(Some(assumed_out_dt)) = dt.get(0) {
                let out_dt = builder.model.outlet_fact(outlet[0])?.datum_type;
                if out_dt != *assumed_out_dt {
                    return Ok(Some(Value::Wire(
                        builder.wire(tract_core::ops::cast::cast(*assumed_out_dt), &outlet)?[0],
                    )));
                }
            }
            return Ok(Some(Value::Wire(outlet[0])));
        }
        if let Some(ew) = self.element_wise_ops.iter().find(|ew| ew.0 == invocation.id) {
            let resolved =
                ResolvedInvocation { invocation, default_params: &ew.3, dt_from_quant_file: dt };
            return Ok(Some(Value::Wire(
                (ew.4)(builder, &resolved)
                    .with_context(|| format!("Deserializing op `{}'", invocation.id))?[0],
            )));
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
                    a = builder.wire(tract_core::ops::cast::cast(b_dt), &[a])?[0];
                } else {
                    b = builder.wire(tract_core::ops::cast::cast(a_dt), &[b])?[0];
                };
            }
            let inputs = multicast(builder, &[a, b])?;
            let mut wire =
                builder.wire(tract_core::ops::binary::TypedBinOp(bin.1.clone()), &inputs)?[0];
            if let Some(Some(out_dt)) = dt.get(0) {
                if out_dt != &a_dt {
                    wire = builder.wire(tract_core::ops::cast::cast(*out_dt), &[wire])?[0];
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
            (r..max_rank).try_fold(i, |w, n| Ok(builder.wire(AxisOp::Add(n), &[w])?[0]))
        })
        .collect()
}
