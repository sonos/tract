use crate::internal::*;

use crate::ast;
use crate::deser::Value;

use tract_core::ops::binary::*;

pub type ToTract = fn(&mut ModelBuilder, &ResolvedInvocation) -> TractResult<TVec<OutletId>>;
pub type FromTract = fn(&mut IntoAst, node: &TypedNode) -> TractResult<Arc<RValue>>;

pub struct Registry {
    pub id: String,
    pub fragments: HashMap<String, FragmentDef>,
    pub primitives: HashMap<String, (Vec<ast::Parameter>, ToTract)>,
    pub unit_element_wise_ops: Vec<(String, Box<dyn ElementWiseMiniOp>)>,
    pub element_wise_ops: Vec<(String, TypeId, FromTract, Vec<ast::Parameter>, ToTract)>,
    pub binary_ops: Vec<(String, Box<dyn BinMiniOp>)>,
    pub from_tract: HashMap<TypeId, FromTract>,
}

impl Registry {
    pub fn new(id: impl Into<String>) -> Registry {
        Registry {
            id: id.into(),
            primitives: Default::default(),
            fragments: Default::default(),
            from_tract: Default::default(),
            unit_element_wise_ops: Default::default(),
            element_wise_ops: Default::default(),
            binary_ops: Default::default(),
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
        self.unit_element_wise_ops.push((id.into(), tract_core::dyn_clone::clone_box(ew)));
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
        self.binary_ops.push((id.into(), tract_core::dyn_clone::clone_box(op)));
    }

    pub fn serialize(
        &self,
        ast: &mut IntoAst,
        node: &TypedNode,
    ) -> TractResult<Option<Arc<RValue>>> {
        use tract_core::ops;
        if let Some(op) = node.op().downcast_ref::<ops::element_wise::ElementWiseOp>() {
            if std::mem::size_of_val(op.0.as_ref()) == 0 {
                if let Some(op) =
                    self.unit_element_wise_ops.iter().find(|ew| ew.1.as_ref().type_id() == op.0.type_id())
                {
                    let a = ast.mapping[&node.inputs[0]].clone();
                    return Ok(Some(invocation(&*op.0, &[a], &[])));
                }
            } else {
                if let Some(op) =
                    self.element_wise_ops.iter().find(|ew| ew.1 == op.0.type_id())
                {
                    return Ok(Some((op.2)(ast, node)?));
                }
            }
        } else if let Some(op) = node.op().downcast_ref::<ops::binary::TypedBinOp>() {
            if let Some(op) =
                self.binary_ops.iter().find(|ew| ew.1.as_ref().type_id() == op.0.type_id())
            {
                let a = ast.mapping[&node.inputs[0]].clone();
                let b = ast.mapping[&node.inputs[1]].clone();
                return Ok(Some(invocation(&*op.0, &[a, b], &[])));
            }
        } else if let Some(unary) = node.op().downcast_ref::<ops::binary::UnaryOp>() {
            if let Some(o) =
                self.binary_ops.iter().find(|bo| bo.1.as_ref().type_id() == unary.mini_op.type_id())
            {
                let a = ast.konst(format!("{}-a", node.name), &unary.a);
                let b = ast.mapping[&node.inputs[0]].clone();
                return Ok(Some(invocation(&*o.0, &[a, b], &[])));
            }
        } else if let Some(op) = self.from_tract.get(&node.op().type_id()) {
            return Ok(Some(op(ast, node)?));
        }
        Ok(None)
    }

    pub fn deserialize(
        &self,
        builder: &mut ModelBuilder,
        invocation: &ast::Invocation,
    ) -> TractResult<Option<Value>> {
        if let Some(op) = self.primitives.get(&invocation.id) {
            let resolved = ResolvedInvocation { invocation, default_params: &*op.0 };
            let outlets = (op.1)(builder, &resolved)?;
            return Ok(Some(Value::Tuple(outlets.into_iter().map(Value::Wire).collect())));
        }
        if let Some(ew) = self.unit_element_wise_ops.iter().find(|ew| ew.0 == invocation.id) {
            let input = invocation.arguments[0].rvalue.resolve(builder)?.to::<OutletId>(builder)?;
            let outlet = builder
                .wire(tract_core::ops::element_wise::ElementWiseOp(ew.1.clone()), &[input])?;
            return Ok(Some(Value::Wire(outlet[0])));
        }
        if let Some(ew) = self.element_wise_ops.iter().find(|ew| ew.0 == invocation.id) {
            let resolved = ResolvedInvocation { invocation, default_params: &ew.3 };
            return Ok(Some(Value::Wire((ew.4)(builder, &resolved)?[0])))
        }
        if let Some(bin) = self.binary_ops.iter().find(|bin| bin.0 == invocation.id) {
            let a = invocation.arguments[0].rvalue.resolve(builder)?.to::<OutletId>(builder)?;
            let b = invocation.arguments[1].rvalue.resolve(builder)?.to::<OutletId>(builder)?;
            let inputs = multicast(builder, &[a, b])?;
            return Ok(Some(Value::Wire(
                builder.wire(tract_core::ops::binary::TypedBinOp(bin.1.clone()), &inputs)?[0],
            )));
        }
        if let Some(frag) = self.fragments.get(&invocation.id) {
            let resolved = ResolvedInvocation { invocation, default_params: &frag.decl.parameters };
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
