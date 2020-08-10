use crate::internal::*;

mod core;
pub mod deser;
pub mod ser;

use crate::ast;
use crate::deser::Value;

pub fn stdlib() -> Vec<FragmentDef> {
    crate::ast::parse::parse_fragments(include_str!("../../stdlib.nnef")).unwrap()
}

pub type ToTract = fn(&mut ModelBuilder, &ResolvedInvocation) -> TractResult<TVec<OutletId>>;
pub type FromTract = fn(&mut IntoAst, node: &TypedNode) -> TractResult<Arc<RValue>>;

pub struct Registry {
    pub id: String,
    fragments: HashMap<String, FragmentDef>,
    primitives: HashMap<String, (Vec<ast::Parameter>, ToTract)>,
    element_wise_ops: Vec<(String, Box<dyn ElementWiseMiniOp>)>,
    from_tract: HashMap<TypeId, FromTract>,
}

impl Registry {
    pub fn new(id: impl Into<String>) -> Registry {
        Registry {
            id: id.into(),
            primitives: Default::default(),
            fragments: Default::default(),
            from_tract: Default::default(),
            element_wise_ops: Default::default(),
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

    pub fn register_element_wise(&mut self, id: impl Into<String>, ew: &dyn ElementWiseMiniOp) {
        assert!(std::mem::size_of_val(ew) == 0);
        self.element_wise_ops.push((id.into(), tract_core::dyn_clone::clone_box(ew)));
    }

    pub fn serialize(
        &self,
        ast: &mut IntoAst,
        node: &TypedNode,
    ) -> TractResult<Option<Arc<RValue>>> {
        use tract_core::ops;
        if let Some(op) = node.op().downcast_ref::<ops::element_wise::ElementWiseOp>() {
            if let Some(op) =
                self.element_wise_ops.iter().find(|ew| ew.1.as_ref().type_id() == op.0.type_id())
            {
                let a = ast.mapping[&node.inputs[0]].clone();
                return Ok(Some(invocation(&*op.0, &[a], &[])));
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
        if let Some(ew) = self.element_wise_ops.iter().find(|ew| ew.0 == invocation.id) {
            let input = invocation.arguments[0].rvalue.resolve(builder)?.to::<OutletId>(builder)?;
            let outlet = builder
                .wire(tract_core::ops::element_wise::ElementWiseOp(ew.1.clone()), &[input])?;
            return Ok(Some(Value::Wire(outlet[0])));
        }
        if let Some(frag) = self.fragments.get(&invocation.id) {
            let resolved = ResolvedInvocation { invocation, default_params: &frag.decl.parameters };
            return Ok(Some(
                builder.wire_fragment_invocation(&resolved, &frag.decl, frag.body.as_deref().unwrap())?,
            ));
        }
        Ok(None)
    }

}

pub fn tract_nnef() -> Registry {
    let mut reg = Registry::new("tract_nnef");
    deser::register(&mut reg);
    ser::register(&mut reg);
    reg
}

pub fn tract_core() -> Registry {
    let mut reg = Registry::new("tract_core");
    core::register(&mut reg);
    reg
}
