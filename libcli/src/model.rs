use tract_core::internal::*;
use tract_core::{downcast_rs, dyn_clone};

/// Common methods for all variants of model.
pub trait Model:
    downcast_rs::Downcast + std::fmt::Debug + dyn_clone::DynClone + Send + Sync
{
    /// Lookup node id by name
    fn node_id_by_name(&self, name: &str) -> TractResult<usize>;

    /// Node name by id
    fn node_name(&self, id: usize) -> &str;

    /// Node op by id
    fn node_op(&self, id: usize) -> &dyn Op;

    /// Node is const
    fn node_const(&self, id: usize) -> bool;

    /// Node op by id
    fn node_op_name(&self, id: usize) -> Cow<str>;

    /// Node inputs by id
    fn node_inputs(&self, id: usize) -> &[OutletId];

    /// Number of outputs for a node, by id.
    fn node_output_count(&self, id: usize) -> usize;

    /// Number nodes
    fn nodes_len(&self) -> usize;

    /// Formatted node label
    fn node_display(&self, id: usize) -> String;

    /// Formatted node label
    fn node_debug(&self, id: usize) -> String;

    /// Eval order for the model
    fn eval_order(&self) -> TractResult<Vec<usize>>;

    /// Eval order for the model
    fn eval_order_opt_ram(&self) -> TractResult<Vec<usize>>;

    /// Inputs of the model
    fn input_outlets(&self) -> &[OutletId];

    fn set_input_names(&mut self, names: &[&str]) -> TractResult<()>;
    fn set_output_names(&mut self, names: &[&str]) -> TractResult<()>;

    /// Outputs of the model
    fn output_outlets(&self) -> &[OutletId];

    /// Tensorfact for an outlet
    fn outlet_typedfact(&self, outlet: OutletId) -> TractResult<TypedFact>;

    /// Short outlet formatter (id plus fact)
    fn outlet_fact_format(&self, outlet: OutletId) -> String;

    /// Labels for an outlet
    fn outlet_label(&self, id: OutletId) -> Option<&str>;

    /// List consumers of an outlet
    fn outlet_successors(&self, outlet: OutletId) -> &[InletId];

    /// Subnets of a node
    fn nested_models(&self, id: usize) -> Vec<(String, &dyn Model)> {
        if let Some(submodel) =
            self.node_op(id).downcast_ref::<tract_core::ops::submodel::SubmodelOp>()
        {
            return vec![("submodel".into(), submodel.model())];
        }
        if let Some(lir) = self.node_op(id).downcast_ref::<tract_core::ops::scan::OptScan>() {
            return vec![("loop".into(), lir.plan.model())];
        }
        if let Some(mir) = self.node_op(id).downcast_ref::<tract_core::ops::scan::Scan>() {
            return vec![("loop".into(), &mir.body)];
        }
        if let Some(mir) = self.node_op(id).downcast_ref::<tract_core::ops::logic::IfThenElse>() {
            return vec![("then".into(), &mir.then_body), ("else".into(), &mir.else_body)];
        }
        #[cfg(feature = "hir")]
        if let Some(hir) = self.node_op(id).downcast_ref::<tract_hir::ops::scan::InferenceScan>() {
            return vec![("loop".into(), &hir.body)];
        }
        #[cfg(feature = "onnx")]
        if let Some(hir) = self.node_op(id).downcast_ref::<tract_onnx::ops::logic::If>() {
            return vec![("then".into(), &hir.then_body), ("else".into(), &hir.else_body)];
        }
        vec![]
    }

    /// Subnets of a node
    fn nested_models_iters(&self, id: usize, input: &[&TypedFact]) -> Option<TDim> {
        if let Some(submodel) =
            self.node_op(id).downcast_ref::<tract_core::ops::submodel::SubmodelOp>()
        {
            submodel.iteration_count(input)
        } else if let Some(lir) = self.node_op(id).downcast_ref::<tract_core::ops::scan::OptScan>()
        {
            lir.iteration_count(input)
        } else if let Some(mir) = self.node_op(id).downcast_ref::<tract_core::ops::scan::Scan>() {
            mir.iteration_count(input)
        } else {
            None
        }
    }

    fn auto_outputs(&mut self) -> TractResult<()>;

    fn properties(&self) -> &HashMap<String, Arc<Tensor>>;

    fn symbols(&self) -> &SymbolScope;

    fn get_or_intern_symbol(&self, name: &str) -> Symbol;

    fn rename_node(&mut self, id: usize, name: &str) -> TractResult<()>;
}

downcast_rs::impl_downcast!(Model);
dyn_clone::clone_trait_object!(Model);

impl<F, O> Model for Graph<F, O>
where
    F: Fact + Hash + Clone + 'static,
    O: std::fmt::Debug
        + std::fmt::Display
        + AsRef<dyn Op>
        + AsMut<dyn Op>
        + Clone
        + 'static
        + Send
        + Sync,
    Graph<F, O>: Send + Sync + 'static,
{
    fn node_id_by_name(&self, name: &str) -> TractResult<usize> {
        self.nodes
            .iter()
            .find(|n| n.name == name)
            .map(|n| n.id)
            .with_context(|| format!("No node found for name: \"{name}\""))
    }

    fn node_name(&self, id: usize) -> &str {
        &self.nodes[id].name
    }

    fn node_op_name(&self, id: usize) -> Cow<str> {
        self.node(id).op().name()
    }

    fn node_const(&self, id: usize) -> bool {
        self.node_op_name(id) == "Const"
    }

    fn node_inputs(&self, id: usize) -> &[OutletId] {
        &self.nodes[id].inputs
    }

    fn node_output_count(&self, id: usize) -> usize {
        self.nodes[id].outputs.len()
    }

    fn nodes_len(&self) -> usize {
        self.nodes.len()
    }

    fn node_display(&self, id: usize) -> String {
        format!("{}", self.nodes[id])
    }

    fn node_debug(&self, id: usize) -> String {
        format!("{:?}", self.nodes[id])
    }

    fn eval_order(&self) -> TractResult<Vec<usize>> {
        tract_core::model::order::eval_order(self)
    }

    fn eval_order_opt_ram(&self) -> TractResult<Vec<usize>> {
        tract_core::model::order::eval_order_opt_ram(self)
    }

    fn input_outlets(&self) -> &[OutletId] {
        &self.inputs
    }

    fn set_input_names(&mut self, names: &[&str]) -> TractResult<()> {
        self.set_input_names(names.iter())
    }

    fn set_output_names(&mut self, names: &[&str]) -> TractResult<()> {
        self.set_output_names(names)
    }

    fn output_outlets(&self) -> &[OutletId] {
        &self.outputs
    }

    fn node_op(&self, id: usize) -> &dyn Op {
        self.nodes[id].op.as_ref()
    }

    fn outlet_typedfact(&self, outlet: OutletId) -> TractResult<TypedFact> {
        Ok(self.outlet_fact(outlet)?.to_typed_fact()?.into_owned())
    }

    fn outlet_fact_format(&self, outlet: OutletId) -> String {
        format!("{:?}", self.outlet_fact(outlet).unwrap())
    }

    fn outlet_label(&self, id: OutletId) -> Option<&str> {
        self.outlet_label(id)
    }

    fn outlet_successors(&self, outlet: OutletId) -> &[InletId] {
        &self.nodes[outlet.node].outputs[outlet.slot].successors
    }

    fn auto_outputs(&mut self) -> TractResult<()> {
        self.auto_outputs()
    }

    fn properties(&self) -> &HashMap<String, Arc<Tensor>> {
        &self.properties
    }

    fn symbols(&self) -> &SymbolScope {
        &self.symbols
    }
    fn rename_node(&mut self, id: usize, name: &str) -> TractResult<()> {
        self.rename_node(id, name)
    }

    fn get_or_intern_symbol(&self, name: &str) -> Symbol {
        self.symbols.sym(name)
    }
}
