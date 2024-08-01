use crate::tfpb::tensorflow::{GraphDef, NodeDef, SavedModel};
use prost::Message;
use std::{fs, path};
use tract_hir::internal::*;

#[derive(Default)]
pub struct ParsingContext {
    pub node_output_arities: HashMap<String, usize>,
}

type OpBuilder = fn(&ParsingContext, node: &NodeDef) -> TractResult<Box<dyn InferenceOp>>;

#[derive(Clone, Default)]
pub struct TfOpRegister(pub HashMap<String, OpBuilder>);

impl TfOpRegister {
    pub fn insert(&mut self, s: &'static str, builder: OpBuilder) {
        self.0.insert(s.into(), builder);
    }
}

pub struct Tensorflow {
    pub op_register: TfOpRegister,
}

pub struct TfModelExtensions {
    pub control_inputs: Vec<(usize, usize)>,
    pub initializing_nodes: Vec<usize>,
}

impl TfModelExtensions {
    pub fn preproc(&self, mut original: InferenceModel) -> TractResult<InferenceModel> {
        if self.initializing_nodes.len() > 0 {
            let as_outlets =
                self.initializing_nodes.iter().map(|n| OutletId::new(*n, 0)).collect::<Vec<_>>();
            let plan = SimplePlan::build(
                &original,
                &as_outlets,
                &self.control_inputs,
                &PlanOptions::default(),
            )?;
            let mut state = SimpleState::new(plan)?;
            state.exec()?;
            let tensors = state.session_state.tensors;
            for node in &mut original.nodes {
                if let Some(var) = node.op_as_mut::<crate::ops::vars::VariableV2>() {
                    if let Some(value) = tensors.get(&var.id) {
                        var.initializer = Some(value.clone().into_arc_tensor());
                    }
                }
            }
        }
        Ok(original)
    }
}

pub struct TfModelAndExtensions(pub InferenceModel, pub TfModelExtensions);

impl Tensorflow {
    // From the node_def.proto documentation:
    // Each input is "node:src_output" with "node" being a string name and
    // "src_output" indicating which output tensor to use from "node". If
    // "src_output" is 0 the ":0" suffix can be omitted. Regular inputs may
    // optionally be followed by control inputs that have the format "^node".
    fn parse_input(i: &str) -> TractResult<(&str, usize)> {
        let pair = if let Some(stripped) = i.strip_prefix('^') {
            (stripped, 0)
        } else {
            let splits: Vec<_> = i.splitn(2, ':').collect();
            (splits[0], if splits.len() > 1 { splits[1].parse::<usize>()? } else { 0 })
        };
        Ok(pair)
    }

    pub fn determinize(model: &mut GraphDef) -> TractResult<()> {
        for pbnode in &mut model.node {
            if pbnode.op == "RandomUniform"
                && pbnode.get_attr_int::<i64>("seed")? == 0
                && pbnode.get_attr_int::<i64>("seed2")? == 0
            {
                pbnode.attr.insert("seed".to_string(), 1.into());
                pbnode.attr.insert("seed2".to_string(), 1.into());
            }
        }
        Ok(())
    }

    #[cfg(target_family = "wasm")]
    pub fn read_frozen_from_path(&self, p: impl AsRef<path::Path>) -> TractResult<GraphDef> {
        use std::io::Read;
        let mut file = fs::File::open(p)?;
        let mut v = Vec::with_capacity(file.metadata()?.len() as usize);
        file.read_to_end(&mut v)?;
        let b = bytes::Bytes::from(v);
        Ok(GraphDef::decode(b)?)
    }

    #[cfg(all(any(windows, unix), not(target_os = "emscripten")))]
    pub fn read_frozen_from_path(&self, p: impl AsRef<path::Path>) -> TractResult<GraphDef> {
        let map = unsafe { memmap2::Mmap::map(&fs::File::open(p)?)? };
        Ok(GraphDef::decode(&*map)?)
    }

    pub fn read_frozen_model(&self, r: &mut dyn std::io::Read) -> TractResult<GraphDef> {
        let mut v = vec![];
        r.read_to_end(&mut v)?;
        let b = bytes::Bytes::from(v);
        Ok(GraphDef::decode(b)?)
    }

    pub fn open_saved_model(&self, r: &mut dyn std::io::Read) -> TractResult<SavedModel> {
        let mut v = vec![];
        r.read_to_end(&mut v)?;
        let b = bytes::Bytes::from(v);
        Ok(SavedModel::decode(b)?)
    }

    /// Convenience method: will read the first model in the saved model
    /// container. Use open_avec_model for more control.
    pub fn read_saved_model(&self, r: &mut dyn std::io::Read) -> TractResult<GraphDef> {
        let mut saved = self.open_saved_model(r)?;
        Ok(saved.meta_graphs.remove(0).graph_def.unwrap())
    }

    pub fn parse_graph(&self, graph: &GraphDef) -> TractResult<TfModelAndExtensions> {
        self.parse_graph_with_template(graph, Default::default())
    }

    pub fn parse_graph_with_template(
        &self,
        graph: &GraphDef,
        mut model: InferenceModel
    ) -> TractResult<TfModelAndExtensions> {
        use crate::ops::control_flow as cf;

        let mut inputs = tvec!();
        let mut context = ParsingContext::default();
        let mut control_inputs = vec![];

        // compute min output arity for all nodes
        for pbnode in &graph.node {
            for i in &pbnode.input {
                let (node, slot) = Self::parse_input(i)?;
                let arity = context.node_output_arities.entry(node.to_string()).or_insert(1);
                *arity = (*arity).max(slot + 1);
            }
        }

        for pbnode in &graph.node {
            let name = &pbnode.name;

            if pbnode.op == "NextIteration" {
                let source_op = cf::NextIteration::new(name.clone(), cf::NextIterationRole::Source);
                let sink_op = cf::NextIteration::new(name.clone(), cf::NextIterationRole::Sink);
                let _source =
                    model.add_node(name.clone(), source_op, tvec!(InferenceFact::default()))?;
                let _sink = model.add_node(format!("{name}-Sink"), sink_op, tvec!())?;
                continue;
            }

            let op = match self.op_register.0.get(&pbnode.op) {
                Some(builder) => (builder)(&context, pbnode)?,
                None => tract_hir::ops::unimpl::UnimplementedOp::new(
                    context.node_output_arities.get(name).cloned().unwrap_or(1),
                    &pbnode.op,
                    format!("{pbnode:?}"),
                )
                .into(),
            };

            let noutputs =
                op.nboutputs()?.max(context.node_output_arities.get(name).cloned().unwrap_or(1));
            let facts = tvec!(InferenceFact::default(); noutputs);

            let node_id = model.add_node(name.clone(), op, facts)?;
            if pbnode.op == "Placeholder" {
                let dt = pbnode.get_attr_datum_type("dtype")?;
                let mut fact = InferenceFact::dt(dt);
                if let Some(shape) = pbnode.get_attr_opt_shape("shape")? {
                    let shape_factoid = ShapeFactoid::closed(
                        shape
                            .iter()
                            .map(|d| {
                                if *d == -1 {
                                    GenericFactoid::Any
                                } else {
                                    GenericFactoid::Only(d.to_dim())
                                }
                            })
                            .collect(),
                    );
                    fact = fact.with_shape(shape_factoid);
                }
                inputs.push(OutletId::new(node_id, 0));
                model.set_outlet_fact(OutletId::new(node_id, 0), fact)?;
            }
        }

        for pbnode in &graph.node {
            let node_id = if pbnode.op == "NextIteration" {
                model.node_by_name(&*format!("{}-Sink", &pbnode.name))?.id
            } else {
                model.node_by_name(&pbnode.name)?.id
            };
            for (ix, i) in pbnode.input.iter().filter(|n| !n.starts_with('^')).enumerate() {
                let input = Self::parse_input(i)?;
                let prec = model.node_by_name(input.0)?.id;
                let outlet = OutletId::new(prec, input.1);
                let inlet = InletId::new(node_id, ix);
                model.add_edge(outlet, inlet)?;
                model.set_outlet_label(outlet, i.to_string())?;
            }
            for i in pbnode.input.iter().filter(|n| n.starts_with('^')) {
                let input = Self::parse_input(i)?;
                let prec = model.node_by_name(input.0)?.id;
                control_inputs.push((model.node_id_by_name(&pbnode.name)?, prec));
            }
        }

        // variable -> assign rewire
        //  * Assign consumes this by_ref tensor on #0 and somehow performs
        //      updates on it (it has a second input on #1 for the value to
        //      assign)
        //
        // in tract:
        //  * VariableV2 outputs a regular tensor stored in the session state
        //  * Assign has the same inputs, but do not uses the #0, udating the
        //      state session instead
        for id in 0..model.nodes().len() {
            use crate::ops::vars::*;
            if model.node(id).op_is::<Assign>() {
                let prec = model.node(id).inputs[0];
                let var_id = model.node(prec.node).op_as::<VariableV2>().map(|v| v.id.clone());
                if let (Some(var_id), Some(assign)) =
                    (var_id, model.node_mut(id).op_as_mut::<Assign>())
                {
                    assign.var_id = Some(var_id);
                } else {
                    bail!("Model contains unlinked Assign/Variable2");
                }
            }
        }
        model.set_input_outlets(&inputs)?;
        model.auto_outputs()?;
        let extensions = TfModelExtensions { control_inputs, initializing_nodes: vec![] };
        Ok(TfModelAndExtensions(model, extensions))
    }
}

impl Framework<GraphDef, InferenceModel> for Tensorflow {
    /// This method will try to read as frozen model, then as a saved model.
    fn proto_model_for_path(&self, r: impl AsRef<path::Path>) -> TractResult<GraphDef> {
        self.read_frozen_model(&mut fs::File::open(r.as_ref())?)
            .or_else(|_| self.read_saved_model(&mut fs::File::open(r.as_ref())?))
    }

    /// This method expects a frozen model, use open_saved_model for TF2 saved
    /// model format.
    fn proto_model_for_read(&self, r: &mut dyn std::io::Read) -> TractResult<GraphDef> {
        self.read_frozen_model(r)
    }

    fn model_for_proto_model_with_model_template(
            &self,
            proto: &GraphDef,
            template: InferenceModel,
        ) -> TractResult<InferenceModel> {
        Ok(self.parse_graph_with_template(proto, template)?.0)
    }
}
