use crate::tfpb::graph::GraphDef;
use crate::tfpb::node_def::NodeDef;
use tract_core::internal::*;

pub struct ParsingContext;

#[derive(Clone, Default)]
pub struct TfOpRegister(pub HashMap<String, fn(&ParsingContext, node: &NodeDef) -> TractResult<Box<InferenceOp>>>);

impl TfOpRegister {
    pub fn insert(&mut self, s: &'static str, builder: fn(&ParsingContext, node: &NodeDef) -> TractResult<Box<InferenceOp>>) {
        self.0.insert(s.into(), builder);
    }
}

pub struct Tensorflow {
    pub op_register: TfOpRegister,
}

impl Tensorflow {
    // From the node_def.proto documentation:
    // Each input is "node:src_output" with "node" being a string name and
    // "src_output" indicating which output tensor to use from "node". If
    // "src_output" is 0 the ":0" suffix can be omitted. Regular inputs may
    // optionally be followed by control inputs that have the format "^node".
    fn parse_input(i: &str) -> TractResult<(&str, usize)> {
        let pair = if i.starts_with("^") {
            (&i[1..], 0)
        } else {
            let splits: Vec<_> = i.splitn(2, ':').collect();
            (splits[0], if splits.len() > 1 { splits[1].parse::<usize>()? } else { 0 })
        };
        Ok(pair)
    }

    pub fn determinize(model: &mut GraphDef) -> TractResult<()> {
        for pbnode in model.mut_node().iter_mut() {
            if pbnode.get_op() == "RandomUniform" {
                if pbnode.get_attr_int::<i64>("seed")? == 0 && pbnode.get_attr_int::<i64>("seed2")? == 0 {
                    pbnode.mut_attr().insert("seed".to_string(), 1.into());
                    pbnode.mut_attr().insert("seed2".to_string(), 1.into());
                }
            }
        }
        Ok(())
    }
}

impl Framework<GraphDef> for Tensorflow {
    fn proto_model_for_read(&self, r: &mut std::io::Read) -> TractResult<GraphDef> {
        Ok(::protobuf::parse_from_reader::<GraphDef>(r).map_err(|e| format!("{:?}", e))?)
    }

    fn model_for_proto_model(&self, graph: &GraphDef) -> TractResult<InferenceModel> {
        use crate::ops::control_flow as cf;

        let mut model = InferenceModel::default();
        // compute min output arity for all nodes
        let mut arities = HashMap::new();
        for pbnode in graph.get_node().iter() {
            for i in pbnode.get_input().iter() {
                let (node, slot) = Self::parse_input(i)?;
                let arity = arities.entry(node).or_insert(1);
                *arity = (*arity).max(slot + 1);
            }
        }

        for pbnode in graph.get_node().iter() {
            let name = pbnode.get_name().to_string();
            let output_arity = arities.get(&*name).cloned().unwrap_or(1);
            let facts = tvec!(TensorFact::default(); output_arity);

            if pbnode.get_op() == "NextIteration" {
                let source_op = cf::NextIteration::new(name.clone(), cf::NextIterationRole::Source);
                let sink_op = cf::NextIteration::new(name.clone(), cf::NextIterationRole::Sink);
                let _source = model.add_node(name.clone(), source_op, tvec!(TensorFact::default()))?;
                let _sink = model.add_node(format!("{}-Sink", name), sink_op, tvec!())?;
                continue;
            }

            let op = match self.op_register.0.get(pbnode.get_op()) {
                Some(builder) => (builder)(&ParsingContext, pbnode)?,
                None => tract_core::ops::unimpl::UnimplementedOp::new(pbnode.get_op(),
                            format!("{:?}", pbnode)).into(),
            };

            let node_id = model.add_node(name.clone(), op, facts)?;
            if pbnode.get_op() == "PlaceHolder" {
                let dt = pbnode.get_attr_datum_type("dtype")?;
                let mut fact = TensorFact::dt(dt);
                if let Some(shape) = pbnode.get_attr_opt_shape("shape")? {
                    fact = fact.with_shape(shape)
                }
                model.set_outlet_fact(OutletId::new(node_id, 0), fact)?;
            }
        }

        for pbnode in graph.get_node().iter() {
            let node_id = if pbnode.get_op() == "NextIteration" {
                model.node_by_name(&*format!("{}-Sink", pbnode.get_name()))?.id
            } else {
                model.node_by_name(pbnode.get_name())?.id
            };
            for (ix, i) in pbnode.get_input().iter().enumerate() {
                let input = Self::parse_input(i)?;
                let prec = model.node_by_name(input.0)?.id;
                if i.starts_with("^") {
                    model.node_mut(node_id).control_inputs.push(prec);
                } else {
                    let outlet = OutletId::new(prec, input.1);
                    let inlet = InletId::new(node_id, ix);
                    model.add_edge(outlet, inlet)?;
                }
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
        model.auto_outputs()?;
        Ok(model)
    }
}

