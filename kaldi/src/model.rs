use std::collections::BTreeMap;

use tract_hir::internal::*;

#[derive(Clone, Debug)]
pub struct KaldiProtoModel {
    pub config_lines: ConfigLines,
    pub components: HashMap<String, Component>,
    pub adjust_final_offset: isize,
}

#[derive(Clone, Debug)]
pub struct ConfigLines {
    pub input_name: String,
    pub input_dim: usize,
    pub nodes: Vec<(String, NodeLine)>,
    pub outputs: Vec<OutputLine>,
}

#[derive(Clone, Debug)]
pub enum NodeLine {
    Component(ComponentNode),
    DimRange(DimRangeNode),
}

#[derive(Clone, Debug)]
pub struct OutputLine {
    pub output_alias: String,
    pub descriptor: GeneralDescriptor,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GeneralDescriptor {
    Append(Vec<GeneralDescriptor>),
    IfDefined(Box<GeneralDescriptor>),
    Name(String),
    Offset(Box<GeneralDescriptor>, isize),
}

impl GeneralDescriptor {
    pub fn inputs(&self) -> TVec<&str> {
        match self {
            GeneralDescriptor::Append(ref gds) => gds.iter().fold(tvec!(), |mut acc, gd| {
                gd.inputs().iter().for_each(|i| {
                    if !acc.contains(i) {
                        acc.push(i)
                    }
                });
                acc
            }),
            GeneralDescriptor::IfDefined(ref gd) => gd.inputs(),
            GeneralDescriptor::Name(ref s) => tvec!(&**s),
            GeneralDescriptor::Offset(ref gd, _) => gd.inputs(),
        }
    }

    pub fn as_conv_shape_dilation(&self) -> Option<(usize, usize)> {
        if let GeneralDescriptor::Name(_) = self {
            return Some((1, 1));
        }
        if let GeneralDescriptor::Append(ref appendees) = self {
            let mut offsets = vec![];
            for app in appendees {
                match app {
                    GeneralDescriptor::Name(_) => offsets.push(0),
                    GeneralDescriptor::Offset(_, offset) => offsets.push(*offset),
                    _ => return None,
                }
            }
            let dilation = offsets[1] - offsets[0];
            if offsets.windows(2).all(|pair| pair[1] - pair[0] == dilation) {
                return Some((offsets.len(), dilation as usize));
            }
        }
        return None;
    }

    fn wire<'a>(
        &'a self,
        inlet: InletId,
        name: &str,
        model: &mut InferenceModel,
        deferred: &mut BTreeMap<InletId, String>,
        adjust_final_offset: Option<isize>,
    ) -> TractResult<()> {
        use GeneralDescriptor::*;
        match &self {
            &Name(n) => {
                deferred.insert(inlet, n.to_string());
                return Ok(());
            }
            &Append(appendees) => {
                let name = format!("{}.Append", name);
                let id = model.add_node(
                    &*name,
                    expand(tract_hir::ops::array::Concat::new(1)),
                    tvec!(InferenceFact::default()),
                )?;
                model.add_edge(OutletId::new(id, 0), inlet)?;
                for (ix, appendee) in appendees.iter().enumerate() {
                    let name = format!("{}-{}", name, ix);
                    appendee.wire(
                        InletId::new(id, ix),
                        &*name,
                        model,
                        deferred,
                        adjust_final_offset,
                    )?;
                }
                return Ok(());
            }
            &IfDefined(ref o) => {
                if let &Offset(ref n, ref o) = &**o {
                    if let Name(n) = &**n {
                        let name = format!("{}.memory", name);
                        model.add_node(
                            &*name,
                            crate::ops::memory::Memory::new(n.to_string(), *o),
                            tvec!(InferenceFact::default()),
                        )?;
                        deferred.insert(inlet, name);
                        return Ok(());
                    }
                }
            }
            &Offset(ref n, o) if *o > 0 => {
                let name = format!("{}-Delay", name);
                let crop = *o as isize + adjust_final_offset.unwrap_or(0);
                if crop < 0 {
                    bail!("Invalid offset adjustment (network as {}, adjustment is {}", o, crop)
                }
                let id = model.add_node(
                    &*name,
                    expand(tract_hir::ops::array::Crop::new(0, crop as usize, 0)),
                    tvec!(InferenceFact::default()),
                )?;
                model.add_edge(OutletId::new(id, 0), inlet)?;
                n.wire(InletId::new(id, 0), &*name, model, deferred, adjust_final_offset)?;
                return Ok(());
            }
            _ => (),
        }
        bail!("Unhandled input descriptor: {:?}", self)
    }
}

#[derive(Clone, Debug)]
pub struct DimRangeNode {
    pub input: GeneralDescriptor,
    pub offset: usize,
    pub dim: usize,
}

#[derive(Clone, Debug)]
pub struct ComponentNode {
    pub input: GeneralDescriptor,
    pub component: String,
}

#[derive(Clone, Debug, Default)]
pub struct Component {
    pub klass: String,
    pub attributes: HashMap<String, Arc<Tensor>>,
}

pub struct ParsingContext<'a> {
    pub proto_model: &'a KaldiProtoModel,
}

#[derive(Clone, Default)]
pub struct KaldiOpRegister(
    pub HashMap<String, fn(&ParsingContext, node: &str) -> TractResult<Box<dyn InferenceOp>>>,
);

impl KaldiOpRegister {
    pub fn insert(
        &mut self,
        s: &'static str,
        builder: fn(&ParsingContext, node: &str) -> TractResult<Box<dyn InferenceOp>>,
    ) {
        self.0.insert(s.into(), builder);
    }
}

#[derive(Clone, Default)]
pub struct Kaldi {
    pub op_register: KaldiOpRegister,
}

impl Framework<KaldiProtoModel, InferenceModel> for Kaldi {
    fn proto_model_for_read(&self, r: &mut dyn std::io::Read) -> TractResult<KaldiProtoModel> {
        use crate::parser;
        let mut v = vec![];
        r.read_to_end(&mut v)?;
        parser::nnet3(&*v)
    }

    fn model_for_proto_model(&self, proto_model: &KaldiProtoModel) -> TractResult<InferenceModel> {
        let ctx = ParsingContext { proto_model };
        let mut model = InferenceModel::default();
        let s = tract_hir::tract_core::pulse::stream_dim();
        model.add_source(
            proto_model.config_lines.input_name.clone(),
            InferenceFact::dt_shape(
                f32::datum_type(),
                shapefactoid!(s, (proto_model.config_lines.input_dim)),
            ),
        )?;
        let mut inputs_to_wire: BTreeMap<InletId, String> = Default::default();
        for (name, node) in &proto_model.config_lines.nodes {
            match node {
                NodeLine::Component(line) => {
                    let component = &proto_model.components[&line.component];
                    if crate::ops::AFFINE.contains(&&*component.klass)
                        && line.input.as_conv_shape_dilation().is_some()
                    {
                        let op = crate::ops::affine::affine_component(&ctx, name)?;
                        let id = model.add_node(
                            name.to_string(),
                            op,
                            tvec!(InferenceFact::default()),
                        )?;
                        inputs_to_wire
                            .insert(InletId::new(id, 0), line.input.inputs()[0].to_owned());
                    } else {
                        let op = match self.op_register.0.get(&*component.klass) {
                            Some(builder) => (builder)(&ctx, name)?,
                            None => Box::new(tract_hir::ops::unimpl::UnimplementedOp::new(
                                1,
                                component.klass.to_string(),
                                format!("{:?}", line),
                            )),
                        };
                        let id = model.add_node(
                            name.to_string(),
                            op,
                            tvec!(InferenceFact::default()),
                        )?;
                        line.input.wire(
                            InletId::new(id, 0),
                            name,
                            &mut model,
                            &mut inputs_to_wire,
                            None,
                        )?
                    }
                }
                NodeLine::DimRange(line) => {
                    let op = tract_hir::ops::array::Slice::new(
                        1,
                        line.offset as usize,
                        (line.offset + line.dim) as usize,
                    );
                    let id =
                        model.add_node(name.to_string(), op, tvec!(InferenceFact::default()))?;
                    line.input.wire(
                        InletId::new(id, 0),
                        name,
                        &mut model,
                        &mut inputs_to_wire,
                        None,
                    )?
                }
            }
        }
        let mut outputs = vec![];
        for o in &proto_model.config_lines.outputs {
            let output = model.add_node(
                &*o.output_alias,
                tract_hir::ops::identity::Identity::default(),
                tvec!(InferenceFact::default()),
            )?;
            o.descriptor.wire(
                InletId::new(output, 0),
                "output",
                &mut model,
                &mut inputs_to_wire,
                Some(proto_model.adjust_final_offset),
            )?;
            outputs.push(OutletId::new(output, 0));
        }
        for (inlet, name) in inputs_to_wire {
            let src = OutletId::new(model.node_by_name(&*name)?.id, 0);
            model.add_edge(src, inlet)?;
        }
        model.set_output_outlets(&*outputs)?;
        Ok(model)
    }
}
