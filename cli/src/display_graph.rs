use crate::draw::DrawingState;
use crate::CliResult;
use crate::SomeGraphDef;
use ansi_term::Color::*;
use ansi_term::Style;
use std::borrow::Borrow;
use std::collections::HashMap;
#[allow(unused_imports)]
use std::convert::TryFrom;
use std::sync::Arc;
use tract_core::internal::*;
use tract_core::itertools::Itertools;
#[cfg(feature = "onnx")]
use tract_onnx::pb::ModelProto;
#[cfg(feature = "tf")]
use tract_tensorflow::tfpb::tensorflow::GraphDef;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Io {
    None,
    Short,
    Long,
}

impl Default for Io {
    fn default() -> Io {
        Io::Short
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DisplayOptions {
    pub konst: bool,
    pub invariants: bool,
    pub quiet: bool,
    pub natural_order: bool,
    pub debug_op: bool,
    pub node_ids: Option<Vec<TVec<usize>>>,
    pub op_name: Option<String>,
    pub node_name: Option<String>,
    pub expect_canonic: bool,
    pub outlet_labels: bool,
    pub io: Io,
    pub info: bool,
}

impl DisplayOptions {
    pub fn filter(
        &self,
        model: &dyn Model,
        current_prefix: &[usize],
        node_id: usize,
    ) -> CliResult<bool> {
        if let Some(nodes) = self.node_ids.as_ref() {
            return Ok(nodes.iter().any(|n| {
                n.len() == current_prefix.len() + 1
                    && &n[0..current_prefix.len()] == current_prefix
                    && *n.last().unwrap() == node_id
            }));
        }
        if let Some(node_name) = self.node_name.as_ref() {
            return Ok(model.node_name(node_id).starts_with(&*node_name));
        }
        if let Some(op_name) = self.op_name.as_ref() {
            return Ok(model.node_op(node_id).name().starts_with(op_name));
        }
        /*
        if let Some(successor) = self.successors {
        return Ok(model.node_inputs(node_id).iter().any(|i| i.node == successor));
        }
        */
        Ok(model.node_op(node_id).name() != "Const" || self.konst)
    }

    pub fn should_draw(&self) -> bool {
        !self.natural_order
    }
}

#[derive(Debug, Clone)]
pub struct DisplayGraph<'a> {
    model: &'a dyn Model,
    prefix: TVec<usize>,
    pub options: Arc<DisplayOptions>,
    node_color: HashMap<usize, Style>,
    node_labels: HashMap<usize, Vec<String>>,
    node_sections: HashMap<usize, Vec<Vec<String>>>,
    node_nested_graphs: HashMap<usize, Vec<(String, DisplayGraph<'a>)>>,
    model_input_labels: HashMap<usize, String>,
    model_output_labels: HashMap<usize, String>,
}

impl<'a> DisplayGraph<'a> {
    pub fn render(&self) -> CliResult<()> {
        self.render_prefixed("")
    }

    pub fn render_prefixed(&self, prefix: &str) -> CliResult<()> {
        if self.options.quiet {
            return Ok(());
        }
        let mut drawing_state =
            if self.options.should_draw() { Some(DrawingState::default()) } else { None };
        let node_ids = if self.options.natural_order {
            (0..self.model.nodes_len()).collect()
        } else {
            self.model.eval_order()?
        };
        for node in node_ids {
            if self.options.filter(self.model, &*self.prefix, node)? {
                self.render_node_prefixed(node, prefix, drawing_state.as_mut())?
            } else if let Some(ref mut ds) = drawing_state {
                let _prefix = ds.draw_node_vprefix(self.model, node, &self.options)?;
                let _body = ds.draw_node_body(self.model, node, &self.options)?;
                let _suffix = ds.draw_node_vsuffix(self.model, node, &self.options)?;
            }
        }
        Ok(())
    }

    pub fn render_node(&self, node_id: usize) -> CliResult<()> {
        self.render_node_prefixed(node_id, "", None)
    }

    pub fn render_node_prefixed(
        &self,
        node_id: usize,
        prefix: &str,
        mut drawing_state: Option<&mut DrawingState>,
    ) -> CliResult<()> {
        let model = self.model.borrow();
        let name_color = self.node_color.get(&node_id).cloned().unwrap_or(White.into());
        let node_name = model.node_name(node_id);
        let node_op_name = model.node_op(node_id).name();
        // println!("{:?}", model.node_format(node_id));
        if let Some(ref mut ds) = &mut drawing_state {
            for l in ds.draw_node_vprefix(model, node_id, &self.options)? {
                println!("{}{} ", prefix, l);
            }
        }
        let mut drawing_lines: Box<dyn Iterator<Item = String>> = if let Some(ds) =
            drawing_state.as_mut()
        {
            let body = ds.draw_node_body(model, node_id, &self.options)?;
            let suffix = ds.draw_node_vsuffix(model, node_id, &self.options)?;
            let filler = ds.draw_node_vfiller(model, node_id)?;
            Box::new(body.into_iter().chain(suffix.into_iter()).chain(std::iter::repeat(filler)))
        } else {
            Box::new(std::iter::repeat(String::new()))
        };
        macro_rules! prefix {
            () => {
                print!("{}{} ", prefix, drawing_lines.next().unwrap(),)
            };
        };
        prefix!();
        println!(
            "{} {} {}",
            White.bold().paint(format!("{}", node_id)),
            (if node_name == "UnimplementedOp" {
                Red.bold()
            } else {
                if self.options.expect_canonic && !model.node_op(node_id).is_canonic() {
                    Yellow.bold()
                } else {
                    Blue.bold()
                }
            })
            .paint(node_op_name),
            name_color.italic().paint(node_name)
        );
        for label in self.node_labels.get(&node_id).unwrap_or(&vec![]).iter() {
            prefix!();
            println!("  * {}", label);
        }
        match self.options.io {
            Io::Long => {
                for (ix, i) in model.node_inputs(node_id).iter().enumerate() {
                    let star = if ix == 0 { '*' } else { ' ' };
                    prefix!();
                    println!(
                        "  {} input fact  #{}: {} {}",
                        star,
                        ix,
                        White.bold().paint(format!("{:?}", i)),
                        model.outlet_fact_format(*i),
                    );
                }
                for ix in 0..model.node_output_count(node_id) {
                    let star = if ix == 0 { '*' } else { ' ' };
                    let io = if let Some(id) = self
                        .model
                        .borrow()
                        .input_outlets()
                        .iter()
                        .position(|n| n.node == node_id && n.slot == ix)
                    {
                        format!(
                            "{} {}",
                            Cyan.bold().paint(format!("MODEL INPUT #{}", id)).to_string(),
                            self.model_input_labels.get(&id).map(|s| &**s).unwrap_or("")
                        )
                    } else if let Some(id) = self
                        .model
                        .borrow()
                        .output_outlets()
                        .iter()
                        .position(|n| n.node == node_id && n.slot == ix)
                    {
                        format!(
                            "{} {}",
                            Yellow.bold().paint(format!("MODEL OUTPUT #{}", id)).to_string(),
                            self.model_output_labels.get(&id).map(|s| &**s).unwrap_or("")
                        )
                    } else {
                        "".to_string()
                    };
                    let outlet = OutletId::new(node_id, ix);
                    let successors = model.outlet_successors(outlet);
                    prefix!();
                    println!(
                        "  {} output fact #{}: {} {} {}",
                        star,
                        ix,
                        model.outlet_fact_format(outlet),
                        White.bold().paint(successors.iter().map(|s| format!("{:?}", s)).join(" ")),
                        io
                    );
                    if self.options.outlet_labels {
                        if let Some(label) = model.outlet_label(OutletId::new(node_id, ix)) {
                            prefix!();
                            println!("            {} ", White.italic().paint(label));
                        }
                    }
                }
            }
            Io::Short => {
                let same = model.node_inputs(node_id).len() > 0
                    && model.node_output_count(node_id) == 1
                    && model.outlet_fact_format(node_id.into())
                        == model.outlet_fact_format(model.node_inputs(node_id)[0]);
                if !same {
                    let style = drawing_state
                        .and_then(|w| w.wires.last())
                        .and_then(|w| w.color)
                        .unwrap_or(White.into());
                    for ix in 0..model.node_output_count(node_id) {
                        prefix!();
                        println!(
                            "  {}{}{} {}",
                            style.paint(box_drawing::heavy::HORIZONTAL),
                            style.paint(box_drawing::heavy::HORIZONTAL),
                            style.paint(box_drawing::heavy::HORIZONTAL),
                            model.outlet_fact_format((node_id, ix).into())
                        );
                    }
                }
            }
            Io::None => (),
        }
        if self.options.info {
            for info in model.node_op(node_id).info()? {
                prefix!();
                println!("  * {}", info);
            }
        }
        if self.options.invariants {
            if let Some(typed) = model.downcast_ref::<TypedModel>() {
                let node = typed.node(node_id);
                prefix!();
                println!("  * {:?}", node.op().as_typed().unwrap().invariants(&typed, &node)?);
            }
        }
        if self.options.debug_op {
            prefix!();
            println!("  * {:?}", model.node_op(node_id));
        }
        if let Some(node_sections) = self.node_sections.get(&node_id) {
            for section in node_sections {
                if section.is_empty() {
                    continue;
                }
                prefix!();
                println!("  * {}", section[0]);
                for s in &section[1..] {
                    prefix!();
                    println!("    {}", s);
                }
            }
        }
        for (label, sub) in self.node_nested_graphs.get(&node_id).unwrap_or(&vec![]) {
            let prefix = drawing_lines.next().unwrap();
            sub.render_prefixed(&format!("{} [{}] ", prefix, label))?
        }
        Ok(())
    }

    pub fn from_model_and_options(
        model: &'a dyn Model,
        options: Arc<DisplayOptions>,
    ) -> CliResult<DisplayGraph<'a>> {
        Self::from_model_prefix_and_options(model, [].as_ref(), options)
    }

    fn from_model_prefix_and_options(
        model: &'a dyn Model,
        prefix: &[usize],
        options: Arc<DisplayOptions>,
    ) -> CliResult<DisplayGraph<'a>> {
        let mut node_nested_graphs = HashMap::new();
        for n in 0..model.nodes_len() {
            let subs = model.node_op(n).nested_models();
            if subs.len() > 0 {
                let mut prefix: TVec<usize> = prefix.into();
                prefix.push(n);
                node_nested_graphs.insert(
                    n,
                    subs.into_iter()
                        .map(|(label, sub, inputs, outputs)| {
                            let mut dg = Self::from_model_prefix_and_options(
                                sub,
                                &*prefix,
                                Arc::clone(&options),
                            )?;
                            inputs.into_iter().enumerate().for_each(|(ix, i)| {
                                dg.model_input_labels.insert(ix, i);
                            });
                            outputs.into_iter().enumerate().for_each(|(ix, o)| {
                                dg.model_output_labels.insert(ix, o);
                            });
                            Ok((label.into_owned(), dg))
                        })
                        .collect::<CliResult<_>>()?,
                );
            }
        }
        Ok(DisplayGraph {
            model,
            prefix: prefix.into(),
            options,
            node_color: HashMap::new(),
            node_labels: HashMap::new(),
            node_sections: HashMap::new(),
            model_input_labels: HashMap::new(),
            model_output_labels: HashMap::new(),
            node_nested_graphs,
        })
    }

    pub fn with_graph_def(self, graph_def: &SomeGraphDef) -> CliResult<DisplayGraph<'a>> {
        match graph_def {
            SomeGraphDef::NoGraphDef => Ok(self),
            #[cfg(feature = "kaldi")]
            SomeGraphDef::Kaldi(kaldi) => self.with_kaldi(kaldi),
            #[cfg(feature = "onnx")]
            SomeGraphDef::Onnx(onnx, _) => self.with_onnx_model(onnx),
            #[cfg(feature = "tf")]
            SomeGraphDef::Tf(tf) => self.with_tf_graph_def(tf),
        }
    }

    pub fn set_node_color<S: Into<Style>>(&mut self, id: usize, color: S) -> CliResult<()> {
        self.node_color.insert(id, color.into());
        Ok(())
    }

    pub fn add_node_label<S: Into<String>>(&mut self, id: &[usize], label: S) -> CliResult<()> {
        if id.len() == 1 {
            self.node_labels.entry(id[0]).or_insert(vec![]).push(label.into());
            Ok(())
        } else {
            self.node_nested_graphs.get_mut(&id[0]).unwrap()[0].1.add_node_label(&id[1..], label)
        }
    }

    pub fn add_node_section(&mut self, id: &[usize], section: Vec<String>) -> CliResult<()> {
        if id.len() == 1 {
            self.node_sections.entry(id[0]).or_insert(vec![]).push(section);
            Ok(())
        } else {
            self.node_nested_graphs.get_mut(&id[0]).unwrap()[0]
                .1
                .add_node_section(&id[1..], section)
        }
    }

    #[cfg(feature = "kaldi")]
    pub fn with_kaldi(
        mut self,
        proto_model: &tract_kaldi::KaldiProtoModel,
    ) -> CliResult<DisplayGraph<'a>> {
        use tract_kaldi::model::NodeLine;
        let bold = Style::new().bold();
        for (name, proto_node) in &proto_model.config_lines.nodes {
            if let Ok(node_id) = self.model.borrow().node_id_by_name(&*name) {
                let mut vs = vec![];
                match proto_node {
                    NodeLine::Component(compo) => {
                        let comp = &proto_model.components[&compo.component];
                        for (k, v) in &comp.attributes {
                            let value = format!("{:?}", v);
                            vs.push(format!("Attr {}: {:.240}", bold.paint(k), value));
                        }
                    }
                    _ => (),
                }
                self.add_node_section(&[node_id], vs)?;
            }
        }
        Ok(self)
    }

    #[cfg(feature = "tf")]
    pub fn with_tf_graph_def(mut self, graph_def: &GraphDef) -> CliResult<DisplayGraph<'a>> {
        let bold = Style::new().bold();
        for gnode in graph_def.node.iter() {
            if let Ok(node_id) = self.model.borrow().node_id_by_name(&gnode.name) {
                let mut v = vec![];
                for a in gnode.attr.iter() {
                    let value = if let Some(
                        tract_tensorflow::tfpb::tensorflow::attr_value::Value::Tensor(r),
                    ) = &a.1.value
                    {
                        format!("{:?}", r)
                    } else {
                        format!("{:?}", a.1)
                    };
                    v.push(format!("Attr {}: {:.240}", bold.paint(a.0), value));
                }
                self.add_node_section(&[node_id], v)?;
            }
        }
        Ok(self)
    }

    #[cfg(feature = "onnx")]
    pub fn with_onnx_model(mut self, model_proto: &ModelProto) -> CliResult<DisplayGraph<'a>> {
        let bold = Style::new().bold();
        for gnode in model_proto.graph.as_ref().unwrap().node.iter() {
            let mut node_name = &gnode.name;
            if node_name == "" && gnode.output.len() > 0 {
                node_name = &gnode.output[0];
            }
            if let Ok(id) = self.model.borrow().node_id_by_name(&*node_name) {
                let mut v = vec![];
                for a in gnode.attribute.iter() {
                    let value = if let Some(t) = &a.t {
                        format!("{:?}", Tensor::try_from(t)?)
                    } else {
                        format!("{:?}", a)
                    };
                    v.push(format!("Attr {}: {:.240}", bold.paint(&a.name), value));
                }
                self.add_node_section(&[id], v)?;
            }
        }
        Ok(self)
    }
}
