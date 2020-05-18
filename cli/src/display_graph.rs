use crate::display_params::*;
use crate::draw::DrawingState;
use crate::CliResult;
use crate::SomeGraphDef;
use ansi_term::Color::*;
use ansi_term::Style;
use std::borrow::Borrow;
use std::collections::HashMap;
#[allow(unused_imports)]
use std::convert::TryFrom;
use std::time::Duration;
use tract_core::internal::*;
use tract_core::itertools::Itertools;
#[cfg(feature = "onnx")]
use tract_onnx::pb::ModelProto;
#[cfg(feature = "tf")]
use tract_tensorflow::tfpb::tensorflow::GraphDef;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct NodeQId(pub TVec<(usize, String)>, pub usize);

impl From<usize> for NodeQId {
    fn from(id: usize) -> NodeQId {
        NodeQId(tvec!(), id)
    }
}

#[derive(Debug, Default, Clone)]
pub struct NodeTags {
    pub cost: Option<Vec<(Cost, TDim)>>,
    pub style: Option<Style>,
    pub labels: Vec<String>,
    pub sections: Vec<Vec<String>>,
    pub profile: Option<Duration>,
    pub model_input: Option<String>,
    pub model_output: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DisplayGraph<'a> {
    options: &'a DisplayParams,
    pub tags: HashMap<NodeQId, NodeTags>,
    pub profile_summary: Option<crate::profile::ProfileSummary>,
}

impl<'a> DisplayGraph<'a> {
    pub fn node_mut(&mut self, qid: NodeQId) -> &mut NodeTags {
        self.tags.entry(qid).or_default()
    }
}

/*
node_nested_graphs: HashMap<usize, Vec<(String, DisplayGraph<'a>)>>,


cost: Option<Vec<Vec<(Cost, TDim)>>>,

node_color: HashMap<usize, Style>,
node_labels: HashMap<usize, Vec<String>>,
node_sections: HashMap<usize, Vec<Vec<String>>>,
model_input_labels: HashMap<usize, String>,
model_output_labels: HashMap<usize, String>,
}
*/

impl<'a> DisplayGraph<'a> {
    pub fn render(&self, model: &dyn Model) -> CliResult<()> {
        self.render_prefixed(model, "", &[])
    }

    pub fn render_node(&self, model: &dyn Model, node_id: usize) -> CliResult<()> {
        self.render_node_prefixed(model, "", &[], node_id, None)
    }

    fn render_prefixed(
        &self,
        model: &dyn Model,
        prefix: &str,
        scope: &[(usize, String)],
    ) -> CliResult<()> {
        if self.options.quiet {
            return Ok(());
        }
        let mut drawing_state =
            if self.options.should_draw() { Some(DrawingState::default()) } else { None };
        let node_ids = if self.options.natural_order {
            (0..model.nodes_len()).collect()
        } else {
            model.eval_order()?
        };
        for node in node_ids {
            if self.options.filter(model, scope, node)? {
                self.render_node_prefixed(model, prefix, scope, node, drawing_state.as_mut())?
            } else if let Some(ref mut ds) = drawing_state {
                let _prefix = ds.draw_node_vprefix(model, node, &self.options)?;
                let _body = ds.draw_node_body(model, node, &self.options)?;
                let _suffix = ds.draw_node_vsuffix(model, node, &self.options)?;
            }
        }
        Ok(())
    }

    fn render_node_prefixed(
        &self,
        model: &dyn Model,
        prefix: &str,
        scope: &[(usize, String)],
        node_id: usize,
        mut drawing_state: Option<&mut DrawingState>,
    ) -> CliResult<()> {
        let model = model.borrow();
        let qid = NodeQId(scope.into(), node_id);
        let tags = self.tags.get(&qid).cloned().unwrap_or_default();
        let name_color = tags.style.clone().unwrap_or(White.into());
        let node_name = model.node_name(node_id);
        let node_op_name = model.node_op(node_id).name();
        let cost_column_pad = format!("{:>1$}", "", self.options.cost as usize * 25);
        let profile_column_pad = format!("{:>1$}", "", self.options.profile as usize * 20);

        if let Some(ref mut ds) = &mut drawing_state {
            for l in ds.draw_node_vprefix(model, node_id, &self.options)? {
                println!("{}{}{}{} ", cost_column_pad, profile_column_pad, prefix, l);
            }
        }

        // cost column
        let mut cost_column = if self.options.cost {
            Some(
                tags.cost
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(|c| format!("{:1$}", format!("{:?}:{}", c.0, c.1), 25))
                    .peekable(),
            )
        } else {
            None
        };

        // profile column
        let mut profile_column = tags.profile.map(|measure| {
            let ratio = measure.as_secs_f64() / self.profile_summary.as_ref().unwrap().sum.as_secs_f64();
            let ratio_for_color =
                measure.as_secs_f64() / self.profile_summary.as_ref().unwrap().max.as_secs_f64();
            let color = colorous::RED_YELLOW_GREEN.eval_continuous(1.0 - ratio_for_color);
            let color = ansi_term::Color::RGB(color.r, color.g, color.b);
            let label = format!(
                "{:7.3} ms/i {}  ",
                measure.as_secs_f64() * 1e3,
                color.bold().paint(format!("{:>4.1}%", ratio * 100.0))
            );
            std::iter::once(label)
        });

        // drawing column
        let mut drawing_lines: Box<dyn Iterator<Item = String>> = if let Some(ds) =
            drawing_state.as_mut()
        {
            let body = ds.draw_node_body(model, node_id, &self.options)?;
            let suffix = ds.draw_node_vsuffix(model, node_id, &self.options)?;
            let filler = ds.draw_node_vfiller(model, node_id)?;
            Box::new(body.into_iter().chain(suffix.into_iter()).chain(std::iter::repeat(filler)))
        } else {
            Box::new(std::iter::repeat(cost_column_pad.clone()))
        };

        macro_rules! prefix {
            () => {
                let cost = cost_column
                    .as_mut()
                    .map(|it| it.next().unwrap_or_else(|| cost_column_pad.to_string()))
                    .unwrap_or("".to_string());
                let profile = profile_column
                    .as_mut()
                    .map(|it| it.next().unwrap_or_else(|| profile_column_pad.to_string()))
                    .unwrap_or("".to_string());
                print!("{}{}{}{} ", cost, profile, prefix, drawing_lines.next().unwrap(),)
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
        for label in tags.labels.iter() {
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
                    let io = if let Some(id) =
                        model.input_outlets().iter().position(|n| n.node == node_id && n.slot == ix)
                    {
                        format!(
                            "{} {}",
                            Cyan.bold().paint(format!("MODEL INPUT #{}", id)).to_string(),
                            tags.model_input.as_deref().unwrap_or("")
                        )
                    } else if let Some(id) = model
                        .output_outlets()
                        .iter()
                        .position(|n| n.node == node_id && n.slot == ix)
                    {
                        format!(
                            "{} {}",
                            Yellow.bold().paint(format!("MODEL OUTPUT #{}", id)).to_string(),
                            tags.model_output.as_deref().unwrap_or("")
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
        for section in tags.sections {
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
        if let Some(tmodel) = model.downcast_ref::<TypedModel>() {
            for (label, sub, _, _) in tmodel.node(node_id).op.nested_models() {
                let prefix = drawing_lines.next().unwrap();
                let mut scope: TVec<_> = scope.into();
                scope.push((node_id, label.to_string()));
                self.render_prefixed(sub, &format!("{} [{}] ", prefix, label), &*scope)?
            }
        }
        while cost_column.as_mut().map(|cost| cost.peek().is_some()).unwrap_or(false) {
            prefix!();
            println!("");
        }
        Ok(())
    }

    pub fn from_model_and_options(
        model: &'a dyn Model,
        options: &'a DisplayParams,
    ) -> CliResult<DisplayGraph<'a>> {
        Self::from_model_prefix_and_options(model, [].as_ref(), options)
    }

    fn from_model_prefix_and_options(
        model: &'a dyn Model,
        prefix: &[(usize, String)],
        options: &'a DisplayParams,
    ) -> CliResult<DisplayGraph<'a>> {
        for n in 0..model.nodes_len() {
            for (label, sub, ins, outs) in model.node_op(n).nested_models() {
                let mut prefix: TVec<(usize, String)> = prefix.into();
                prefix.push((n, label.to_string()));
                let mut dg = Self::from_model_prefix_and_options(sub, &*prefix, options)?;
                ins.into_iter().enumerate().for_each(|(ix, i)| {
                    let qid = NodeQId(prefix.clone(), ix);
                    dg.tags.entry(qid).or_default().model_input = Some(i);
                });
                outs.into_iter().enumerate().for_each(|(ix, o)| {
                    let qid = NodeQId(prefix.clone(), ix);
                    dg.tags.entry(qid).or_default().model_output = Some(o);
                });
            }
        }
        let mut dg = DisplayGraph { tags: HashMap::new(), options, profile_summary: None };
        if dg.options.cost && prefix.len() == 0 {
            dg.extract_costs(model, &[], 1.0)?;
        }
        Ok(dg)
    }

    pub fn with_graph_def(
        self,
        model: &dyn Model,
        graph_def: &SomeGraphDef,
    ) -> CliResult<DisplayGraph<'a>> {
        match graph_def {
            SomeGraphDef::NoGraphDef => Ok(self),
            #[cfg(feature = "kaldi")]
            SomeGraphDef::Kaldi(kaldi) => self.with_kaldi(model, kaldi),
            #[cfg(feature = "onnx")]
            SomeGraphDef::Onnx(onnx, _) => self.with_onnx_model(model, onnx),
            #[cfg(feature = "tf")]
            SomeGraphDef::Tf(tf) => self.with_tf_graph_def(model, tf),
        }
    }

    /*
    fn get_context(&self, id: &[(usize, String)]) -> (&DisplayGraph<'a>, usize) {
    use std::ops::Deref;
    if id.len() == 1 {
    (&self, id[0].0)
    } else {
    self.node_nested_graphs
    .get(&id[0].0)
    .unwrap()
    .deref()
    .iter()
    .find(|sub| sub.0 == id[0].1)
    .unwrap()
    .1
    .get_context(&id[1..])
    }
    }
    */

    /*
    fn get_context_mut(&mut self, id: &[(usize, String)]) -> (&mut DisplayGraph<'a>, usize) {
        use std::ops::DerefMut;
        if id.len() == 1 {
            (&mut *self, id[0].0)
        } else {
            self.node_nested_graphs
                .get_mut(&id[0].0)
                .unwrap()
                .deref_mut()
                .iter_mut()
                .find(|sub| sub.0 == id[0].1)
                .unwrap()
                .1
                .get_context_mut(&id[1..])
        }
    }
    */

    /*
    pub fn node_name(&self, id: &[(usize, String)]) -> &str {
    let (ctx, id) = self.get_context(id);
    ctx.model.node_name(id)
    }

    pub fn add_node_label<S: Into<String>>(
        &mut self,
        id: &[(usize, String)],
        label: S,
    ) -> CliResult<()> {
        let (ctx, id) = self.get_context_mut(id);
        ctx.node_labels.entry(id).or_insert(vec![]).push(label.into());
        Ok(())
    }

    pub fn add_node_section(
        &mut self,
        id: &[(usize, String)],
        section: Vec<String>,
    ) -> CliResult<()> {
        let (ctx, id) = self.get_context_mut(id);
        ctx.node_sections.entry(id).or_insert(vec![]).push(section);
        Ok(())
    }

    pub fn set_node_color<S: Into<Style>>(
        &mut self,
        id: &[(usize, String)],
        color: S,
    ) -> CliResult<()> {
        let (ctx, id) = self.get_context_mut(id);
        ctx.node_color.insert(id, color.into());
        Ok(())
    }
    */

    #[cfg(feature = "kaldi")]
    pub fn with_kaldi(
        mut self,
        model: &dyn Model,
        proto_model: &tract_kaldi::KaldiProtoModel,
    ) -> CliResult<DisplayGraph<'a>> {
        use tract_kaldi::model::NodeLine;
        let bold = Style::new().bold();
        for (name, proto_node) in &proto_model.config_lines.nodes {
            if let Ok(node_id) = model.node_id_by_name(&*name) {
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
                self.node_mut(node_id.into()).sections.push(vs)
            }
        }
        Ok(self)
    }

    #[cfg(feature = "tf")]
    pub fn with_tf_graph_def(
        mut self,
        model: &dyn Model,
        graph_def: &GraphDef,
    ) -> CliResult<DisplayGraph<'a>> {
        let bold = Style::new().bold();
        for gnode in graph_def.node.iter() {
            if let Ok(node_id) = model.node_id_by_name(&gnode.name) {
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
                self.node_mut(node_id.into()).sections.push(v);
            }
        }
        Ok(self)
    }

    #[cfg(feature = "onnx")]
    pub fn with_onnx_model(
        mut self,
        model: &dyn Model,
        model_proto: &ModelProto,
    ) -> CliResult<DisplayGraph<'a>> {
        let bold = Style::new().bold();
        for gnode in model_proto.graph.as_ref().unwrap().node.iter() {
            let mut node_name = &gnode.name;
            if node_name == "" && gnode.output.len() > 0 {
                node_name = &gnode.output[0];
            }
            if let Ok(id) = model.node_id_by_name(&*node_name) {
                let mut v = vec![];
                for a in gnode.attribute.iter() {
                    let value = if let Some(t) = &a.t {
                        format!("{:?}", Tensor::try_from(t)?)
                    } else {
                        format!("{:?}", a)
                    };
                    v.push(format!("Attr {}: {:.240}", bold.paint(&a.name), value));
                }
                self.node_mut(id.into()).sections.push(v);
            }
        }
        Ok(self)
    }

    /*
    pub fn with_profiling_results(self, profile_data: ProfileData) -> DisplayGraph<'a> {
        DisplayGraph { profile_data: Some(profile_data), ..self }
    }
    */

    fn extract_costs(
        &mut self,
        model: &dyn Model,
        prefix: &[(usize, String)],
        multiplier: f64,
    ) -> CliResult<()> {
        if let Some(model) = model.downcast_ref::<TypedModel>() {
            for node_id in 0..model.nodes().len() {
                let inputs = model.node_input_facts(node_id)?;
                let cost = model.node(node_id).op.cost(&*inputs)?;
                self.node_mut(NodeQId(prefix.into(), node_id))
                    .cost
                    .replace(cost.into_iter().map(|(k, v)| (k, v * multiplier)).collect());

                let nested_subs = model.node(node_id).op.nested_models();
                let nested_multis = model.node(node_id).op.nested_model_multipliers(&*inputs);
                for ((name, sub, _,_), (_name, multi)) in nested_subs.iter().zip(nested_multis.iter()) {
                    let mut prefix:TVec<_> = prefix.into();
                    prefix.push((node_id, name.to_string()));
                    self.extract_costs(*sub, &*prefix, multiplier * multi)?;
                }
            }
        }
        Ok(())
    }

    /*
    pub fn total_cost(&self) -> CliResult<HashMap<Cost, TDim>> {
        use tract_num_traits::Zero;
        let mut total: HashMap<Cost, TDim> = HashMap::default();
        fn sum_cost(total: &mut HashMap<Cost, TDim>, dg: &DisplayGraph) {
            for nodes in dg.cost.as_ref().unwrap() {
                for cost in nodes {
                    let s = total.entry(cost.0).or_insert(TDim::zero());
                    *s = s.clone() + &cost.1;
                }
            }
            for node in dg.node_nested_graphs.values() {
                for sub in node {
                    sum_cost(total, &sub.1);
                }
            }
        }
        sum_cost(&mut total, self);
        Ok(total)
    }
    */
}
