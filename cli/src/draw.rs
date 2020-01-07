use crate::display_graph::DisplayOptions;
use crate::{CliResult, Model};
use ansi_term::{Color, Style};
use box_drawing::light::*;
use std::fmt::Write;
use tract_core::model::OutletId;
use tract_core::ops::konst::Const;

#[derive(Clone, Default)]
pub struct DrawingState {
    next_color_mem: usize,
    wires: Vec<Option<Wire>>,
}

impl DrawingState {
    fn current_color(&self) -> Style {
        let colors = &[
            Color::Red.normal(),
            Color::Green.normal(),
            Color::Yellow.normal(),
            Color::Blue.normal(),
            Color::Purple.normal(),
            Color::Cyan.normal(),
            Color::White.normal(),
            Color::Red.bold(),
            Color::Green.bold(),
            Color::Yellow.bold(),
            Color::Blue.bold(),
            Color::Purple.bold(),
            Color::Cyan.bold(),
            Color::White.bold(),
        ];
        colors[self.next_color_mem % colors.len()]
    }
    fn next_color(&mut self) -> Style {
        self.next_color_mem += 1;
        self.current_color()
    }

    fn inputs<'a, 'm: 'a>(&self, model: &'m dyn Model, node: usize) -> &'a [OutletId] {
        if model.input_outlets().contains(&OutletId::new(node, 0)) {
            &[]
        } else {
            model.node_inputs(node)
        }
    }

    fn first_input_wire(&self, model: &dyn Model, node: usize) -> usize {
        let mut memory_wires: Vec<_> = self.wires.clone();
        for i in self.inputs(model, node) {
            let pos = memory_wires.iter().position(|w| *i == w.as_ref().unwrap().outlet).unwrap();
            memory_wires[pos].as_mut().unwrap().pos -= 1;
        }
        memory_wires.retain(|w| w.unwrap().pos > 0);
        memory_wires.len()
    }

    pub fn draw_node(
        &mut self,
        model: &dyn Model,
        node: usize,
        opts: &DisplayOptions,
    ) -> CliResult<Vec<String>> {
        let mut all = self.draw_node_vprefix(model, node, opts)?;
        all.extend(self.draw_node_body(model, node, opts)?);
        all.extend(self.draw_node_vsuffix(model, node, opts)?);
        Ok(all)
    }

    fn draw_node_vprefix(
        &mut self,
        model: &dyn Model,
        node: usize,
        _opts: &DisplayOptions,
    ) -> CliResult<Vec<String>> {
        let mut lines = vec![String::new()];
        macro_rules! p { ($($args: expr),*) => { write!(lines.last_mut().unwrap(), $($args),*)?;} }
        macro_rules! ln {
            () => {
                lines.push(String::new())
            };
        };
        let inputs = self.inputs(model, node);
        let mut memory_wires: Vec<_> = self.wires.clone();
        for i in inputs {
            let pos = memory_wires.iter().position(|w| *i == w.as_ref().unwrap().outlet).unwrap();
            memory_wires[pos].as_mut().unwrap().pos -= 1;
        }
        memory_wires.retain(|w| w.unwrap().pos > 0);
        let first_input_wire = memory_wires.len();
        while self.wires.len() < first_input_wire + inputs.len() {
            self.wires.push(None);
        }
        for (ix, &input) in inputs.iter().enumerate().rev() {
            let wire =
                self.wires.iter().position(|o| o.is_some() && o.unwrap().outlet == input).unwrap();
            let wanted = first_input_wire + ix;
            if wire != wanted {
                let little = wire.min(wanted);
                let big = wire.max(wanted);
                let moving = self.wires[little].unwrap();
                if moving.display {
                    for w in &self.wires[0..little] {
                        if let Some(w) = w {
                            p!("{}", w.color.paint(VERTICAL));
                        } else {
                            p!(" ");
                        }
                    }
                    if moving.pos == 1 {
                        p!("{}", moving.color.paint(UP_RIGHT));
                    } else {
                        p!("{}", moving.color.paint(VERTICAL_RIGHT));
                    };
                    for _ in little + 1..big {
                        p!("{}", moving.color.paint(HORIZONTAL));
                    }
                    p!("{}", moving.color.paint(DOWN_LEFT));
                }
                let w = self.wires[little];
                self.wires[little].as_mut().unwrap().pos -= 1;
                if self.wires[little].unwrap().pos == 0 {
                    for i in little..big {
                        self.wires[i] = self.wires[i + 1];
                    }
                }
                self.wires[wanted] = w;
                if moving.pos != 1 {
                    self.wires[wanted].as_mut().unwrap().color = self.next_color();
                }
                if moving.display {
                    if big < self.wires.len() {
                        for w in &self.wires[big + 1..] {
                            if let Some(w) = w {
                                if w.display {
                                    p!("{}", w.color.paint(VERTICAL));
                                } else {
                                    p!(" ");
                                }
                            } else {
                                p!(" ");
                            }
                        }
                    }
                    ln!();
                }
            }
        }
        while lines.last().map(|s| &**s) == Some("") {
            lines.pop();
        }
        Ok(lines)
    }

    fn draw_node_body(
        &mut self,
        model: &dyn Model,
        node: usize,
        opts: &DisplayOptions,
    ) -> CliResult<Vec<String>> {
        println!("node {}", node);
        let mut lines = vec![String::new()];
        macro_rules! p { ($($args: expr),*) => { write!(lines.last_mut().unwrap(), $($args),*)?;} }
        macro_rules! ln {
            () => {
                lines.push(String::new())
            };
        };
        println!("node {:?}", model.node_format(node));
        let inputs = self.inputs(model, node);
        dbg!(inputs);
        let first_input_wire = self.first_input_wire(model, node);
        dbg!(first_input_wire);
        dbg!(&self.wires);
        let display = opts.konst || !(model.node_op(node).is::<Const>());
        if display {
            for wire in &self.wires[0..first_input_wire] {
                p!("{}", wire.unwrap().color.paint(VERTICAL));
            }
        }
        let node_output_count = model.node_output_count(node);
        let node_color: Style = if inputs.len() > 0 {
            self.wires[first_input_wire].unwrap().color
        } else {
            self.next_color()
        };
        if display {
            match (inputs.len(), node_output_count) {
                (0, 1) => p!("{}", node_color.paint(DOWN_RIGHT)),
                (1, 0) => p!("{}", node_color.paint("╵")),
                (u, d) => {
                    p!("{}", node_color.paint("┝"));
                    for _ in 1..u.min(d) {
                        p!("{}", node_color.paint("┿"));
                    }
                    for _ in u..d {
                        p!("{}", node_color.paint("┯"));
                    }
                    for _ in d..u {
                        p!("{}", node_color.paint("┷"));
                    }
                }
            }
            ln!();
        }
        while lines.last().map(|s| &**s) == Some("") {
            lines.pop();
        }
        Ok(lines)
    }

    fn draw_node_vsuffix(
        &mut self,
        model: &dyn Model,
        node: usize,
        opts: &DisplayOptions,
    ) -> CliResult<Vec<String>> {
        let first_input_wire = self.first_input_wire(model, node);
        let node_output_count = model.node_output_count(node);
        let display = opts.konst || !(model.node_op(node).is::<Const>());
        let node_color = self
            .wires
            .get(first_input_wire)
            .map(|w| w.unwrap().color)
            .unwrap_or_else(|| self.current_color());
        self.wires.truncate(first_input_wire);
        for ix in 0..node_output_count {
            let outlet = OutletId::new(node, ix);
            let successors = model.outlet_successors(outlet);
            if successors.len() == 0 {
                continue;
            }
            let color = if ix == 0 { node_color } else { self.next_color() };
            self.wires.push(Some(Wire { outlet, color, pos: successors.len(), display }));
        }
        Ok(vec![])
    }
}

#[derive(Clone, Copy, Debug)]
struct Wire {
    outlet: OutletId,
    color: Style,
    pos: usize,
    display: bool,
}

pub fn render(model: &dyn Model, options: DisplayOptions) -> CliResult<()> {
    let mut state = DrawingState::default();
    for node in model.eval_order()? {
        for line in state.draw_node(model, node, &options)? {
            println!("{}", line);
        }
    }
    Ok(())
}
