use crate::display_graph::DisplayOptions;
use crate::{CliResult, Model};
use ansi_term::{Color, Style};
use box_drawing::light::*;
use std::fmt::Write;
use tract_core::model::{InletId, OutletId};
use tract_core::ops::konst::Const;

#[derive(Clone, Debug)]
struct Wire {
    outlet: OutletId,
    color: Style,
    successors: Vec<InletId>,
    display: bool,
}

#[derive(Clone, Default)]
pub struct DrawingState {
    current_color: Style,
    wires: Vec<Wire>,
}

impl DrawingState {
    fn current_color(&self) -> Style {
        self.current_color
    }

    fn next_color(&mut self) -> Style {
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
        let color = colors
            .iter()
            .min_by_key(|&c| self.wires.iter().filter(|w| w.display && w.color == *c).count())
            .unwrap();
        self.current_color = *color;
        *color
    }

    fn inputs_to_draw(&self, model: &dyn Model, node: usize) -> Vec<OutletId> {
        model
            .node_inputs(node)
            .iter()
            .cloned()
            .filter(|o| self.wires.iter().find(|w| w.outlet == *o).unwrap().display)
            .collect()
    }

    fn passthrough_count(&self, node: usize) -> usize {
        self.wires.iter().filter(|w| w.successors.iter().find(|i| i.node != node).is_some()).count()
    }

    pub fn draw_node_vprefix(
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
        let passthrough_count = self.passthrough_count(node);
        //        println!("{:?}", self.wires);
        for (ix, &input) in model.node_inputs(node).iter().enumerate().rev() {
            let wire = self.wires.iter().position(|o| o.outlet == input).unwrap();
            let wanted = passthrough_count + ix;
            if wire != wanted {
                let little = wire.min(wanted);
                let big = wire.max(wanted);
                let moving = self.wires[little].clone();
                let must_clone = moving.successors.iter().find(|i| i.node != node).is_some();
                //                println!("{}->{}", little, big);
                if moving.display
                    && (must_clone || self.wires[little + 1..big].iter().any(|w| w.display))
                {
                    for w in &self.wires[0..little] {
                        if w.display {
                            p!("{}", w.color.paint(VERTICAL));
                        }
                    }
                    if must_clone {
                        p!("{}", moving.color.paint(VERTICAL_RIGHT));
                        for w in little + 1..big {
                            if self.wires[w].display {
                                p!("{}", moving.color.paint(HORIZONTAL));
                            }
                        }
                    } else {
                        p!("{}", moving.color.paint(UP_RIGHT));
                        for w in little + 1..big - 1 {
                            if self.wires[w].display {
                                p!("{}", moving.color.paint(HORIZONTAL));
                            }
                        }
                    };
                    p!("{}", moving.color.paint(DOWN_LEFT));
                }
                while self.wires.len() <= big {
                    self.wires.push(Wire { successors: vec![], ..self.wires[little] });
                }
                if must_clone {
                    self.wires[little].successors.retain(|&i| i != InletId::new(node, ix));
                    self.wires[big] =
                        Wire { successors: vec![InletId::new(node, ix)], ..self.wires[little] };
                } else {
                    for i in little..big {
                        self.wires.swap(i, i + 1);
                    }
                }
                if moving.display {
                    if big < self.wires.len() {
                        for w in &self.wires[big + 1..] {
                            if w.display {
                                p!("{}", w.color.paint(VERTICAL));
                            } else {
                                p!(" ");
                            }
                        }
                    }
                    ln!();
                }
            }
        }
        while lines.last().map(|s| s.trim()) == Some("") {
            lines.pop();
        }
        Ok(lines)
    }

    pub fn draw_node_body(
        &mut self,
        model: &dyn Model,
        node: usize,
        opts: &DisplayOptions,
    ) -> CliResult<Vec<String>> {
        let mut lines = vec![String::new()];
        macro_rules! p { ($($args: expr),*) => { write!(lines.last_mut().unwrap(), $($args),*)?;} }
        macro_rules! ln {
            () => {
                lines.push(String::new())
            };
        };
        let inputs = self.inputs_to_draw(model, node);
        let passthrough_count = self.passthrough_count(node);
        let display = opts.konst || !(model.node_op(node).is::<Const>());
        if display {
            for wire in &self.wires[0..passthrough_count] {
                if wire.display {
                    p!("{}", wire.color.paint(VERTICAL));
                }
            }
        }
        let node_output_count = model.node_output_count(node);
        if display {
            let node_color: Style = if inputs.len() > 0 {
                self.wires[passthrough_count].color
            } else {
                self.next_color()
            };
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
        while lines.last().map(|s| s.trim()) == Some("") {
            lines.pop();
        }
        Ok(lines)
    }

    pub fn draw_node_vfiller(&self, model: &dyn Model, node: usize) -> CliResult<String> {
        let mut s = String::new();
        for wire in &self.wires {
            if wire.display {
                write!(&mut s, "{}", wire.color.paint(VERTICAL))?;
            }
        }
        for _ in self.wires.len()..model.node_output_count(node) {
            write!(&mut s, " ")?;
        }
        Ok(s)
    }

    pub fn draw_node_vsuffix(
        &mut self,
        model: &dyn Model,
        node: usize,
        opts: &DisplayOptions,
    ) -> CliResult<Vec<String>> {
        let mut lines = vec![];
        let passthrough_count = self.passthrough_count(node);
        let node_output_count = model.node_output_count(node);
        let node_color = self
            .wires
            .get(passthrough_count)
            .map(|w| w.color)
            .unwrap_or_else(|| self.current_color());
        self.wires.truncate(passthrough_count);
        let display = opts.konst || !(model.node_op(node).is::<Const>());
        for slot in 0..node_output_count {
            let outlet = OutletId::new(node, slot);
            let successors = model.outlet_successors(outlet).to_vec();
            let color = if slot == 0 { node_color } else { self.next_color() };
            self.wires.push(Wire { outlet, color, successors, display });
        }
        let wires_before = self.wires.clone();
        self.wires.retain(|w| w.successors.len() > 0);
        for (wanted_at, w) in self.wires.iter().enumerate() {
            let is_at = wires_before.iter().position(|w2| w.outlet == w2.outlet).unwrap();
            if wanted_at < is_at {
                let mut s = String::new();
                for w in 0..wanted_at {
                    if self.wires[w].display {
                        write!(&mut s, "{}", self.wires[w].color.paint(VERTICAL))?;
                    }
                }
                let color = self.wires[wanted_at].color;
                write!(&mut s, "{}", color.paint(DOWN_RIGHT))?;
                for w in is_at + 1..wanted_at {
                    if self.wires[w].display {
                        write!(&mut s, "{}", color.paint(HORIZONTAL))?;
                    }
                }
                write!(&mut s, "{}", color.paint(UP_LEFT))?;
                for w in is_at..self.wires.len() {
                    if self.wires[w].display {
                        write!(&mut s, "{}", self.wires[w].color.paint(VERTICAL))?;
                    }
                }
                lines.push(s);
            }
        }
        while lines.last().map(|s| s.trim()) == Some("") {
            lines.pop();
        }
        Ok(lines)
    }
}
