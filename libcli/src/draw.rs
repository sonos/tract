use crate::display_params::DisplayParams;
use crate::model::Model;
use nu_ansi_term::{Color, Style};
use box_drawing::heavy::*;
use std::fmt;
use std::fmt::Write;
use tract_core::internal::*;

#[derive(Clone)]
pub struct Wire {
    pub outlet: OutletId,
    pub color: Option<Style>,
    pub should_change_color: bool,
    pub successors: Vec<InletId>,
}

impl fmt::Debug for Wire {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let s = format!("{:?} {:?}", self.outlet, self.successors);
        if let Some(c) = self.color {
            write!(fmt, "{}", c.paint(s))
        } else {
            write!(fmt, "{s}")
        }
    }
}

#[derive(Clone, Default)]
pub struct DrawingState {
    pub current_color: Style,
    pub latest_node_color: Style,
    pub wires: Vec<Wire>,
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
            .min_by_key(|&c| self.wires.iter().filter(|w| w.color == Some(*c)).count())
            .unwrap();
        self.current_color = *color;
        *color
    }

    fn inputs_to_draw(&self, model: &dyn Model, node: usize) -> Vec<OutletId> {
        model.node_inputs(node).to_vec()
    }

    fn passthrough_count(&self, node: usize) -> usize {
        self.wires.iter().filter(|w| w.successors.iter().any(|i| i.node != node)).count()
    }

    pub fn draw_node_vprefix(
        &mut self,
        model: &dyn Model,
        node: usize,
        _opts: &DisplayParams,
    ) -> TractResult<Vec<String>> {
        let mut lines = vec![String::new()];
        macro_rules! p { ($($args: expr),*) => { write!(lines.last_mut().unwrap(), $($args),*)?;} }
        macro_rules! ln {
            () => {
                lines.push(String::new())
            };
        }
        let passthrough_count = self.passthrough_count(node);
        /*
        println!("\n{}", model.node_format(node));
        for (ix, w) in self.wires.iter().enumerate() {
            println!(" * {} {:?}", ix, w);
        }
        */
        for (ix, &input) in model.node_inputs(node).iter().enumerate().rev() {
            let wire = self.wires.iter().position(|o| o.outlet == input).unwrap();
            let wanted = passthrough_count + ix;
            if wire != wanted {
                let little = wire.min(wanted);
                let big = wire.max(wanted);
                let moving = self.wires[little].clone();
                let must_clone = moving.successors.iter().any(|i| i.node != node);
                let offset = self
                    .wires
                    .iter()
                    .skip(little + 1)
                    .take(big - little)
                    .filter(|w| w.color.is_some())
                    .count()
                    + must_clone as usize;
                // println!("{}->{} (offset: {})", little, big, offset);
                if moving.color.is_some() && offset != 0 {
                    let color = moving.color.unwrap();
                    for w in &self.wires[0..little] {
                        if let Some(c) = w.color {
                            p!("{}", c.paint(VERTICAL));
                        }
                    }
                    // println!("offset: {}", offset);
                    p!("{}", color.paint(if must_clone { VERTICAL_RIGHT } else { UP_RIGHT }));
                    for _ in 0..offset - 1 {
                        p!("{}", color.paint(HORIZONTAL));
                    }
                    p!("{}", color.paint(DOWN_LEFT));
                }
                while self.wires.len() <= big {
                    self.wires.push(Wire { successors: vec![], ..self.wires[little] });
                }
                if must_clone {
                    self.wires[little].successors.retain(|&i| i != InletId::new(node, ix));
                    self.wires[big] = Wire {
                        successors: vec![InletId::new(node, ix)],
                        should_change_color: true,
                        ..self.wires[little]
                    };
                } else {
                    for i in little..big {
                        self.wires.swap(i, i + 1);
                    }
                }
                if moving.color.is_some() {
                    if big < self.wires.len() {
                        for w in &self.wires[big + 1..] {
                            if let Some(c) = w.color {
                                p!("{}", c.paint(VERTICAL));
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
        opts: &DisplayParams,
    ) -> TractResult<Vec<String>> {
        let mut lines = vec![String::new()];
        macro_rules! p { ($($args: expr),*) => { write!(lines.last_mut().unwrap(), $($args),*)?;} }
        macro_rules! ln {
            () => {
                lines.push(String::new())
            };
        }
        let inputs = self.inputs_to_draw(model, node);
        let passthrough_count = self.passthrough_count(node);
        let display = opts.konst || !model.node_const(node);
        if display {
            for wire in &self.wires[0..passthrough_count] {
                if let Some(color) = wire.color {
                    p!("{}", color.paint(VERTICAL));
                }
            }
        }
        let node_output_count = model.node_output_count(node);
        if display {
            self.latest_node_color = if !inputs.is_empty() {
                let wire0 = &self.wires[passthrough_count];
                if wire0.color.is_some() && !wire0.should_change_color {
                    wire0.color.unwrap()
                } else {
                    self.next_color()
                }
            } else {
                self.next_color()
            };
            match (inputs.len(), node_output_count) {
                (0, 1) => {
                    p!("{}", self.latest_node_color.paint(DOWN_RIGHT));
                }
                (1, 0) => {
                    p!("{}", self.latest_node_color.paint("╹"));
                }
                (u, d) => {
                    p!("{}", self.latest_node_color.paint(VERTICAL_RIGHT));
                    for _ in 1..u.min(d) {
                        p!("{}", self.latest_node_color.paint(VERTICAL_HORIZONTAL));
                    }
                    for _ in u..d {
                        p!("{}", self.latest_node_color.paint(DOWN_HORIZONTAL));
                    }
                    for _ in d..u {
                        p!("{}", self.latest_node_color.paint(UP_HORIZONTAL));
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

    pub fn draw_node_vfiller(&self, model: &dyn Model, node: usize) -> TractResult<String> {
        let mut s = String::new();
        for wire in &self.wires {
            if let Some(color) = wire.color {
                write!(&mut s, "{}", color.paint(VERTICAL))?;
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
        opts: &DisplayParams,
    ) -> TractResult<Vec<String>> {
        let mut lines = vec![];
        let passthrough_count = self.passthrough_count(node);
        let node_output_count = model.node_output_count(node);
        let node_color = self
            .wires
            .get(passthrough_count)
            .map(|w| w.color)
            .unwrap_or_else(|| Some(self.current_color()));
        self.wires.truncate(passthrough_count);
        for slot in 0..node_output_count {
            let outlet = OutletId::new(node, slot);
            let successors = model.outlet_successors(outlet).to_vec();
            let color = if !opts.konst && model.node_const(node) {
                None
            } else if slot == 0 && node_color.is_some() {
                Some(self.latest_node_color)
            } else {
                Some(self.next_color())
            };
            self.wires.push(Wire { outlet, color, successors, should_change_color: false });
        }
        let wires_before = self.wires.clone();
        self.wires.retain(|w| !w.successors.is_empty());
        for (wanted_at, w) in self.wires.iter().enumerate() {
            let is_at = wires_before.iter().position(|w2| w.outlet == w2.outlet).unwrap();
            if wanted_at < is_at {
                let mut s = String::new();
                for w in 0..wanted_at {
                    if let Some(color) = self.wires[w].color {
                        write!(&mut s, "{}", color.paint(VERTICAL))?;
                    }
                }
                if let Some(color) = self.wires[wanted_at].color {
                    write!(&mut s, "{}", color.paint(DOWN_RIGHT))?;
                    for w in is_at + 1..wanted_at {
                        if self.wires[w].color.is_some() {
                            write!(&mut s, "{}", color.paint(HORIZONTAL))?;
                        }
                    }
                    write!(&mut s, "{}", color.paint(UP_LEFT))?;
                    for w in is_at..self.wires.len() {
                        if let Some(color) = self.wires[w].color {
                            write!(&mut s, "{}", color.paint(VERTICAL))?;
                        }
                    }
                }
                lines.push(s);
            }
        }
        // println!("{:?}", self.wires);
        while lines.last().map(|s| s.trim()) == Some("") {
            lines.pop();
        }
        Ok(lines)
    }
}
