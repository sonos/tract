use crate::display_params::DisplayParams;
use crate::model::Model;
use box_drawing::heavy::*;
use nu_ansi_term::{Color, Style};
use std::collections::HashSet;
use std::fmt::Write;
use tract_core::internal::*;

/// A wire that is not rendered (const node output when konst=false).
#[derive(Clone, Debug)]
struct HiddenWire {
    successors: Vec<InletId>,
}

/// A wire that occupies a visual column.
#[derive(Clone, Debug)]
struct VisibleWire {
    outlet: OutletId,
    color: Style,
    successors: Vec<InletId>,
    should_change_color: bool,
}

/// White circled number for model inputs: ⓪①②...⑳
pub fn circled_input(ix: usize) -> char {
    match ix {
        0 => '⓪',
        1..=20 => char::from_u32(0x2460 + (ix as u32 - 1)).unwrap(),
        _ => '○',
    }
}

/// Filled circled number for model outputs: ⓿❶❷...❿
pub fn circled_output(ix: usize) -> char {
    match ix {
        0 => '⓿',
        1..=10 => char::from_u32(0x2776 + (ix as u32 - 1)).unwrap(),
        _ => '●',
    }
}

#[derive(Clone, Default)]
pub struct DrawingState {
    hidden: Vec<HiddenWire>,
    visible: Vec<VisibleWire>, // index = visual column
    latest_node_color: Style,
    visited: HashSet<usize>,
}

impl DrawingState {
    fn next_color(&self) -> Style {
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
        *colors
            .iter()
            .min_by_key(|&c| self.visible.iter().filter(|w| w.color == *c).count())
            .unwrap()
    }

    /// Number of visible wires that pass through (have successors to nodes other than `node`).
    fn passthrough_count(&self, node: usize) -> usize {
        self.visible.iter().filter(|w| w.successors.iter().any(|i| i.node != node)).count()
    }

    /// Color of the last visible wire, or the latest node color.
    pub fn last_wire_color(&self) -> Style {
        self.visible.last().map(|w| w.color).unwrap_or(self.latest_node_color)
    }

    /// Render a filler line: one ┃ per visible wire.
    fn render_filler(&self) -> String {
        let mut s = String::new();
        for w in &self.visible {
            let _ = write!(s, "{}", w.color.paint(VERTICAL));
        }
        s
    }

    pub fn draw_node_vprefix(
        &mut self,
        model: &dyn Model,
        node: usize,
        _opts: &DisplayParams,
    ) -> TractResult<Vec<String>> {
        let mut lines = vec![];

        // Prune wires whose only remaining successors are all already visited.
        self.visible.retain(|w| w.successors.iter().any(|i| !self.visited.contains(&i.node)));
        self.hidden.retain(|w| w.successors.iter().any(|i| !self.visited.contains(&i.node)));

        // Build target layout: passthroughs in current order, then visible inputs in input order.
        let inputs = model.node_inputs(node);
        let mut passthroughs: Vec<VisibleWire> = Vec::new();
        let mut input_wires: Vec<Option<VisibleWire>> = vec![None; inputs.len()];

        for w in &self.visible {
            // Check if this wire feeds any input of this node
            let mut matched_input = None;
            for (ix, &inlet) in inputs.iter().enumerate() {
                if w.outlet == inlet {
                    matched_input = Some(ix);
                    break;
                }
            }

            if let Some(ix) = matched_input {
                let this_inlet = InletId::new(node, ix);
                let must_clone = w.successors.iter().any(|i| *i != this_inlet);
                if must_clone {
                    // Wire feeds this node AND others: clone it.
                    // Original (with other successors) stays as passthrough.
                    let mut pass_wire = w.clone();
                    pass_wire.successors.retain(|i| *i != this_inlet);
                    passthroughs.push(pass_wire);
                    input_wires[ix] = Some(VisibleWire {
                        outlet: w.outlet,
                        color: w.color,
                        successors: vec![this_inlet],
                        should_change_color: true,
                    });
                } else {
                    // Wire feeds only this node: move entirely to input position.
                    input_wires[ix] = Some(w.clone());
                }
            } else {
                passthroughs.push(w.clone());
            }
        }

        // Target = passthroughs ++ visible input wires
        let pt = passthroughs.len();
        let mut target: Vec<VisibleWire> = passthroughs;
        for w in input_wires.iter().flatten() {
            target.push(w.clone());
        }

        // Build working state with empty slots for the input region.
        // Cols 0..pt are passthroughs (occupied), cols pt..target.len() start empty.
        let n_inputs_visible = input_wires.iter().filter(|w| w.is_some()).count();
        let total_cols = pt + n_inputs_visible;
        let mut slots: Vec<Option<VisibleWire>> = Vec::with_capacity(total_cols);
        for w in &self.visible {
            slots.push(Some(w.clone()));
        }
        while slots.len() < total_cols {
            slots.push(None); // empty reserved slots
        }

        // Process inputs right to left. For each input:
        // - Find the wire in `slots` (by outlet)
        // - Compute its target column in the final layout
        // - Render the routing line and update slots
        for (ix, &inlet) in inputs.iter().enumerate().rev() {
            let Some(ref input_wire) = input_wires[ix] else { continue };

            let target_col = target
                .iter()
                .position(|w| w.outlet == inlet && w.successors.iter().any(|i| i.node == node))
                .unwrap();

            let cur_col =
                match slots.iter().position(|s| s.as_ref().is_some_and(|w| w.outlet == inlet)) {
                    Some(c) => c,
                    None => continue,
                };

            let must_clone = input_wire.should_change_color; // proxy: cloned wires have this set

            if cur_col == target_col && !must_clone {
                continue;
            }

            // Render the routing line from cur_col to target_col.
            let mut s = String::new();
            let color = slots[cur_col].as_ref().unwrap().color;
            let from = cur_col.min(target_col);
            let to = cur_col.max(target_col);

            // Leading verticals (cols before the leftmost endpoint)
            for w in slots[..from].iter().flatten() {
                let _ = write!(s, "{}", w.color.paint(VERTICAL));
            }

            if must_clone {
                // Split: ┣ at cur_col, horizontals in between, ┓ at target_col
                let _ = write!(s, "{}", color.paint(VERTICAL_RIGHT));
            } else {
                // Swap: ┗ at cur_col, horizontals in between, ┓ at target_col
                let _ = write!(s, "{}", color.paint(UP_RIGHT));
            }
            for _ in from + 1..to {
                let _ = write!(s, "{}", color.paint(HORIZONTAL));
            }
            let _ = write!(s, "{}", color.paint(DOWN_LEFT));

            // Trailing verticals (cols after the rightmost endpoint)
            for w in slots[to + 1..].iter().flatten() {
                let _ = write!(s, "{}", w.color.paint(VERTICAL));
            }

            lines.push(s);

            // Update slots: place the wire/clone at target_col
            if must_clone {
                // Original stays at cur_col, clone goes to target_col
                slots[target_col] = Some(input_wire.clone());
            } else {
                // Move: remove from cur_col, place at target_col
                slots[cur_col] = None;
                slots[target_col] = Some(input_wire.clone());
            }
        }

        // Set final state
        self.visible = target;

        lines.retain(|l: &String| !l.trim().is_empty());
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

        let inputs = model.node_inputs(node).to_vec();
        let passthrough_count = self.passthrough_count(node);
        let display = opts.konst || !model.node_const(node);

        if display {
            // Draw passthrough verticals
            for w in &self.visible[..passthrough_count] {
                p!("{}", w.color.paint(VERTICAL));
            }

            let node_output_count = model.node_output_count(node);

            // Determine node color
            self.latest_node_color = if !inputs.is_empty() && passthrough_count < self.visible.len()
            {
                let wire0 = &self.visible[passthrough_count];
                if !wire0.should_change_color { wire0.color } else { self.next_color() }
            } else {
                self.next_color()
            };

            // Draw junction
            match (inputs.len(), node_output_count) {
                (0, 1) => {
                    // Source node: use circled number if it's a model input
                    let input_idx = model.input_outlets().iter().position(|o| o.node == node);
                    let symbol = match input_idx {
                        Some(i) => circled_input(i).to_string(),
                        _ => DOWN_RIGHT.to_string(),
                    };
                    p!("{}", self.latest_node_color.paint(symbol));
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

    pub fn draw_node_vfiller(&self, _model: &dyn Model, _node: usize) -> TractResult<String> {
        Ok(self.render_filler())
    }

    pub fn draw_node_vsuffix(
        &mut self,
        model: &dyn Model,
        node: usize,
        opts: &DisplayParams,
    ) -> TractResult<Vec<String>> {
        // Mark node as visited now that its inputs have been consumed.
        self.visited.insert(node);
        let mut lines = vec![];
        let passthrough_count = self.passthrough_count(node);
        let node_output_count = model.node_output_count(node);

        // Remove input wires (keep passthroughs)
        self.visible.truncate(passthrough_count);

        // Add output wires
        for slot in 0..node_output_count {
            let outlet = OutletId::new(node, slot);
            let successors = model.outlet_successors(outlet).to_vec();
            let color = if !opts.konst && model.node_const(node) {
                // Const node: wire goes to hidden, not visible
                self.hidden.push(HiddenWire { successors });
                continue;
            } else if slot == 0 {
                self.latest_node_color
            } else {
                self.next_color()
            };
            self.visible.push(VisibleWire {
                outlet,
                color,
                successors,
                should_change_color: false,
            });
        }

        // Mark model outputs with a circled number on a filler line.
        let model_outputs = model.output_outlets();
        let has_output_marker = self.visible.iter().any(|w| model_outputs.contains(&w.outlet));
        if has_output_marker {
            let mut s = String::new();
            for w in &self.visible {
                if model_outputs.contains(&w.outlet) {
                    let output_idx = model_outputs.iter().position(|o| *o == w.outlet);
                    let symbol = match output_idx {
                        Some(i) => circled_output(i),
                        _ => '●',
                    };
                    let _ = write!(s, "{}", w.color.paint(symbol.to_string()));
                } else {
                    let _ = write!(s, "{}", w.color.paint(VERTICAL));
                }
            }
            lines.push(s);
        }

        // Remove wires with no successors
        self.visible.retain(|w| !w.successors.is_empty());

        lines.retain(|l: &String| !l.trim().is_empty());
        Ok(lines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display_params::DisplayParams;
    use crate::model::Model;
    use tract_core::ops::identity::Identity;
    use tract_core::ops::math;

    fn strip_ansi(s: &str) -> String {
        let mut out = String::new();
        let mut in_escape = false;
        for c in s.chars() {
            if in_escape {
                if c == 'm' {
                    in_escape = false;
                }
            } else if c == '\x1b' {
                in_escape = true;
            } else {
                out.push(c);
            }
        }
        out
    }

    fn draw_all(model: &dyn Model, ds: &mut DrawingState, node: usize) -> Vec<String> {
        let opts = DisplayParams { konst: true, ..DisplayParams::default() };
        let mut lines = vec![];
        for l in ds.draw_node_vprefix(model, node, &opts).unwrap() {
            lines.push(strip_ansi(&l));
        }
        for l in ds.draw_node_body(model, node, &opts).unwrap() {
            lines.push(strip_ansi(&l));
        }
        for l in ds.draw_node_vsuffix(model, node, &opts).unwrap() {
            lines.push(strip_ansi(&l));
        }
        lines.retain(|l| !l.trim().is_empty());
        lines
    }

    /// Source → Identity (linear chain, no branching)
    #[test]
    fn linear_chain() -> TractResult<()> {
        let mut model = TypedModel::default();
        let s = model.add_source("s", f32::fact([1]))?;
        let _id = model.wire_node("id", Identity, &[s])?[0];
        model.auto_outputs()?;
        let mut ds = DrawingState::default();
        let lines0 = draw_all(&model, &mut ds, 0);
        assert_eq!(lines0, vec!["⓪"]); // circled 0 (first model input)
        let lines1 = draw_all(&model, &mut ds, 1);
        assert_eq!(lines1[0], VERTICAL_RIGHT); // ┣ (1 in, 1 out)
        assert!(lines1.len() == 2 && lines1[1] == "⓿"); // output marker
        Ok(())
    }

    /// Source → Add(source, source) — fan-in from one source to two inputs
    #[test]
    fn fanin_from_one_source() -> TractResult<()> {
        let mut model = TypedModel::default();
        let s = model.add_source("s", f32::fact([1]))?;
        let _add = model.wire_node("add", math::add(), &[s, s])?[0];
        model.auto_outputs()?;
        let mut ds = DrawingState::default();
        let lines0 = draw_all(&model, &mut ds, 0);
        assert_eq!(lines0, vec!["⓪"]); // circled 0 (first model input)
        let lines1 = draw_all(&model, &mut ds, 1);
        let joined = lines1.join("|");
        assert!(
            joined.contains(UP_HORIZONTAL), // ┻ (merge)
            "Expected merge pattern, got: {lines1:?}"
        );
        Ok(())
    }

    /// Two sources → Add → two consumers (fork)
    #[test]
    fn fork_after_merge() -> TractResult<()> {
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact([1]))?;
        let b = model.add_source("b", f32::fact([1]))?;
        let add = model.wire_node("add", math::add(), &[a, b])?[0];
        let _id1 = model.wire_node("id1", Identity, &[add])?[0];
        let _id2 = model.wire_node("id2", Identity, &[add])?[0];
        model.auto_outputs()?;
        let mut ds = DrawingState::default();
        draw_all(&model, &mut ds, 0); // source a
        draw_all(&model, &mut ds, 1); // source b
        let lines_add = draw_all(&model, &mut ds, 2); // add (2 inputs, 1 output)
        let joined = lines_add.join("|");
        assert!(
            joined.contains(UP_HORIZONTAL), // ┻ (2 inputs merge)
            "Expected merge in body, got: {lines_add:?}"
        );
        let lines_id1 = draw_all(&model, &mut ds, 3); // id1
        assert!(!lines_id1.is_empty(), "id1 should render");
        Ok(())
    }

    /// No blank lines in prefix output (regression for leading-empty-line bug)
    #[test]
    fn no_blank_prefix_lines() -> TractResult<()> {
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact([1]))?;
        let b = model.add_source("b", f32::fact([1]))?;
        let add = model.wire_node("add", math::add(), &[a, b])?[0];
        let _id = model.wire_node("id", Identity, &[add])?[0];
        model.auto_outputs()?;
        let opts = DisplayParams { konst: true, ..DisplayParams::default() };
        let mut ds = DrawingState::default();
        let order = model.eval_order()?;
        for &node in &order {
            let prefix = ds.draw_node_vprefix(&model, node, &opts).unwrap();
            for (i, l) in prefix.iter().enumerate() {
                let stripped = strip_ansi(l);
                assert!(
                    !stripped.trim().is_empty() || i == prefix.len() - 1,
                    "Blank line at position {i} in prefix for node {node}: {prefix:?}"
                );
            }
            ds.draw_node_body(&model, node, &opts).unwrap();
            ds.draw_node_vsuffix(&model, node, &opts).unwrap();
        }
        Ok(())
    }

    /// Filler width matches the number of visible wires (post-suffix state)
    #[test]
    fn filler_width_matches_visible() -> TractResult<()> {
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact([1]))?;
        let b = model.add_source("b", f32::fact([1]))?;
        let add = model.wire_node("add", math::add(), &[a, b])?[0];
        let _id1 = model.wire_node("id1", Identity, &[add])?[0];
        let _id2 = model.wire_node("id2", Identity, &[add])?[0];
        model.auto_outputs()?;
        let opts = DisplayParams { konst: true, ..DisplayParams::default() };
        let mut ds = DrawingState::default();
        let order = model.eval_order()?;
        for &node in &order {
            ds.draw_node_vprefix(&model, node, &opts).unwrap();
            ds.draw_node_body(&model, node, &opts).unwrap();
            ds.draw_node_vsuffix(&model, node, &opts).unwrap();
            let filler = ds.draw_node_vfiller(&model, node).unwrap();
            let filler_w = strip_ansi(&filler).chars().count();
            let visible_count = ds.visible.len();
            assert_eq!(
                filler_w, visible_count,
                "Filler width {filler_w} != visible wire count {visible_count} for node {node}"
            );
        }
        Ok(())
    }
}
