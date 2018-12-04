use box_drawing::light::*;
use colored::Color;
use colored::Colorize;
use tract_core::model::OutletId;
use tract_core::Model;
use CliResult;

const COLORS: &'static [Color] = &[
    Color::Red,
    Color::Green,
    Color::Yellow,
    Color::Blue,
    Color::Magenta,
    Color::Cyan,
    Color::White,
    Color::BrightRed,
    Color::BrightGreen,
    Color::BrightYellow,
    Color::BrightBlue,
    Color::BrightMagenta,
    Color::BrightCyan,
    Color::BrightWhite,
];

pub fn render(model: &Model) -> CliResult<()> {
    let mut next_color: usize = 0;
    let mut wires: Vec<(OutletId, Color)> = vec![];
    for node in model.eval_order()? {
        let node = &model.nodes()[node];
        let inputs = if model.inputs()?.contains(&OutletId::new(node.id,0)) {
            &[]
        } else {
            &*node.inputs
        };
        let first_input_wire = wires.len() - inputs.len();
        for (ix, &input) in inputs.iter().enumerate().rev() {
            let wire = wires.iter().rposition(|o| o.0 == input).unwrap();
            let wanted = first_input_wire + ix;
            if wire != wanted {
                let little = wire.min(wanted);
                let big = wire.max(wanted);
                for w in &wires[0..little] {
                    print!("{}", VERTICAL.color(w.1));
                }
                print!("{}", UP_RIGHT.color(wires[little].1));
                for _ in little + 1..big {
                    print!("{}", HORIZONTAL.color(wires[little].1));
                }
                print!("{}", DOWN_LEFT.color(wires[little].1));
                if big < wires.len() {
                    for w in &wires[big + 1..] {
                        print!("{}", VERTICAL.color(w.1));
                    }
                }
                println!("");
                let w = wires.remove(wire);
                wires.insert(wanted, w);
            }
        }
        for wire in &wires[0..first_input_wire] {
            print!("{}", VERTICAL.color(wire.1));
        }
        let node_color = if inputs.len() == 1 && node.outputs.len() == 1 {
            wires[first_input_wire].1
        } else {
            let col = COLORS[next_color % COLORS.len()];
            next_color += 1;
            col
        };
        match (inputs.len(), node.outputs.len()) {
            (0, 1) => print!("{}", DOWN_RIGHT.color(node_color)),
            (1, 0) => print!("{}", "╵".color(node_color)),
            (u, d) => {
                print!("{}", "┝".color(node_color));
                for _ in 1..u.min(d) {
                    print!("{}", "┿".color(node_color));
                }
                for _ in u..d {
                    print!("{}", "┯".color(node_color));
                }
                for _ in d..u {
                    print!("{}", "┷".color(node_color));
                }
            }
        }
        println!(
            " {} {} {}",
            format!("{}", node.id).color(node_color),
            node.op.name(),
            node.name
        );
        wires.truncate(first_input_wire);
        for (ix, output) in node.outputs.iter().enumerate() {
            if output.successors.len() == 0 {
                continue;
            }
            let color = if ix == 0 {
                node_color
            } else {
                let col = COLORS[next_color % COLORS.len()];
                next_color += 1;
                col
            };
            for _ in 0..output.successors.len() {
                wires.push((OutletId::new(node.id, ix), node_color));
            }
            if output.successors.len() > 1 {
                for wire in &wires[0..(first_input_wire + ix)] {
                    print!("{}", VERTICAL.color(wire.1));
                }
                print!("{}", VERTICAL_RIGHT.color(color));
                for _ in 02..output.successors.len() {
                    print!("{}", DOWN_HORIZONTAL.color(color));
                }
                println!("{}", DOWN_LEFT.color(color));
            }
        }
    }
    Ok(())
}
