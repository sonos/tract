use box_drawing::light::*;
use ansi_term::{ Color, Style };
use tract_core::model::OutletId;
use tract_core::Model;
use CliResult;

pub fn render(model: &Model) -> CliResult<()> {
    let colors: &[Style] = &[
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

    let mut next_color: usize = 0;
    let mut wires: Vec<(OutletId, Style)> = vec![];
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
                    print!("{}", w.1.paint(VERTICAL));
                }
                print!("{}", wires[little].1.paint(UP_RIGHT));
                for _ in little + 1..big {
                    print!("{}", wires[little].1.paint(HORIZONTAL));
                }
                print!("{}", wires[little].1.paint(DOWN_LEFT));
                if big < wires.len() {
                    for w in &wires[big + 1..] {
                        print!("{}", w.1.paint(VERTICAL));
                    }
                }
                println!("");
                let w = wires.remove(wire);
                wires.insert(wanted, w);
            }
        }
        for wire in &wires[0..first_input_wire] {
            print!("{}", wire.1.paint(VERTICAL));
        }
        let node_color:Style = if inputs.len() == 1 && node.outputs.len() == 1 {
            wires[first_input_wire].1
        } else {
            let col = colors[next_color % colors.len()];
            next_color += 1;
            col
        };
        match (inputs.len(), node.outputs.len()) {
            (0, 1) => print!("{}", node_color.paint(DOWN_RIGHT)),
            (1, 0) => print!("{}", node_color.paint("╵")),
            (u, d) => {
                print!("{}", node_color.paint("┝"));
                for _ in 1..u.min(d) {
                    print!("{}", node_color.paint("┿"));
                }
                for _ in u..d {
                    print!("{}", node_color.paint("┯"));
                }
                for _ in d..u {
                    print!("{}", node_color.paint("┷"));
                }
            }
        }
        println!(
            " {} {} {}",
            node_color.paint(format!("{}", node.id)),
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
                let col = colors[next_color % colors.len()];
                next_color += 1;
                col
            };
            for _ in 0..output.successors.len() {
                wires.push((OutletId::new(node.id, ix), node_color));
            }
            if output.successors.len() > 1 {
                for wire in &wires[0..(first_input_wire + ix)] {
                    print!("{}", wire.1.paint(VERTICAL));
                }
                print!("{}", color.paint(VERTICAL_RIGHT));
                for _ in 02..output.successors.len() {
                    print!("{}", color.paint(DOWN_HORIZONTAL));
                }
                println!("{}", color.paint(DOWN_LEFT));
            }
        }
    }
    Ok(())
}
