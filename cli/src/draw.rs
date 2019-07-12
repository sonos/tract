use crate::display_graph::DisplayOptions;
use crate::{CliResult, SomeModel};
use ansi_term::{Color, Style};
use box_drawing::light::*;
use tract_core::model::OutletId;
use tract_core::ops::konst::Const;

pub fn render(model: &SomeModel, options: DisplayOptions) -> CliResult<()> {
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
    let mut wires: Vec<Option<(OutletId, Style, usize, bool)>> = vec![];
    for node in model.eval_order()? {
        let inputs = if model.input_outlets()?.contains(&OutletId::new(node, 0)) {
            &[]
        } else {
            model.node_inputs(node)
        };
        let mut memory_wires: Vec<_> = wires.clone();
        for i in inputs {
            let pos = memory_wires.iter().position(|w| *i == w.as_ref().unwrap().0).unwrap();
            memory_wires[pos].as_mut().unwrap().2 -= 1;
        }
        memory_wires.retain(|w| w.unwrap().2 > 0);
        let first_input_wire = memory_wires.len();
        while wires.len() < first_input_wire + inputs.len() {
            wires.push(None);
        }
        for (ix, &input) in inputs.iter().enumerate().rev() {
            let wire = wires.iter().position(|o| o.is_some() && o.unwrap().0 == input).unwrap();
            let wanted = first_input_wire + ix;
            if wire != wanted {
                let little = wire.min(wanted);
                let big = wire.max(wanted);
                let moving = wires[little].unwrap();
                if moving.3 {
                    for w in &wires[0..little] {
                        if let Some(w) = w {
                            print!("{}", w.1.paint(VERTICAL));
                        } else {
                            print!(" ");
                        }
                    }
                    if moving.2 == 1 {
                        print!("{}", moving.1.paint(UP_RIGHT));
                    } else {
                        print!("{}", moving.1.paint(VERTICAL_RIGHT));
                    };
                    for _ in little + 1..big {
                        print!("{}", moving.1.paint(HORIZONTAL));
                    }
                    print!("{}", moving.1.paint(DOWN_LEFT));
                }
                let w = wires[little];
                wires[little].as_mut().unwrap().2 -= 1;
                if wires[little].unwrap().2 == 0 {
                    for i in little..big {
                        wires[i] = wires[i + 1];
                    }
                }
                wires[wanted] = w;
                if moving.3 {
                    if big < wires.len() {
                        for w in &wires[big + 1..] {
                            if let Some(w) = w {
                                if w.3 {
                                    print!("{}", w.1.paint(VERTICAL));
                                } else {
                                    print!(" ");
                                }
                            } else {
                                print!(" ");
                            }
                        }
                    }
                    println!("");
                }
            }
        }
        let display = options.konst || !(model.node_op(node).is::<Const>());
        if display {
            for wire in &wires[0..first_input_wire] {
                print!("{}", wire.unwrap().1.paint(VERTICAL));
            }
        }
        let node_output_count = model.node_output_count(node);
        let node_color: Style = if inputs.len() == 1 && node_output_count == 1 {
            wires[first_input_wire].unwrap().1
        } else {
            let col = colors[next_color % colors.len()];
            next_color += 1;
            col
        };
        if display {
            match (inputs.len(), node_output_count) {
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
                node_color.paint(format!("{}", node)),
                model.node_op_name(node),
                model.node_name(node),
            );
        }
        wires.truncate(first_input_wire);
        for ix in 0..node_output_count {
            let outlet = OutletId::new(node, ix);
            let successors = model.outlet_successors(outlet);
            if successors.len() == 0 {
                continue;
            }
            let color = if ix == 0 {
                node_color
            } else {
                let col = colors[next_color % colors.len()];
                next_color += 1;
                col
            };
            wires.push(Some((outlet, color, successors.len(), display)));
        }
    }
    Ok(())
}
