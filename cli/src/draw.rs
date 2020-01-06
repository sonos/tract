use crate::display_graph::DisplayOptions;
use crate::{CliResult, Model};
use ansi_term::{Color, Style};
use box_drawing::light::*;
use tract_core::model::OutletId;
use tract_core::ops::konst::Const;

#[derive(Clone, Copy)]
struct Wire {
    outlet: OutletId,
    color: Style,
    pos: usize,
    display: bool,
}

pub fn render(model: &dyn Model, options: DisplayOptions) -> CliResult<()> {
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

    let mut next_color_mem: usize = 0;
    let mut next_color = || {
        let col = colors[next_color_mem % colors.len()];
        next_color_mem += 1;
        col
    };

    let mut wires: Vec<Option<Wire>> = vec![];
    for node in model.eval_order()? {
        let inputs = if model.input_outlets().contains(&OutletId::new(node, 0)) {
            &[]
        } else {
            model.node_inputs(node)
        };
        let mut memory_wires: Vec<_> = wires.clone();
        for i in inputs {
            let pos = memory_wires.iter().position(|w| *i == w.as_ref().unwrap().outlet).unwrap();
            memory_wires[pos].as_mut().unwrap().pos -= 1;
        }
        memory_wires.retain(|w| w.unwrap().pos > 0);
        let first_input_wire = memory_wires.len();
        while wires.len() < first_input_wire + inputs.len() {
            wires.push(None);
        }
        for (ix, &input) in inputs.iter().enumerate().rev() {
            let wire =
                wires.iter().position(|o| o.is_some() && o.unwrap().outlet == input).unwrap();
            let wanted = first_input_wire + ix;
            if wire != wanted {
                let little = wire.min(wanted);
                let big = wire.max(wanted);
                let moving = wires[little].unwrap();
                if moving.display {
                    for w in &wires[0..little] {
                        if let Some(w) = w {
                            print!("{}", w.color.paint(VERTICAL));
                        } else {
                            print!(" ");
                        }
                    }
                    if moving.pos == 1 {
                        print!("{}", moving.color.paint(UP_RIGHT));
                    } else {
                        print!("{}", moving.color.paint(VERTICAL_RIGHT));
                    };
                    for _ in little + 1..big {
                        print!("{}", moving.color.paint(HORIZONTAL));
                    }
                    print!("{}", moving.color.paint(DOWN_LEFT));
                }
                let w = wires[little];
                wires[little].as_mut().unwrap().pos -= 1;
                if wires[little].unwrap().pos == 0 {
                    for i in little..big {
                        wires[i] = wires[i + 1];
                    }
                }
                wires[wanted] = w;
                if moving.pos != 1 {
                    wires[wanted].as_mut().unwrap().color = next_color();
                }
                if moving.display {
                    if big < wires.len() {
                        for w in &wires[big + 1..] {
                            if let Some(w) = w {
                                if w.display {
                                    print!("{}", w.color.paint(VERTICAL));
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
                print!("{}", wire.unwrap().color.paint(VERTICAL));
            }
        }
        let node_output_count = model.node_output_count(node);
        let node_color: Style =
            if inputs.len() > 0 { wires[first_input_wire].unwrap().color } else { next_color() };
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
                model.node_op(node).name(),
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
            let color = if ix == 0 { node_color } else { next_color() };
            wires.push(Some(Wire { outlet, color, pos: successors.len(), display }));
        }
    }
    Ok(())
}
