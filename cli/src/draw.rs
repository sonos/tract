use std::fmt::{Debug, Display};

use crate::display_graph::DisplayOptions;
use crate::{CliResult, SomeModel};
use ansi_term::{Color, Style};
use box_drawing::light::*;
use tract_core::model::{Model, Op, OutletId, TensorInfo};
use tract_core::ops::konst::Const;

pub fn render(model: &SomeModel, options: DisplayOptions) -> CliResult<()> {
    match model {
        SomeModel::Inference(m) => render_t(m, options),
        SomeModel::Typed(m) => render_t(m, options),
        SomeModel::Normalized(m) => render_t(m, options),
        SomeModel::Pulsed(_, m) => render_t(m, options),
    }
}

fn render_t<TI, O>(model: &Model<TI, O>, options: DisplayOptions) -> CliResult<()> 
where
    TI: TensorInfo,
    O: AsRef<Op> + AsMut<Op> + Display + Debug,
{
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
        let node = &model.nodes()[node];
        let inputs = if model.input_outlets()?.contains(&OutletId::new(node.id, 0)) {
            &[]
        } else {
            &*node.inputs
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
        let display = options.konst || !(node.op_is::<Const>());
        if display {
            for wire in &wires[0..first_input_wire] {
                print!("{}", wire.unwrap().1.paint(VERTICAL));
            }
        }
        let node_color: Style = if inputs.len() == 1 && node.outputs.len() == 1 {
            wires[first_input_wire].unwrap().1
        } else {
            let col = colors[next_color % colors.len()];
            next_color += 1;
            col
        };
        if display {
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
                node.op.as_ref().name(),
                node.name
            );
        }
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
            wires.push(Some((OutletId::new(node.id, ix), color, output.successors.len(), display)));
        }
    }
    Ok(())
}
