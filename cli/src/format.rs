extern crate tfdeploy;

use std::cmp;

use tfdeploy::tfpb;

use errors::*;
use format;


/// Returns the number of hidden escape codes in the string.
pub fn hidden_len(s: &String) -> usize {
    let mut step = 0;
    let mut hidden = 0;

    for c in s.chars() {
        step = match (step, c) {
            (0, '\u{1b}') => 1,
            (1, '[') => 2,
            (1, _) => 0,
            (2, 'm') => 3,
            _ => step
        };

        if step > 0 {
            hidden += 1;
        }

        if step == 3 {
            step = 0;
        }
    }

    hidden
}


/// Prints a box containing arbitrary information about a node.
pub fn print_box(id: String, op: String, name: String, status: String, sections: Vec<Vec<String>>) {
    use colored::Colorize;

    // Box size configuration.
    let small = 42;
    let large = 43;
    let tiny = 13;
    let total = small + large + tiny + 2;

    println!("┌─────┬{:─>3$}┬{:─>4$}┬{:─>5$}┐", "", "", "", small, large, tiny);

    println!(
        "│{:^5}│ Operation: {:4$} │ Name: {:5$} │ {:^6$} │",
        id.bold(),
        op.bold().blue(),
        name.bold(),
        status.bold(),
        small - 13,
        large - 8,
        tiny - 2 + 13
    );

    let sections: Vec<Vec<String>> = sections
        .into_iter()
        .filter(|s| s.len() > 0)
        .collect();

    if sections.len() == 0 {
        println!("└─────┴{:─>3$}┴{:─>4$}┴{:─>5$}┘", "", "", "", small, large, tiny);
    } else {
        println!("└─────┼{:─>3$}┴{:─>4$}┴{:─>5$}┤", "", "", "", small, large, tiny);


        for (i, section) in sections.iter().enumerate() {
            if i > 0 {
                println!("{:6}├{:─>2$}┤", "", "", total);
            }

            for line in section {
                println!("{:6}│ {:2$} │", "", line, total - 2 + hidden_len(line));
            }
        }
        
        println!("{:6}└{:─>2$}┘", "", "", total);
    }
}


/// Splits a line into multiple lines to respect a two-column layout.
pub fn with_header(header: String, content: String, length: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = content.as_str();
    let mut first = true;

    let pad = header.len() - hidden_len(&header);

    while !cur.is_empty() {
        let (chunk, rest) = cur.split_at(cmp::min(length, cur.len()));
    
        if first {
            lines.push(format!("{} {}", header, chunk));
            first = false;
        } else {
            lines.push(format!("{:2$} {}", "", chunk, pad));
        }

        cur = rest;
    }
    
    lines
}


/// Returns information about a node.
pub fn node_info(
    node: &tfdeploy::Node,
    graph: &tfpb::graph::GraphDef,
    state: &::tfdeploy::ModelState
) -> Result<Vec<Vec<String>>> {
    use colored::Colorize;

    // First section: node attributes.
    let mut attributes = Vec::new();
    let proto_node = graph
        .get_node()
        .iter()
        .find(|n| n.get_name() == node.name)
        .unwrap();

    for attr in proto_node.get_attr() {
        attributes.extend(
            format::with_header(
                format!("Attribute {}:", attr.0.bold()),
                format!("{:?}", attr.1),
                80
            )
        );
    }

    let mut inputs = Vec::new();

    for (ix, &(n, i)) in node.inputs.iter().enumerate() {
        let data = &state.outputs[n].as_ref().unwrap()[i.unwrap_or(0)];
        inputs.extend(
            format::with_header(
                format!(
                    "{} ({}/{}):",
                    format!("Input {}", ix).bold(),
                    n,
                    i.unwrap_or(0),
                ),
                data.partial_dump(false).unwrap(),
                80
            )
        );
    }

    Ok(vec![attributes, inputs])
}