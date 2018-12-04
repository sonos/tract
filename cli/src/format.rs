use std::borrow::Borrow;
use std::cmp::min;

use prettytable as pt;
use prettytable::format::{FormatBuilder, TableFormat};
use prettytable::Table;
use terminal_size::{terminal_size, Width};
use textwrap;
use tract_core;
use tract_core::plan::{SimplePlan, SimpleState};
use tract_core::{Model, Node};

use colored::Colorize;
use format;
use rusage::Duration;

use itertools::Itertools;

use SomeGraphDef;

/// A single row, which has either one or two columns.
/// The two-column layout is usually used when displaying a header and some content.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Row {
    Simple(String),
    Double(String, String),
}

/// Returns a table format with no borders or padding.
fn format_none() -> TableFormat {
    FormatBuilder::new().build()
}

/// Returns a table format with only inter-column borders.
fn format_only_columns() -> TableFormat {
    FormatBuilder::new().column_separator('|').build()
}

/// Returns a table format with no right border.
fn format_no_right_border() -> TableFormat {
    FormatBuilder::new()
        .column_separator('|')
        .left_border('|')
        .separators(
            &[
                pt::format::LinePosition::Top,
                pt::format::LinePosition::Bottom,
            ],
            pt::format::LineSeparator::new('-', '+', '+', '+'),
        ).padding(1, 1)
        .build()
}

/// Builds a box header containing the operation, name and status on the same line.
fn build_header(cols: usize, op: &str, name: &str, status: Option<impl AsRef<str>>) -> Table {
    let mut header = if let Some(status) = status {
        let mut name_table = table!([
            " Name: ",
            format!("{:1$}", textwrap::fill(name.as_ref(), cols - 68), cols - 68),
        ]);

        name_table.set_format(format_none());
        table!([
            format!(
                "Op: {:15}",
                if op.starts_with("Unimplemented(") {
                    op.bold().red()
                } else {
                    op.bold().blue()
                }
            ),
            name_table,
            format!(" {:^33}", status.as_ref().bold()),
        ])
    } else {
        let mut name_table = table!([
            " Name: ",
            format!("{:1$}", textwrap::fill(name.as_ref(), cols - 46), cols - 46),
        ]);

        name_table.set_format(format_none());
        table!([format!("Operation: {:15}", op.bold().blue()), name_table])
    };

    header.set_format(format_only_columns());
    table![[header]]
}

/// Builds a box header conntaining the operation and name on one line, and a list
/// of status messages on the other.
fn build_header_wide(cols: usize, op: &str, name: &str, status: &[impl AsRef<str>]) -> Table {
    let mut name_table = table!([
        " Name: ",
        format!("{:1$}", textwrap::fill(name.as_ref(), cols - 46), cols - 46),
    ]);

    name_table.set_format(format_none());

    let mut header = table!([
        format!(
            "Operation: {:15}",
            if op.starts_with("Unimplemented(") {
                op.bold().red()
            } else {
                op.bold().blue()
            },
        ),
        name_table,
    ]);
    header.set_format(format_only_columns());

    let mut t = table![[header]];
    if status.len() > 0 {
        let status = pt::Row::new(
            status
                .iter()
                .map(|s| pt::Cell::new_align(s.as_ref(), pt::format::Alignment::CENTER))
                .collect(),
        );
        t.add_row(status);
    }
    t
}

/// Prints a box containing arbitrary information.
pub fn print_box(
    id: &str,
    op: &str,
    name: &str,
    status: &[impl AsRef<str>],
    sections: Vec<Vec<Row>>,
) {
    // Terminal size
    let cols = if let Ok(cols) = ::std::env::var("COLUMNS") {
        cols.parse().expect("Can not parse COLUMNS as an integer")
    } else {
        match terminal_size() {
            Some((Width(w), _)) => min(w as usize, 120),
            None => 80,
        }
    };

    // Node identifier
    let mut count = table!([format!("{:^5}", id.bold())]);

    count.set_format(format_no_right_border());

    // Content of the table
    let mut right = if status.len() == 1 && status[0].as_ref().len() < 15 {
        build_header(cols, op, name, Some(&status[0]))
    } else {
        build_header_wide(cols, op, name, status)
    };

    for section in sections {
        if section.len() < 1 {
            continue;
        }

        let mut outer = table!();
        outer.set_format(format_none());

        for row in section {
            let mut inner = match row {
                Row::Simple(content) => table!([textwrap::fill(content.as_str(), cols - 30)]),
                Row::Double(header, content) => table!([
                    format!("{} ", header),
                    textwrap::fill(content.as_str(), cols - 30),
                ]),
            };

            inner.set_format(format_none());
            outer.add_row(row![inner]);
        }

        right.add_row(row![outer]);
    }

    // Whole table
    let mut table = table!();
    table.set_format(format_none());
    table.add_row(row![count, right]);
    table.printstd();
}

/// Returns information about a node.
fn node_info<M, P>(
    node: &tract_core::Node,
    graph: &SomeGraphDef,
    state: Option<&SimpleState<M, P>>,
) -> Vec<Vec<Row>>
where
    M: Borrow<Model>,
    P: Borrow<SimplePlan<M>>,
{
    // First section: node attributes.
    let mut attributes = Vec::new();
    if let SomeGraphDef::Tf(graph) = graph {
        let proto_node = graph.get_node().iter().find(|n| n.get_name() == node.name);

        if let Some(proto_node) = proto_node {
            for attr in proto_node.get_attr().iter().sorted_by_key(|a| a.0) {
                attributes.push(Row::Double(
                    format!("Attribute {}:", attr.0.bold()),
                    if attr.1.has_tensor() {
                        let tensor = attr.1.get_tensor();
                        format!(
                            "Tensor: {:?} {:?}",
                            tensor.get_dtype(),
                            tensor.get_tensor_shape().get_dim()
                        )
                    } else {
                        format!("{:?}", attr.1)
                    },
                ));
            }
        }
    }

    let mut inputs = Vec::new();

    for (ix, outlet) in node.inputs.iter().enumerate() {
        if let Some(state) = state {
            let data = &state.values[outlet.node].as_ref().unwrap()[outlet.slot];
            inputs.push(Row::Double(
                format!(
                    "{} ({}/{}):",
                    format!("Input {}", ix).bold(),
                    outlet.node,
                    outlet.slot
                ),
                data.dump(false).unwrap(),
            ));
        }
    }

    vec![attributes, inputs]
}

/// Prints information about a node.
pub fn print_node<M, P>(
    node: &Node,
    graph: &SomeGraphDef,
    state: Option<&SimpleState<M, P>>,
    status: &[impl AsRef<str>],
    sections: Vec<Vec<Row>>,
) where
    M: Borrow<Model>,
    P: Borrow<SimplePlan<M>>,
{
    format::print_box(
        &format!("{}", node.id),
        &node.op.name(),
        &node.name,
        &status,
        [format::node_info(&node, &graph, state), sections].concat(),
    );
}

/// Prints some text with a line underneath.
pub fn print_header(text: String, color: &str) {
    println!("{}", text.bold().color(color));
    println!("{}", format!("{:=<1$}", "", text.len()).bold().color(color));
}

/// Format a rusage::Duration showing avgtime in ms.
pub fn dur_avg_oneline(measure: Duration) -> String {
    format!(
        "Real: {} User: {} Sys: {}",
        format!("{:.3} ms/i", measure.avg_real() * 1e3)
            .white()
            .bold(),
        format!("{:.3} ms/i", measure.avg_user() * 1e3)
            .white()
            .bold(),
        format!("{:.3} ms/i", measure.avg_sys() * 1e3)
            .white()
            .bold()
    )
}

/// Format a rusage::Duration showing avgtime in ms.
pub fn dur_avg_multiline(measure: Duration) -> String {
    format!(
        "Real: {}\nUser: {}\nSys: {}",
        format!("{:.3} ms/i", measure.avg_real() * 1e3)
            .white()
            .bold(),
        format!("{:.3} ms/i", measure.avg_user() * 1e3)
            .white()
            .bold(),
        format!("{:.3} ms/i", measure.avg_sys() * 1e3)
            .white()
            .bold()
    )
}

/// Format a rusage::Duration showing avgtime in ms, with percentage to a global
/// one.
pub fn dur_avg_oneline_ratio(measure: Duration, global: Duration) -> String {
    format!(
        "Real: {} {} User: {} {} Sys: {} {}",
        format!("{:7.3} ms/i", measure.avg_real() * 1e3)
            .white()
            .bold(),
        format!("{:2.0}%", measure.avg_real() / global.avg_real() * 100.)
            .yellow()
            .bold(),
        format!("{:7.3} ms/i", measure.avg_user() * 1e3)
            .white()
            .bold(),
        format!("{:2.0}%", measure.avg_user() / global.avg_user() * 100.)
            .yellow()
            .bold(),
        format!("{:7.3} ms/i", measure.avg_sys() * 1e3)
            .white()
            .bold(),
        format!("{:2.0}%", measure.avg_sys() / global.avg_sys() * 100.)
            .yellow()
            .bold(),
    )
}
