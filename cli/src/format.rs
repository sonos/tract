use std::cmp::min;

use prettytable as pt;
use prettytable::Table;
use prettytable::format::{TableFormat, FormatBuilder};
use terminal_size::{terminal_size, Width};
use textwrap;
use tfdeploy;
use tfdeploy::tfpb;
use tfdeploy::tfpb::graph::GraphDef;
use tfdeploy::ModelState;
use tfdeploy::Node;

use format;
use rusage::Duration;
use colored::Colorize;

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
        )
        .padding(1, 1)
        .build()
}

/// Builds a box header containing the operation, name and status on the same line.
fn build_header(cols: usize, op: String, name: String, status: Option<String>) -> Table {
    let mut header = if let Some(status) = status {
        let mut name_table = table!([
            " Name: ",
            format!("{:1$}", textwrap::fill(name.as_str(), cols - 68), cols - 68),
        ]);

        name_table.set_format(format_none());
        table!([
            format!("Operation: {:15}", op.bold().blue()),
            name_table,
            format!(" {:^33}", status.bold()),
        ])
    } else {
        let mut name_table = table!([
            " Name: ",
            format!("{:1$}", textwrap::fill(name.as_str(), cols - 46), cols - 46),
        ]);

        name_table.set_format(format_none());
        table!([
            format!("Operation: {:15}", op.bold().blue()),
            name_table
        ])
    };

    header.set_format(format_only_columns());
    table![[header]]
}

/// Builds a box header conntaining the operation and name on one line, and a list
/// of status messages on the other.
fn build_header_wide(cols: usize, op: String, name: String, status: Vec<String>) -> Table {
    let mut name_table = table!([
        " Name: ",
        format!("{:1$}", textwrap::fill(name.as_str(), cols - 46), cols - 46),
    ]);

    name_table.set_format(format_none());

    let status = pt::row::Row::new(status.iter().map(|s| pt::cell::Cell::new_align(s, pt::format::Alignment::CENTER)).collect());

    let mut header = table!([
        format!("Operation: {:15}", op.bold().blue()),
        name_table,
    ]);
    header.set_format(format_only_columns());

    let mut t = table![[header]];
    t.add_row(status);
    t
}

/// Prints a box containing arbitrary information.
fn print_box(id: String, op: String, name: String, mut status: Vec<String>, sections: Vec<Vec<Row>>) {
    // Terminal size
    let cols = match terminal_size() {
        Some((Width(w), _)) => min(w as usize, 120),
        None => 80,
    };

    // Node identifier
    let mut count = table!([format!("{:^5}", id.bold())]);

    count.set_format(format_no_right_border());

    // Content of the table
    let mut right = if status.len() < 2 && status[0].len() < 10 {
        build_header(cols, op, name, status.pop())
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
fn node_info(
    node: &tfdeploy::Node,
    graph: &tfpb::graph::GraphDef,
    state: Option<&::tfdeploy::ModelState>,
) -> Vec<Vec<Row>> {
    // First section: node attributes.
    let mut attributes = Vec::new();
    let proto_node = graph
        .get_node()
        .iter()
        .find(|n| n.get_name() == node.name)
        .unwrap();

    for attr in proto_node.get_attr() {
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

    let mut inputs = Vec::new();

    for (ix, &(n, i)) in node.inputs.iter().enumerate() {
        if let Some(state) = state {
            let data = &state.outputs[n].as_ref().unwrap()[i.unwrap_or(0)];
            inputs.push(Row::Double(
                format!(
                    "{} ({}/{}):",
                    format!("Input {}", ix).bold(),
                    n,
                    i.unwrap_or(0),
                ),
                data.partial_dump(false).unwrap(),
            ));
        }
    }

    vec![attributes, inputs]
}

/// Prints information about a node.
pub fn print_node(
    node: &Node,
    graph: &GraphDef,
    state: Option<&ModelState>,
    status: Vec<String>,
    sections: Vec<Vec<Row>>,
) {
    format::print_box(
        node.id.to_string(),
        node.op_name.to_string(),
        node.name.to_string(),
        status,
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
    format!("Real: {} User: {} Sys: {}",
        format!("{:.3} ms/i", measure.avg_real() * 1e3).white().bold(),
        format!("{:.3} ms/i", measure.avg_user() * 1e3).white().bold(),
        format!("{:.3} ms/i", measure.avg_sys() * 1e3).white().bold())
}

/// Format a rusage::Duration showing avgtime in ms, with percentage to a global
/// one.
pub fn dur_avg_oneline_ratio(measure: Duration, global:Duration) -> String {
    format!("Real: {} {} User: {} {} Sys: {} {}",
        format!("{:7.3} ms/i", measure.avg_real() * 1e3).white().bold(),
        format!("{:2.0}%", measure.avg_real() / global.avg_real() * 100.).yellow().bold(),
        format!("{:7.3} ms/i", measure.avg_user() * 1e3).white().bold(),
        format!("{:2.0}%", measure.avg_user() / global.avg_user() * 100.).yellow().bold(),
        format!("{:7.3} ms/i", measure.avg_sys() * 1e3).white().bold(),
        format!("{:2.0}%", measure.avg_sys() / global.avg_sys() * 100.).yellow().bold(),
        )
}

