use prettytable as pt;
use prettytable::format::consts::*;
use prettytable::format::FormatBuilder;
use textwrap;
use tfdeploy;
use tfdeploy::tfpb;
use tfdeploy::tfpb::graph::GraphDef;
use tfdeploy::ModelState;
use tfdeploy::Node;

use format;

/// A single row, which has either one or two columns.
/// The two-column layout is usually used when displaying a header and some content.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Row {
    Simple(String),
    Double(String, String),
}

/// Prints a box containing arbitrary information.
fn print_box(id: String, op: String, name: String, status: String, sections: Vec<Vec<Row>>) {
    use colored::Colorize;

    // Common formats
    let format_none = FormatBuilder::new().build();
    let format_no_right_border = FormatBuilder::new()
        .column_separator('|')
        .left_border('|')
        .separators(&[pt::format::LinePosition::Top,
                      pt::format::LinePosition::Bottom],
                    pt::format::LineSeparator::new('-', '+', '+', '+'))
        .padding(1, 1)
        .build();

    // Node identifier
    let mut count = table!([
        format!("{:^5}", id.bold())
    ]);

    count.set_format(format_no_right_border);

    // Node name
    let mut name_table = table!([
        "Name: ",
        format!("{:67}", textwrap::fill(name.as_str(), 67))
    ]);

    name_table.set_format(format_none);

    // Table header
    let mut header = table!([
        format!("Operation: {:15}", op.bold().blue()),
        name_table,
        format!(" {:^26}", status.bold())
    ]);

    header.set_format(*FORMAT_NO_BORDER);

    // Content of the table
    let mut right = table![[header]];

    for section in sections {
        if section.len() < 1 {
            continue;
        }

        let mut outer = table!();
        outer.set_format(format_none);

        for row in section {
            let mut inner = match row {
                Row::Simple(content) => table!([
                    textwrap::fill(content.as_str(), 105)
                ]),
                Row::Double(header, content) => table!([
                    format!("{} ", header),
                    textwrap::fill(content.as_str(), 105)
                ])
            };

            inner.set_format(format_none);
            outer.add_row(row![inner]);
        }

        right.add_row(row![outer]);
    }

    // Whole table
    let mut table = table!();
    table.set_format(format_none);
    table.add_row(row![count, right]);
    table.printstd();
}

/// Returns information about a node.
fn node_info(
    node: &tfdeploy::Node,
    graph: &tfpb::graph::GraphDef,
    state: &::tfdeploy::ModelState,
) -> Vec<Vec<Row>> {
    use colored::Colorize;

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
                format!("Tensor: {:?} {:?}", tensor.get_dtype(), tensor.get_tensor_shape().get_dim())
            } else {
                format!("{:?}", attr.1)
            }
        ));
    }

    let mut inputs = Vec::new();

    for (ix, &(n, i)) in node.inputs.iter().enumerate() {
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

    vec![attributes, inputs]
}

/// Prints information about a node.
pub fn print_node(
    node: &Node,
    graph: &GraphDef,
    state: &ModelState,
    status: String,
    sections: Vec<Vec<Row>>,
) {
    format::print_box(
        node.id.to_string(),
        node.op_name.to_string(),
        node.name.to_string(),
        status,
        [format::node_info(&node, &graph, &state), sections].concat(),
    );
}


/// Prints some text with a line underneath.
pub fn print_header(text: String, color: &str) {
    use colored::Colorize;

    println!("{}", text.bold().color(color));
    println!("{}", format!("{:=<1$}", "", text.len()).bold().color(color));
}