use prettytable as pt;
use prettytable::format::consts::*;
use prettytable::format::FormatBuilder;
use prettytable::format::TableFormat;
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

/// A few table formatting rules.
lazy_static! {
    static ref FORMAT_NONE: TableFormat = FormatBuilder::new()
        .build();

    static ref FORMAT_NO_RIGHT_BORDER: TableFormat = FormatBuilder::new()
        .column_separator('|')
        .left_border('|')
        .separators(&[pt::format::LinePosition::Top,
                      pt::format::LinePosition::Bottom],
                    pt::format::LineSeparator::new('-', '+', '+', '+'))
        .padding(1, 1)
        .build();
}

/// Prints a box containing arbitrary information.
fn print_box(id: String, op: String, name: String, status: String, sections: Vec<Vec<Row>>) {
    use colored::Colorize;

    // Node counter
    let mut count = table!([
        format!("{:^5}", id.bold())
    ]);

    count.set_format(*FORMAT_NO_RIGHT_BORDER);

    // Table header
    let mut header = table!([
        format!("Operation: {:40}", op.bold().blue()),
        format!("Name: {:40}", name.bold()),
        format!("{:^24}", status.bold())
    ]);

    header.set_format(*FORMAT_NO_BORDER);

    // Content of the table
    let mut right = table![[header]];

    for section in sections {
        if section.len() < 1 {
            continue;
        }

        let mut outer = table!();
        outer.set_format(*FORMAT_NONE);

        for row in section {
            let mut inner = match row {
                Row::Simple(content) => table!([
                    textwrap::fill(content.as_str(), 100)
                ]),
                Row::Double(header, content) => table!([
                    format!("{} ", header),
                    textwrap::fill(content.as_str(), 100)
                ])
            };

            inner.set_format(*FORMAT_NONE);
            outer.add_row(row![inner]);
        }

        right.add_row(row![outer]);
    }

    // Whole table
    let mut table = table!();
    table.set_format(*FORMAT_NONE);
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
            format!("{:?}", attr.1),
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
