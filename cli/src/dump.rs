use Parameters;
use errors::*;
use format;

use utils::generate_json;

pub fn handle(params: Parameters, web: bool) -> Result<()> {
    let tfd = params.tfd_model;
    let output = tfd.get_node_by_id(params.output)?;
    let plan = output.eval_order(&tfd)?;

    if web {
        let data = generate_json(&tfd)?;
        ::web::open_web(data);
    } else {
        for n in plan {
            let node = tfd.get_node_by_id(n)?;
            format::print_node(
                node,
                &params.graph,
                None,
                vec![],
                vec![],
            );
        }
    }

    Ok(())
}

