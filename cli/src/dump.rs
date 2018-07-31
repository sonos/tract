use { Parameters, WebParameters };
use errors::*;
use format;

pub fn handle(params: Parameters, web: Option<WebParameters>) -> Result<()> {
    let tfd = params.tfd_model;
    let output = tfd.get_node_by_id(params.output_node_id)?;
    let plan = output.eval_order(&tfd)?;

    if let Some(web) = web {
        ::web::open_web(&tfd, &web)?
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

