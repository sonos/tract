use super::Downsample;
use crate::internal::*;
use crate::ops;

pub fn pull_downsample_over_scan(
    model: &TypedModel,
    scan_node: &TypedNode,
    scan_op: &ops::scan::Typed,
    down_node: &TypedNode,
    down_op: &Downsample,
) -> TractResult<Option<TypedModelPatch>> {
    /*
    let mut inner_model = scan_op.body.clone();
    let downsampled_outputs = scan_op
        .body
        .output_outlets()?
        .into_iter()
        .enumerate()
        .map(|(ix, output)| {
            let fact = scan_op.body.output_fact(ix)?;
            let ds = inner_model.chain_after(
                *output,
                format!("{}-{}", &down_node.name, ix),
                down_op.clone(),
                tvec!(down_op.transform_fact(fact)?),
            )?;
            Ok(OutletId::new(ds, 0))
        })
        .collect::<TractResult<Vec<_>>>()?;
    inner_model.set_output_outlets(&downsampled_outputs)?;

    let mut new_scan = scan_op.clone();
    new_scan.body = inner_model;

    let mut patch = TypedModelPatch::default();
    patch.tap_model(model, scan_node.inputs[0])?;
    let id = patch.chain(scan_node.name.clone(), new_scan, model.node_input_facts(scan_node.id)?.into_iter().cloned().collect())?;
    patch.shunt_outside(OutletId::new(down_node.id, 2), OutletId::new(id, 2))?;
    Ok(Some(patch))
    */
    return Ok(None)
}
